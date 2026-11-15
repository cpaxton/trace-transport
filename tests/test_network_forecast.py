import copy
import json
from pathlib import Path
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from scripts.network_forecast import (SCHEMA,HORIZONS,validate,future_targets,collect_case,load_split,
    case_weights,fit_baseline,predict_baseline,train_model,predict_model,summarize,comparison,analyze)


def dataset(seeds):
    n=len(seeds);x=np.zeros((n,64,len(SCHEMA)),np.float32);rate=50000.
    x[:,:,SCHEMA.index('ack_rate')]=np.log1p(rate/128000.)
    x[:,:,SCHEMA.index('rtt')]=np.log1p(.1/.001)
    y=np.zeros((n,3,3),np.float32);y[:,:,0]=np.log1p(rate/128000.);y[:,:,1]=np.log(2.)
    dt=np.broadcast_to(HORIZONS,(n,3)).copy()
    return dict(history=x,valid=np.ones((n,64),bool),y=y,mask=np.ones((n,3,3),bool),duration=dt,
        ack_bytes=(dt*rate).astype(np.int64),srtt_seconds=np.full((n,3),.1),backlog_fraction=np.ones((n,3)),
        seed=np.asarray(seeds),family=np.array(['steady']*n),anchor_time=np.arange(n),endpoint=np.zeros(n,int))


def test_banks_are_disjoint_and_clock_is_excluded():
    c=json.loads(Path('experiments/network-forecast-v1.json').read_text());s=validate(c,'pilot')
    assert [len(s['splits'][k]) for k in ['train','calibration','test']]==[96,24,48]
    assert 'age' not in SCHEMA and 'feedback_age' in SCHEMA
    c['profiles']['pilot']['splits']['test'][0]=s['splits']['train'][0]
    with pytest.raises(ValueError,match='leakage'):validate(c,'pilot')


def test_labels_are_exact_counter_deltas_and_fresh_rtt_is_masked():
    def frame(time,acked,samples):return dict(time=time,acked=np.array([acked]),samples=np.array([samples]),srtt=np.array([.2]),backlog=np.array([True]))
    trace=[frame(0,0,0),frame(.1,100,1),frame(.2,100,1)]
    y,mask,dt,fraction,raw=future_targets(trace,0,0,[.1,.2])
    assert raw[:,0].tolist()==[100,100] and mask[:,1].all() and fraction==[1.,1.]
    y,mask,_,_,raw=future_targets(trace,1,0,[.1])
    assert y[0,2]==1 and not mask[0,1] and y[0,1]==0 and raw[0,0]==0
    with pytest.raises(ValueError,match='incomplete'):future_targets(trace,0,0,[1.])


def test_collected_case_roundtrips_and_hash_tampering_fails(tmp_path):
    meta=collect_case(dict(seed=700006,history_seconds=.1,anchors=[.1],horizons=[.1,.2,.3],production=.4,drain=.2,output=str(tmp_path)))
    d=load_split(tmp_path,[meta]);assert len(d['seed'])>0 and d['history'].shape[-1]==len(SCHEMA)
    assert np.array_equal(d['y'][:,:,2],d['ack_bytes']==0)
    with (tmp_path/meta['file']).open('ab') as f:f.write(b'tampered')
    with pytest.raises(ValueError,match='hash'):load_split(tmp_path,[meta])


def test_case_weights_balance_endpoints_and_missing_rtt_labels():
    seeds=np.array([1,1,1,2]);mask=np.array([[1,1],[1,0],[1,0],[1,1]],bool)
    w=case_weights(seeds,mask)
    assert np.allclose(w.sum(0),1.) and np.allclose(w[seeds==1].sum(0),.5)
    assert (w[~mask]==0).all()


def test_baseline_training_residuals_and_exact_byte_metrics():
    d=dataset([1,1,2,2]);fit=fit_baseline(d,'persistence');p=predict_baseline(fit,d)
    s=summarize(d,p)
    assert s['summary']['1.0']['ack_mae_bytes']<.1
    assert s['summary']['1.0']['srtt_mae_seconds']<1e-6
    assert s['summary']['1.0']['ack_silence_brier']==0.
    c=comparison(s,s);assert c['difference_ci95']==[0.,0.]


def test_forecast_uses_no_test_labels_metadata_or_future_durations(tmp_path):
    d=dataset([1,1,2,2]);fitted=train_model(d,'current',1,2,tmp_path)
    before=predict_model(fitted,d);changed=copy.deepcopy(d)
    changed['y'][:]=100;changed['seed'][:]=999;changed['family'][:]='hidden';changed['duration'][:]=999
    after=predict_model(fitted,changed)
    assert np.array_equal(before,after)
    assert np.all(before[:,:,:3][:,:,1:]>=before[:,:,:3][:,:,:-1])
    assert np.all(before[:,:,3:6][:,:,1:]>=before[:,:,3:6][:,:,:-1])
    assert ((before[:,:,6]>=0)&(before[:,:,6]<=1)).all()


def test_analysis_reports_every_fixed_model_and_selects_baseline_only_on_calibration(tmp_path):
    data=dict(train=dataset([1,1,2,2]),calibration=dataset([3,3,4,4]),test=dataset([5,5,6,6]))
    r=analyze(data,dict(model_seeds=[1,2,3]),1,tmp_path)
    assert len(r['modes']['test'])==16 and len(r['modes']['calibration'])==16
    assert r['selected_baseline'] in ['constant','persistence','moving-average','ewma']
    assert r['primary']=='transformer-ensemble: 1-second ACK bytes'
    assert len(list(tmp_path.glob('*.pt')))==9 and r['accepted_win'] is False


def test_zero_delivery_can_be_included_in_forecast_interval():
    from scripts.network_forecast import ForecastNet
    d=dataset([1]);model=ForecastNet('current').eval()
    with torch.no_grad():
        model.head.weight.zero_();model.head.bias.copy_(torch.tensor([-2.,-1.,0.,-2.,-1.,0.,0.]*3))
    p=predict_model((model,torch.zeros(len(SCHEMA)),torch.ones(len(SCHEMA))),d)
    assert np.array_equal(p[0,:,:3],np.zeros((3,3)))
    assert (p[0,:,6]==.5).all()


def test_srtt_decoder_matches_actual_v6_observer():
    from trace_transport.reactive_sender import MacroWorld,FEATURES
    from scripts.network_forecast import decode_srtt
    world=MacroWorld(700006,production=.2,drain=.1)
    for srtt in [.002,.1,.5]:
        world.env.sessions[0].srtt=srtt
        encoded=world.env.observe()['sender'][0,FEATURES.index('rtt')]
        assert float(decode_srtt(encoded))==pytest.approx(srtt,rel=1e-6)
