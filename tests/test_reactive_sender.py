from dataclasses import replace
import math
import numpy as np
import pytest
from trace_transport.reactive_sender import PhysicalHistory,TargetBiasEnv,MacroWorld,FEATURES,KEEP
from trace_transport.goodput_reactive import GoodputReactiveEnv,sample_case
from trace_transport.joint_session import SENDER_FEATURES


def test_physical_bins_preserve_integrated_rates_and_do_not_forward_fill_instant_state():
    x=np.zeros((2,len(FEATURES)),np.float32);h=PhysicalHistory(x,seconds=1.,bins=4)
    post=x.copy();post[:,0]=10.;post[:,SENDER_FEATURES.index('ack_rate')]=math.log1p(64000./128000.)
    h.add(0.,.4,post,rate_window=.4)
    result,valid=h.snapshot();width=.25
    volume=(np.expm1(result[:,:,SENDER_FEATURES.index('ack_rate')])*128000.*width).sum(1)
    assert np.allclose(volume,64000.*.4,rtol=1e-6)
    assert result[0,-2,0]==0. # bin ends .15; the state at .4 is not yet observable.
    assert result[0,-1,0]==10. and valid[:,-1].all()


def test_bias_one_preserves_teacher_packet_outcomes():
    cfg,w,_=sample_case(42,production=.4,drain=.2,stage='easy')
    a=GoodputReactiveEnv(cfg,w);b=TargetBiasEnv(cfg,w);a.reset(seed=42);b.reset(seed=42)
    zero=(np.zeros((a.count,3)),np.zeros(a.count,int))
    while not a.done():
        x=a.step(*zero);y=b.step(*zero);assert x[1]==y[1]
    assert a.metrics()==b.metrics()


def test_target_bias_converges_instead_of_compounding_and_window_is_teacher():
    cfg,w,_=sample_case(42,production=.2,drain=.1,stage='easy');env=TargetBiasEnv(cfg,w);env.reset(seed=42)
    s=env.sessions[0];s.last_feedback=0.;s.srtt=.11;s.min_rtt=.01;s.rate=10000.;s.rate_control='delay-fast'
    teacher=env.baseline(0,'delay-fast');env.target_bias[0]=1.2
    for _ in range(200):
        a=env.baseline(0,'delay-fast');assert a[1]==teacher[1]
        s.rate*=math.exp(.2*float(a[0]))
    target=min(env.rate_limit,max(2000.,env.network.payload_size/(.5*.1)))*1.2
    assert s.rate==pytest.approx(target,rel=1e-6)
    assert env.baseline(0,'delay-fast')[0]==pytest.approx(0.,abs=1e-6)


def test_macro_masks_reverse_roles_and_retains_one_second_physical_history():
    world=MacroWorld(700000,production=.4,drain=.2);o=world.observation()
    assert not o['eligible'][1::2].any()
    wrong=np.full(world.env.count,KEEP);wrong[1]=1
    with pytest.raises(ValueError,match='ineligible'):world.step(wrong)
    while not world.env.done():
        result=world.step(np.full(world.env.count,KEEP))
        assert result[4]>0 and not result[0]['eligible'][1::2].any()
    assert result[0]['history'].shape==(world.env.count,64,len(FEATURES))
    assert np.isfinite(result[0]['history']).all()


def test_macro_teacher_matches_raw_v6_and_discounted_micro_return():
    world=MacroWorld(700000,production=.4,drain=.2);plain=world.env.clone();total=discounted=0.;elapsed=0.
    while not plain.done():
        before=plain.now;_,r,_,_,_=plain.step(np.zeros((plain.count,3)),np.zeros(plain.count,int))
        discounted+=.999**(before/.02)*r
    while not world.env.done():
        r=world.step(np.full(world.env.count,KEEP));total+=.999**(elapsed/.02)*r[1];elapsed+=r[4]
    assert world.env.metrics()==plain.metrics()
    assert total==pytest.approx(discounted,abs=1e-10)


def test_tiny_training_checkpoint_is_role_aligned_and_safely_loadable(tmp_path):
    torch=pytest.importorskip('torch')
    from trace_transport.reactive_sender_training import Config,train,load
    train(tmp_path/'run',Config(decisions=4,environments=1,rollout=2,epochs=1),[700000],{})
    actor,meta=load(tmp_path/'run/latest.pt')
    assert meta['macro_decisions']>=4 and meta['training']['decisions']==4
    train(tmp_path/'history',Config(decisions=1,environments=1,rollout=1,epochs=1,kind='history-mlp'),[700000],{})
    a=torch.load(tmp_path/'run/initial.pt',weights_only=True);b=torch.load(tmp_path/'history/initial.pt',weights_only=True)
    assert all(torch.equal(a['critic'][k],b['critic'][k]) for k in a['critic'])
    assert torch.equal(a['torch_rng'],b['torch_rng'])
    assert meta['actor']['samples']>0
    assert torch.isfinite(torch.cat([p.flatten() for p in actor.parameters()])).all()


@pytest.mark.parametrize('advance,latency,expected',[(False,.2,1),(True,.2,2),(True,30.,1)])
def test_run_plan_matches_effective_training_budget(tmp_path,monkeypatch,advance,latency,expected):
    pytest.importorskip('torch')
    import json,sys,hashlib
    from dataclasses import asdict
    from scripts import reactive_sender_pilot as runner
    monkeypatch.chdir(tmp_path);(tmp_path/'source-manifest.json').write_text('{}')
    config=dict(pilot_decisions=2,engineering_decisions=1,optimizer_seeds=[42],kinds=['current-mlp'],training_cases=[700000],validation_cases=[720000])
    (tmp_path/'config.json').write_text(json.dumps(config));(tmp_path/'gate.json').write_text(json.dumps({'advance':advance,'source_sha256':{},'config_sha256':hashlib.sha256((tmp_path/'config.json').read_bytes()).hexdigest()}))
    monkeypatch.setattr(sys,'argv',['pilot','--config',str(tmp_path/'config.json'),'--gate',str(tmp_path/'gate.json'),'--root',str(tmp_path/'run')])
    class Done(Exception):pass
    def capture(output,cfg,seeds,source,**kwargs):
        plan=json.loads((tmp_path/'run/plan.json').read_text())
        assert cfg.decisions==expected and plan['actual_training']==[asdict(cfg)]
        raise Done
    monkeypatch.setattr(runner,'inference_benchmark',lambda kinds:{k:{'warm_ms_p50_p95':[.1,latency]} for k in kinds})
    monkeypatch.setattr(runner,'train',capture)
    with pytest.raises(Done):runner.main()
