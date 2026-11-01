import copy
import json
from pathlib import Path
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from scripts.v7_local_predictability import (validate,continuation,collect_case,fit_choices,
    summarize_choices,bank_values,tensors)
from trace_transport.reactive_sender import MacroWorld,FEATURES


def synthetic(seed,value=1.):
    h=np.zeros((64,len(FEATURES)));h[-1,0]=value
    def bank(offset):return [[dict(return_value=offset+a*value,verified=a,misses=0,wire=a) for _ in range(4)] for a in [0,1,2]]
    return dict(seed=seed,status='complete',history=h.tolist(),valid=[True]*64,banks=dict(fit=bank(0),score=bank(10)))


def test_splits_fail_closed():
    c=json.loads(Path('experiments/v7-local-predictability-v1.json').read_text())
    s=validate(c,'pilot');assert len(s['splits']['train'])==24 and len(s['splits']['test'])==12
    c['profiles']['pilot']['splits']['test'][0]=s['splits']['train'][0]
    with pytest.raises(ValueError,match='leakage'):validate(c,'pilot')


def test_no_score_or_metadata_leakage_into_fitting():
    train=[synthetic(i,(-1.)**i) for i in range(8)];test=[synthetic(i+20,(-1.)**i) for i in range(4)]
    first=fit_choices(train,test,'current',1,150)
    changed=copy.deepcopy(train);changed_test=copy.deepcopy(test)
    for s in changed+changed_test:
        s['banks']['score']=None;s['family']='hidden';s['future_capacity']=99999
    second=fit_choices(changed,changed_test,'current',1,150)
    assert first['test_choices']==second['test_choices']==[2,0,2,0]
    assert all(torch.equal(first['weights'][k],second['weights'][k]) for k in first['weights'])
    x,_=tensors(test);y,_=tensors(changed_test);assert torch.equal(x,y)


def test_independent_score_bank_and_case_weighting():
    states=[synthetic(1),synthetic(1),synthetic(2)]
    states[-1]['banks']['score'][2]=[dict(return_value=15.,verified=2,misses=0,wire=2)]*4
    report=summarize_choices(states,[2,2,2],draws=100)
    # Case 1 difference=1, case 2 difference=4; average cases, not snapshots.
    assert report['return_value_difference_from_teacher']==2.5
    assert report['return_difference_ci95']==[1.,4.]
    teacher=summarize_choices(states,[1,1,1],draws=100)
    assert teacher['return_difference_ci95']==[0.,0.]


def test_snapshot_clones_are_unchanged_and_complete_terminal_returns():
    w=MacroWorld(700006,production=.3,drain=.1);before=w.observation()['history'].copy()
    a=continuation(w,0,2,123);b=continuation(w,0,2,123)
    assert a==b and np.isfinite(a['return_value'])
    assert w.env.now==0 and np.array_equal(w.observation()['history'],before)
    with pytest.raises(ValueError,match='ineligible'):continuation(w,1,2,123)
    c=collect_case(dict(seed=700006,times=[.1],replicas=2,production=.3,drain=.1))
    assert c['complete'] and c['states'][0]['status']=='complete'
    assert len(c['states'][0]['banks']['fit'])==3
    assert all(len(a)==2 for a in c['states'][0]['banks']['score'])
    banks=c['states'][0]['banks']
    fit={r['replica_seed'] for arm in banks['fit'] for r in arm}
    score={r['replica_seed'] for arm in banks['score'] for r in arm}
    assert len(fit)==len(score)==6 and not fit&score


@pytest.mark.parametrize('kind',['history','transformer'])
def test_history_predictors_accept_masked_local_input(kind):
    train=[synthetic(i,(-1.)**i) for i in range(4)];test=[synthetic(10),synthetic(11,-1)]
    out=fit_choices(train,test,kind,1,3)
    assert len(out['test_choices'])==2 and np.isfinite(out['training_normalized_mse'])


def test_end_to_end_analysis_reports_all_fixed_models_and_controls(tmp_path):
    from scripts.v7_local_predictability import analyze
    data=dict(train=[dict(seed=i,states=[synthetic(i)]) for i in [1,2]],
              test=[dict(seed=i,states=[synthetic(i)]) for i in [10,11]])
    report=analyze(data,dict(model_seeds=[1,2,3],fit_steps=2),tmp_path)
    assert len(report['modes'])==12 and report['constant_action']==2
    assert report['modes']['teacher']['return_difference_ci95']==[0.,0.]
    assert report['modes']['snapshot-fit-table']['return_value_difference_from_teacher']==1.
    assert 'comparison_to_current' in report['modes']['transformer-1']
    assert len(list(tmp_path.glob('*.pt')))==9
    data['test'][0]['seed']=1
    with pytest.raises(ValueError,match='leakage'):analyze(data,dict(model_seeds=[1],fit_steps=1),tmp_path)
