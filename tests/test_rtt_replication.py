import copy
import json
from pathlib import Path
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from tests.test_network_forecast import dataset
from scripts.rtt_replication import validate,rtt_points,fit_baselines,predict_rule,rtt_comparison,current_design
from scripts.network_forecast import fit_baseline,summarize,SCHEMA


def test_fresh_split_and_fixed_model_coverage():
    c=json.loads(Path('experiments/rtt-replication-v1.json').read_text());s=validate(c,'replication')
    assert len(s['cases'])==96 and len(c['frozen_model_sha256'])==9
    c['profiles']['engineering']['cases'][0]=s['cases'][0]
    with pytest.raises(ValueError,match='overlap'):validate(c,'replication')


def test_raw_persistence_is_exact_and_trend_is_bounded():
    d=dataset([1,2]);d['history'][:,:,SCHEMA.index('rtt')]=np.linspace(.2,1.,64)
    raw=rtt_points(d,'raw-persistence');assert np.allclose(np.expm1(raw)*.1,.1)
    trend=np.expm1(rtt_points(d,'bounded-trend'))*.1
    assert (trend>=0).all() and (trend<=.2+1e-8).all()
    changed=copy.deepcopy(d);changed['y'][:]=999;changed['seed'][:]=999;changed['family'][:]='unseen'
    assert np.array_equal(rtt_points(d,'bounded-trend'),rtt_points(changed,'bounded-trend'))


def test_fits_are_finite_and_heldout_labels_do_not_affect_predictions():
    train=dataset([1,1,2,2]);ewma=fit_baseline(train,'ewma');fits=fit_baselines(train,dict(ridge_penalty=.1),ewma)
    test=dataset([3,3,4,4]);p=predict_rule(test,'ridge-current',fits['ridge-current'],ewma)
    changed=copy.deepcopy(test);changed['y'][:]=100;changed['srtt_seconds'][:]=100
    assert np.array_equal(p,predict_rule(changed,'ridge-current',fits['ridge-current'],ewma))
    assert np.isfinite(p).all()
    raw=predict_rule(test,'raw-persistence',fits['raw-persistence'],ewma)
    assert np.allclose(np.expm1(raw[:,:,4])*.1,.1)
    assert (raw[:,:,3]<=raw[:,:,4]).all() and (raw[:,:,4]<=raw[:,:,5]).all()


def test_rtt_comparison_clusters_cases_and_ignores_unlabelled_cases():
    d=dataset([1,1,1,2]);ewma=fit_baseline(d,'ewma');fits=fit_baselines(d,dict(ridge_penalty=.1),ewma)
    p=predict_rule(d,'raw-persistence',fits['raw-persistence'],ewma);base=summarize(d,p)
    other=copy.deepcopy(base)
    for table in other['per_case'].values():
        table['srtt_mae_seconds']={k:v+.002 for k,v in table['srtt_mae_seconds'].items()}
    other['case_family']['999']='no-labels';base['case_family']['999']='no-labels'
    c=rtt_comparison(other,base)
    assert c['mae_difference_seconds']==pytest.approx(.002)
    assert c['difference_ci95_seconds']==pytest.approx([.002,.002]) and c['cases_with_fresh_rtt']==2
