from dataclasses import asdict
import numpy as np
import pytest
from trace_transport.goodput_distribution import FAMILIES,GoodputDistributionEnv,sample_case
from trace_transport.joint_workers import EnvironmentPool


def test_distribution_is_reproducible_and_covers_structural_strata():
    for family in FAMILIES:
        a=sample_case(42,family=family);b=sample_case(42,family=family)
        assert asdict(a[0])==asdict(b[0]) and asdict(a[1])==asdict(b[1]) and a[2]==b[2]
        cfg,w,label=a
        assert 1<=w.clients<=4 and max(w.image_bytes,w.map_bytes)<=w.byte_limit
        assert label['family']==family
        assert all(0<=t<12 for t,*_ in cfg.capacity_changes)
        assert all(x<y for x,y in zip([t for t,*_ in cfg.capacity_changes],[t for t,*_ in cfg.capacity_changes][1:]))
    assert sample_case(42,family='wifi-contention')[1].shared_airtime
    assert not sample_case(42,family='wired-lan')[1].shared_airtime
    assert sample_case(42,family='propagation-step')[0].rtt_changes
    assert not sample_case(42,family='propagation-step')[0].capacity_changes
    assert sample_case(42,family='outage-recovery')[0].outages
    with pytest.raises(ValueError):sample_case(42,family='invented')


def test_v3_worker_executes_new_profile_with_exact_delivery_kernel():
    pool=EnvironmentPool(1)
    try:
        o,_,label=pool.reset(0,42,.3,.2,'easy',False,
                            dict(rate='delay',initial_rate=512000),None,'goodput-v3')
        assert label['version']=='goodput-domain-v3.0'
        pool.pipes[0].send(('finish-baseline',None));metrics=pool.receive(0)
        assert metrics['profile']=='goodput-v3' and metrics['wrong']==0
    finally:pool.close()


def test_fast_reference_removes_old_floor_ceiling_and_metrics_reject_mixed_profiles():
    from trace_transport.goodput_evaluation import summarize
    cfg,w,_=sample_case(42,family='wired-lan')
    env=GoodputDistributionEnv(cfg,w);env.reset(seed=42)
    s=env.sessions[0];s.last_feedback=.01;s.srtt=s.min_rtt=.01;s.rate=2000000.
    assert env.baseline(0,'delay')[0]<0
    assert env.baseline(0,'delay-fast')[0]>0
    a=env.metrics();b=dict(a,profile='goodput-v2')
    assert summarize([a])['cases']==1
    with pytest.raises(ValueError):summarize([a,b])


def test_v3_training_initialization_and_receiver_exploration_are_recorded(tmp_path):
    pytest.importorskip('torch')
    from trace_transport.joint_training import train,JointTrainConfig
    from trace_transport.joint_models import JointModelConfig,load_checkpoint
    from trace_transport.joint_session import ACK_ACTIONS
    import math
    cfg=JointTrainConfig(profile='goodput-v3',teacher_rate='delay-fast',receiver_teacher_prior=.5,
                         decisions=2,environments=1,rollout=1,epochs=1,production_seconds=.1,drain_seconds=.1)
    train(tmp_path/'run',cfg,JointModelConfig(history=2,width=8,layers=1,heads=2))
    _,rx,_,meta=load_checkpoint(tmp_path/'run/initial.pt')
    bias=float(rx.head.bias[0].detach())
    assert math.exp(bias)/(math.exp(bias)+len(ACK_ACTIONS))==pytest.approx(.5)
    assert meta['domain']['version']=='goodput-domain-v3.0'
    assert 'goodput_distribution.py' in meta['source_sha256']
