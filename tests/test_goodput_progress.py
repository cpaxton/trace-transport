from dataclasses import replace
import numpy as np
import pytest
from trace_transport.goodput_progress import GoodputProgressEnv,sample_case
from trace_transport.goodput_deployment import GoodputDeploymentEnv
from trace_transport.learning_diagnostics import optimistic_forward_bytes,completion_headroom
from trace_transport.simulator import NetworkConfig
from trace_transport.joint_session import WorkloadConfig


def test_service_bound_integrates_changes_and_accounts_for_minimum_wire_overhead():
    cfg=NetworkConfig(bandwidth_bps=8000,capacity_changes=((.5,16000.),))
    assert optimistic_forward_bytes(cfg,1.)==1500.
    cfg,w,_=sample_case(121007)
    bound=completion_headroom(cfg,w)
    assert bound['optimistic_wire_service_bytes']>bound['minimum_payload_for_all_hosts']
    assert bound['all_host_object_completion_provably_infeasible']
    assert bound['optimistic_max_hosts_with_bulk_object']==1


def test_progress_is_not_whole_object_delivery_and_shaping_cancels_at_terminal():
    cfg,w,_=sample_case(42,production=.4,drain=.1,stage='easy')
    w=replace(w,image_bytes=524288,map_bytes=2097152)
    plain=GoodputProgressEnv(cfg,w);shaped=GoodputProgressEnv(cfg,w,progress_shaping=True)
    plain.reset(seed=42);shaped.reset(seed=42)
    total_plain=total_shaped=0.;discount=1.;saw_potential=False
    tx=np.zeros((plain.count,3));rx=np.zeros(plain.count,int)
    while not plain.done():
        o,r,done,_,_=plain.step(tx,rx);_,s,_,_,_=shaped.step(tx,rx)
        total_plain+=discount*r;total_shaped+=discount*s
        discount*=.999**(o['transition_seconds']/.02)
        saw_potential=saw_potential or shaped.potential()>0
    assert saw_potential and shaped.potential()==0
    assert total_plain==pytest.approx(total_shaped,abs=1e-8)
    m=shaped.metrics()
    assert m['accepted_bulk_payload_bytes']>0 and m['verified_bulk_bytes']==0
    assert m['measurement_incomplete_bulk_payload_bytes']==m['accepted_bulk_payload_bytes']
    assert m['wrong']==0


def test_optional_training_shaping_does_not_change_transport_results():
    cfg,w,_=sample_case(42,production=.2,drain=.1,stage='easy')
    old=GoodputDeploymentEnv(cfg,w);new=GoodputProgressEnv(cfg,w)
    old.reset(seed=42);new.reset(seed=42)
    tx=np.zeros((old.count,3));rx=np.zeros(old.count,int)
    while not old.done():
        a=old.step(tx,rx);b=new.step(tx,rx)
        assert a[1]==b[1]
    before=old.metrics();after=new.metrics()
    assert all(before[k]==after[k] for k in before if k!='profile')


def test_v5_training_records_shaping_and_time_discount_contract(tmp_path):
    pytest.importorskip('torch')
    from trace_transport.joint_training import train,JointTrainConfig
    from trace_transport.joint_models import JointModelConfig,load_checkpoint
    with pytest.raises(ValueError):JointTrainConfig(profile='goodput-v5',progress_shaping=True)
    cfg=JointTrainConfig(profile='goodput-v5',progress_shaping=True,time_aware_gae=True,decisions=2,
                         environments=1,rollout=1,epochs=1,production_seconds=.1,drain_seconds=.1)
    train(tmp_path/'run',cfg,JointModelConfig(history=2,width=8,layers=1,heads=2))
    _,_,_,meta=load_checkpoint(tmp_path/'run/final.pt')
    assert meta['algorithm']=='mappo-time-gae-progress-v1'
    assert meta['baseline']['progress_shaping']


def test_partial_progress_guard_never_turns_zero_completed_goodput_into_a_win():
    from trace_transport.goodput_evaluation import summarize,compare
    cfg,w,_=sample_case(42,production=.2,drain=.1,stage='easy')
    env=GoodputProgressEnv(cfg,replace(w,image_bytes=524288,map_bytes=2097152));env.reset(seed=42)
    while not env.done():env.step(np.zeros((env.count,3)),np.zeros(env.count,int))
    row=env.metrics();row.update(seed=42,domain={'family':'steady'})
    s=summarize([row]);result=compare([row],[row])
    assert s['zero_completed_bulk_hosts']==1 and s['zero_accepted_bulk_hosts']==0
    assert result['guards']['hosts'] and not result['win']
    assert result['relative_goodput_gain'] is None


def test_receiver_teacher_alias_is_masked_and_survives_checkpoint_export(tmp_path):
    pytest.importorskip('torch')
    import torch
    from trace_transport.joint_training import train,JointTrainConfig
    from trace_transport.joint_models import JointModelConfig,load_checkpoint,export_actors
    from trace_transport.joint_session import RECEIVER_FEATURES
    cfg=JointTrainConfig(profile='goodput-v5',mask_receiver_alias=True,teacher_batch=1,teacher_delay=0.,
        receiver_teacher_prior=.5,time_aware_gae=True,decisions=2,environments=1,rollout=1,epochs=1,
        production_seconds=.1,drain_seconds=.1)
    train(tmp_path/'run',cfg,JointModelConfig(history=2,width=8,layers=1,heads=2))
    export_actors(tmp_path/'run/initial.pt',tmp_path/'actor.pt')
    _,receiver,_,_=load_checkpoint(tmp_path/'actor.pt')
    assert receiver.masked_choice==1
    h=torch.zeros(1,2,len(RECEIVER_FEATURES));valid=torch.ones(1,2,dtype=torch.bool)
    p=receiver(h,valid).softmax(-1)
    assert p[0,1]==0 and float(p[0,0].detach())==pytest.approx(.5)
