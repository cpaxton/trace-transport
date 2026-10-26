from dataclasses import replace
import numpy as np
import pytest
from trace_transport.goodput_progress import GoodputProgressEnv
from trace_transport.goodput_reactive import GoodputReactiveEnv,sample_case
from trace_transport.joint_session import ACK_ACTIONS


def pending_world(kind,wait=.25):
    cfg,w,_=sample_case(42,production=.5,drain=.2,stage='easy')
    cfg=replace(cfg,initial_rate=2000.,control_interval=.02,bandwidth_bps=1000000.,feedback_bandwidth_bps=1000000.,base_rtt=.01,jitter=0.,loss=0.,ack_loss=0.)
    env=kind(cfg,w);env.reset(seed=42)
    slow=np.full(env.count,ACK_ACTIONS.index((32,wait))+1)
    tx=np.zeros((env.count,3))
    while env.now<.2:
        env.step(tx,slow)
        if any(s.pending for s in env.sessions):break
    assert any(s.pending for s in env.sessions)
    return env


def test_immediate_ack_action_flushes_existing_backlog_without_new_arrival_and_charges_wire():
    old=pending_world(GoodputProgressEnv);new=pending_world(GoodputReactiveEnv)
    assert old.now==new.now
    old_before=sum(s.feedback_wire for s in old.sessions);new_before=sum(s.feedback_wire for s in new.sessions)
    tx=np.zeros((new.count,3));rx=np.ones(new.count,int)
    _,r0,_,_,_=old.step(tx,rx);_,r1,_,_,_=new.step(tx,rx)
    assert any(s.pending for s in old.sessions)
    assert not any(s.pending for s in new.sessions)
    assert sum(s.feedback_wire for s in old.sessions)==old_before
    assert sum(s.feedback_wire for s in new.sessions)>new_before
    assert r1<r0  # No free off-book feedback: the raw wire penalty sees the flush.


def test_extended_wait_invalidates_original_timer_without_resetting_pending_age():
    env=pending_world(GoodputReactiveEnv,wait=.1)
    target=next((i,mid) for i,s in enumerate(env.sessions) for mid in s.pending)
    i,mid=target;oldest=env.sessions[i].pending_at[mid]
    # Extend the wait past the old 100-ms timer, without restarting its age.
    tx=np.zeros((env.count,3));rx=np.full(env.count,ACK_ACTIONS.index((16,.25))+1)
    count=env.sessions[i].emitted
    while env.now<oldest+.12:env.step(tx,rx)
    assert mid in env.sessions[i].pending
    while env.now<oldest+.25+.001:env.step(tx,rx)
    assert mid not in env.sessions[i].pending
    assert env.sessions[i].emitted==count+1


def test_unchanged_receiver_settings_preserve_original_transport_results():
    old=pending_world(GoodputProgressEnv);new=pending_world(GoodputReactiveEnv)
    tx=np.zeros((new.count,3));rx=np.full(new.count,ACK_ACTIONS.index((32,.25))+1)
    while not old.done():
        a=old.step(tx,rx);b=new.step(tx,rx);assert a[1]==b[1]
    x=old.metrics();y=new.metrics()
    assert all(x[k]==y[k] for k in x if k!='profile')


def test_v6_training_records_new_receiver_control_contract(tmp_path):
    pytest.importorskip('torch')
    from trace_transport.joint_training import train,JointTrainConfig
    from trace_transport.joint_models import JointModelConfig,load_checkpoint
    cfg=JointTrainConfig(profile='goodput-v6',time_aware_gae=True,progress_shaping=True,mask_receiver_alias=True,
        decisions=2,environments=1,rollout=1,epochs=1,production_seconds=.1,drain_seconds=.1)
    train(tmp_path/'run',cfg,JointModelConfig(history=2,width=8,layers=1,heads=2))
    _,_,_,meta=load_checkpoint(tmp_path/'run/final.pt')
    assert meta['domain']['version']=='goodput-domain-v6.0'
    assert 'pending ACKs' in meta['domain']['receiver_control']


def test_batch_change_without_eligibility_keeps_existing_timer_and_generation():
    env=pending_world(GoodputReactiveEnv)
    i,mid=next((i,mid) for i,s in enumerate(env.sessions) for mid in s.pending)
    generation=env.sessions[i].ack_generations.get(mid,0)
    event=next(e for e in env.events if e[2]=='flush' and e[3][:2]==(i,mid))
    tx=np.zeros((env.count,3));rx=np.full(env.count,ACK_ACTIONS.index((16,.25))+1)
    env.step(tx,rx)
    assert env.sessions[i].ack_generations.get(mid,0)==generation
    assert event in env.events
