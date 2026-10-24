from dataclasses import asdict
import numpy as np
import pytest
from trace_transport.goodput_deployment import sample_case,GoodputDeploymentEnv
from trace_transport.joint_session import SENDER_FEATURES,RECEIVER_FEATURES
from scripts.receiver_counterfactual import compare_actions


def test_deployment_distribution_covers_fast_paths_and_exposes_control_timing():
    cfg,w,label=sample_case(42,family='wired-lan')
    assert .0003<=cfg.base_rtt<=.004 and 10000000<=cfg.bandwidth_bps<=1000000000
    env=GoodputDeploymentEnv(cfg,w);obs,_=env.reset(seed=42)
    assert np.allclose(obs['sender'][:,SENDER_FEATURES.index('age')],np.log1p(cfg.control_interval/.002))
    assert np.allclose(obs['receiver'][:,RECEIVER_FEATURES.index('age')],np.log1p(cfg.control_interval/.002))
    assert env.metrics()['profile']=='goodput-v4'
    cases=[sample_case(s,family='wired-lan')[0] for s in range(50)]
    assert min(c.base_rtt for c in cases)<.001
    assert max(c.bandwidth_bps for c in cases)>500000000
    assert {c.control_interval for c in cases}=={.002,.005,.02}


def test_counterfactual_clones_leave_original_world_unchanged():
    cfg,w,_=sample_case(42,production=.2,drain=.1,stage='easy')
    env=GoodputDeploymentEnv(cfg,w);env.reset(seed=42)
    before=env.snapshot();a=compare_actions(env,0,horizon=.04,choices=[0,0])
    assert a[0]==a[1]
    assert env.now==before['now'] and env.reward_counts==before['reward_counts']
    assert env.identity_rng.getstate()==before['identity_rng'].getstate()
    assert env.sequence==before['sequence']


def test_v4_training_records_new_profile_and_semantics(tmp_path):
    pytest.importorskip('torch')
    from trace_transport.joint_training import train,JointTrainConfig
    from trace_transport.joint_models import JointModelConfig,load_checkpoint
    cfg=JointTrainConfig(profile='goodput-v4',teacher_rate='delay-fast',decisions=2,environments=1,
                         rollout=1,epochs=1,production_seconds=.05,drain_seconds=.05)
    train(tmp_path/'run',cfg,JointModelConfig(history=2,width=8,layers=1,heads=2))
    _,_,_,metadata=load_checkpoint(tmp_path/'run/final.pt')
    assert metadata['profile']=='goodput-v4'
    assert 'observation_overrides' in metadata['domain']
    assert 'goodput_deployment.py' in metadata['source_sha256']


def test_divergent_actions_can_desynchronize_cloned_packet_impairments():
    from dataclasses import replace
    cfg,w,_=sample_case(42,production=.5,drain=.1,stage='easy')
    env=GoodputDeploymentEnv(replace(cfg,loss=.2,ack_loss=.2),w);env.reset(seed=42)
    other=env.clone()
    while env.now<.3:
        env.step(np.zeros((env.count,3)),np.zeros(env.count,int))
        a=np.zeros((other.count,3));a[:,:2]=1.
        other.step(a,np.ones(other.count,int))
    assert any(a.getstate()!=b.getstate() for a,b in zip(env.streams,other.streams))


def test_long_service_runtime_is_explicitly_bounded(tmp_path):
    from scripts.run_limited import command
    c=command('trace-caliban-long',['python3'],tmp_path/'log',directory=tmp_path,runtime_hours=72)
    assert '--property=RuntimeMaxSec=259200' in c
    with pytest.raises(ValueError):command('trace-too-long',['python3'],tmp_path/'log',directory=tmp_path,runtime_hours=73)
