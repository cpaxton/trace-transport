import numpy as np
import pytest
pytest.importorskip('torch')
from trace_transport.live_sender import sender_observation,JointFileSender
from trace_transport.joint_session import JointTransportEnv,WorkloadConfig
from trace_transport.messages import Message
from trace_transport.joint_models import JointModelConfig,SenderActor,ReceiverActor,CentralCritic,save_checkpoint
from trace_transport.protocol import Sender


def test_live_features_match_isolated_session_observation():
    # With no periodic producers or opposite-role traffic, the live observation
    # uses the same normalizations and feature order as its simulated counterpart.
    env=JointTransportEnv(workload=WorkloadConfig(interactive_hz=20),auto_producers=False)
    env.reset(seed=3)
    env.admit(0,Message(b'payload','image'))
    session=env.sessions[0];message=next(iter(session.messages.values()))
    message.sender.rate=session.rate;message.sender.cwnd=session.window
    obs=env.observe()
    features,queue,mask=sender_observation(message.sender,0,'image',(0,0,0),.02)
    # Producer rates are known absent in the live single-file adapter.
    expected=obs['sender'][0].copy();expected[20:23]=0
    np.testing.assert_array_equal(features,expected)
    np.testing.assert_array_equal(queue[0],obs['queue'][0])
    np.testing.assert_array_equal(mask[0],obs['queue_mask'][0])


def test_initial_live_actor_follows_baseline_and_rate_limits_decisions(tmp_path):
    cfg=JointModelConfig(history=4,width=32,layers=1)
    path=tmp_path/'actor.pt'
    save_checkpoint(path,SenderActor(cfg),ReceiverActor(cfg),CentralCritic(),cfg,{})
    controller=JointFileSender(path,'image');sender=Sender(b'abc',1)
    initial=(sender.rate,sender.cwnd)
    controller.tick(sender,0);controller.tick(sender,.01)
    assert controller.decisions==1
    assert (sender.rate,sender.cwnd)==initial
    controller.tick(sender,.02)
    assert controller.decisions==2
    assert controller.valid.sum()==2
