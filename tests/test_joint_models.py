import numpy as np
import pytest
pytest.importorskip('torch')
import torch

from trace_transport.joint_session import JointTransportEnv,WorkloadConfig
from trace_transport.joint_models import JointModelConfig,SenderActor,ReceiverActor,CentralCritic,save_checkpoint,load_checkpoint
from trace_transport.joint_training import Histories


def test_both_local_actors_have_gradients_and_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(1);cfg=JointModelConfig(history=4,width=32,layers=1)
    env=JointTransportEnv(workload=WorkloadConfig(production_seconds=.1,drain_seconds=1,image_bytes=4096,map_bytes=1024))
    obs,_=env.reset(seed=1);obs,*_=env.step(np.zeros((2,3)),np.zeros(2,int))
    h=Histories(obs,cfg);h.append(obs)
    sh=torch.from_numpy(h.sender);rh=torch.from_numpy(h.receiver);valid=torch.from_numpy(h.valid)
    q=torch.from_numpy(obs['queue']);mask=torch.from_numpy(obs['queue_mask'])
    sender=SenderActor(cfg);receiver=ReceiverActor(cfg);critic=CentralCritic()
    action,latent,choice,logp=sender.sample(sh,valid,q,mask)
    again,entropy=sender.evaluate(sh,valid,q,mask,latent,choice)
    assert torch.allclose(logp,again)
    ra,rlog=receiver.sample(rh,valid);again_r,rentropy=receiver.evaluate(rh,valid,ra)
    assert torch.allclose(rlog,again_r)
    loss=-(again+again_r).mean()-.001*(entropy+rentropy).mean();loss.backward()
    assert sender.control.weight.grad.abs().sum()>0
    assert receiver.head.weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in list(sender.parameters())+list(receiver.parameters()))
    assert torch.isfinite(critic(torch.from_numpy(env.critic_state())[None])).all()
    path=tmp_path/'joint.pt';save_checkpoint(path,sender,receiver,critic,cfg,{'seed':1})
    s,r,loaded,meta=load_checkpoint(path)
    assert loaded==cfg and meta['seed']==1
    assert torch.equal(s.act(sh,valid,q,mask),sender.act(sh,valid,q,mask))
    assert torch.equal(r.act(rh,valid),receiver.act(rh,valid))


def test_sender_can_fully_cancel_a_baseline_cut():
    env=JointTransportEnv(workload=WorkloadConfig(production_seconds=.1,drain_seconds=1,image_bytes=1024,map_bytes=1024))
    env.reset(seed=1);s=env.sessions[0];s.rate_control='aimd';s.lost=1
    rate=s.rate
    actions=np.zeros((2,3));actions[0,0]=.5
    env.step(actions,np.zeros(2,int))
    assert s.rate==pytest.approx(rate)


def test_actor_export_omits_privileged_critic_and_training_state(tmp_path):
    from trace_transport.joint_models import export_actors
    cfg=JointModelConfig(history=4,width=32,layers=1)
    source=tmp_path/'training.pt';target=tmp_path/'actors.pt'
    save_checkpoint(source,SenderActor(cfg),ReceiverActor(cfg),CentralCritic(),cfg,{'seed':42,'torch_rng':torch.get_rng_state()})
    export_actors(source,target)
    data=torch.load(target,weights_only=True)
    assert 'critic' not in data and 'optimizers' not in data
    assert 'torch_rng' not in data['metadata']
    assert load_checkpoint(target)[2]==cfg
