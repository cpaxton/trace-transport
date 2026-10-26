import math
import pytest
pytest.importorskip('torch')
import torch
from trace_transport.joint_training import JointTrainConfig,train
from trace_transport.joint_models import JointModelConfig,load_checkpoint
from trace_transport.joint_session import ACK_ACTIONS,MAX_CANDIDATES


def test_v2_checkpoint_preserves_adaptive_teacher_and_sampling_prior(tmp_path):
    cfg=JointTrainConfig(profile='goodput-v2',teacher_rate='delay',teacher_initial_rate=512000,
        decisions=200,environments=1,rollout=16,epochs=1,production_seconds=.3,drain_seconds=.2)
    train(tmp_path,cfg,JointModelConfig(history=4,width=32,layers=1))
    sender,receiver,model,meta=load_checkpoint(tmp_path/'initial.pt')
    assert meta['baseline']['rate']=='delay' and meta['baseline']['initial_rate']==512000
    assert meta['profile']=='goodput-v2' and meta['source_sha256']
    prior=math.exp(float(receiver.head.bias[0].detach()))/(math.exp(float(receiver.head.bias[0].detach()))+len(ACK_ACTIONS))
    assert abs(prior-.9)<1e-6
    prior=math.exp(float(sender.follow.bias[0].detach()))/(math.exp(float(sender.follow.bias[0].detach()))+MAX_CANDIDATES)
    assert abs(prior-.9)<1e-6
    data=torch.load(tmp_path/'final.pt',weights_only=True)
    assert data['metadata']['decisions']>=200


def test_fixed_shape_goodput_inference_is_repeatable_across_worker_counts(tmp_path):
    from trace_transport.joint_evaluation import evaluate
    from trace_transport.joint_models import SenderActor,ReceiverActor,CentralCritic,save_checkpoint
    torch.manual_seed(4);cfg=JointModelConfig(history=4,width=32,layers=1)
    sender=SenderActor(cfg);receiver=ReceiverActor(cfg)
    with torch.no_grad():
        sender.control.weight[0,0]=.003;sender.control.bias[0]=.05
    path=tmp_path/'model.pt';save_checkpoint(path,sender,receiver,CentralCritic(),cfg,
        {'profile':'goodput-v2','baseline':{'rate':'delay','initial_rate':512000.,'scheduler':'edf','feedback':'budgeted'}})
    kwargs=dict(checkpoint=path,seeds=[71001,71002,71003],production=.4,drain=.2,profile='goodput-v2')
    assert evaluate(**kwargs,environments=1)==evaluate(**kwargs,environments=2)


def test_shadow_teacher_baseline_matches_zero_actions_without_actor_information_leak():
    import numpy as np
    from trace_transport.joint_workers import EnvironmentPool
    pool=EnvironmentPool(1)
    try:
        obs,critic,label=pool.reset(0,42,.2,.1,'easy',False,
            {'rate':'delay','scheduler':'edf','feedback':'budgeted','initial_rate':512000},None,'goodput-v2',True)
        for _ in range(20):
            actions=(np.zeros((len(obs['sender']),3)),np.zeros(len(obs['receiver']),dtype=int))
            pool.send_step(0,actions);obs,critic,reward,done,metrics,_=pool.receive(0)
            assert reward==0.
            if done:
                assert metrics['raw_training_return']==metrics['teacher_training_return']
                break
        else:pytest.fail('fixed-horizon episode did not terminate')
    finally:pool.close()
