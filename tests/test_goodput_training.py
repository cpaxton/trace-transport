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
