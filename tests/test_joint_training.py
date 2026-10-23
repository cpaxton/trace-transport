import json
import pytest
pytest.importorskip('torch')

from trace_transport.joint_training import train,JointTrainConfig
from trace_transport.joint_models import JointModelConfig


def test_joint_trainer_updates_both_roles_from_shared_rollouts(tmp_path):
    path=train(tmp_path,JointTrainConfig(decisions=200,environments=1,rollout=16,epochs=1,
        production_seconds=.2,drain_seconds=.5),JointModelConfig(history=4,width=32,layers=1))
    assert path.exists()
    logs=[json.loads(line) for line in (tmp_path/'training.jsonl').read_text().splitlines()]
    assert any(r['sender']['gradient_norm']>0 for r in logs)
    assert any(r['receiver']['gradient_norm']>0 for r in logs)
    assert json.loads((tmp_path/'progress.json').read_text())['phase']=='complete'
    with pytest.raises(ValueError,match='already exists'):
        train(tmp_path,JointTrainConfig(decisions=1,environments=1))


def test_resume_preserves_optimizer_budget_and_initial_policy(tmp_path):
    import torch
    cfg=JointTrainConfig(decisions=400,environments=1,rollout=8,epochs=1,
        checkpoint_every=1,production_seconds=.2,drain_seconds=.5)
    model=JointModelConfig(history=4,width=32,layers=1)
    train(tmp_path,cfg,model)
    snapshot=sorted(tmp_path.glob('d*.pt'))[0]
    original=(tmp_path/'initial.pt').read_bytes()
    before=torch.load(snapshot,weights_only=True)['metadata']['decisions']
    train(tmp_path,cfg,model,resume=snapshot)
    final=torch.load(tmp_path/'final.pt',weights_only=True)
    assert final['metadata']['decisions']>=400
    assert final['metadata']['resume']['decisions']==before
    assert final['metadata']['resume']['collector_reset']
    assert (tmp_path/'initial.pt').read_bytes()==original
    with pytest.raises(ValueError,match='already reached'):
        train(tmp_path,cfg,model,resume=tmp_path/'final.pt')
