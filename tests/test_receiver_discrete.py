import pytest

pytest.importorskip('torch')
import torch

from trace_transport.learning import empty_history
from trace_transport.receiver_discrete import ACTIONS,BASELINE_INDEX,make_categorical
from trace_transport.receiver_learning import episode,save,load
from trace_transport.simulator import NetworkConfig


def test_baseline_is_exactly_representable_and_selected_initially(tmp_path):
    torch.set_num_threads(1)
    model=make_categorical(42)
    h,v=empty_history(2,model.config);v[:,-1]=True
    logits,_=model(torch.from_numpy(h),torch.from_numpy(v))
    assert (logits.argmax(-1)==BASELINE_INDEX).all()
    assert ACTIONS[BASELINE_INDEX]==(8,.025)
    path=tmp_path/'receiver.pt';save(path,model,{})
    restored=load(path)
    assert restored.feedback_actions==ACTIONS
    assert torch.equal(logits,restored(torch.from_numpy(h),torch.from_numpy(v))[0])
    cfg=NetworkConfig(loss=.03,ack_loss=.1,feedback_bandwidth_bps=64000)
    learned,score,_=episode(cfg,seed=123,model=restored,receipt_feedback=True)
    fixed,fixed_score,_=episode(cfg,seed=123,baseline=(8,.025),receipt_feedback=True)
    assert learned==fixed
    assert (score==fixed_score).all()


def test_categorical_ppo_has_finite_gradients():
    model=make_categorical(42)
    h,v=empty_history(2,model.config);v[:,-1]=True
    h,v=torch.from_numpy(h),torch.from_numpy(v)
    action,latent,logp,value=model.sample(h,v)
    again,entropy,newvalue=model.evaluate(h,v,latent)
    assert torch.allclose(logp,again)
    loss=-again.mean()+newvalue.square().mean()-.01*entropy.mean()
    loss.backward()
    assert model.actor.weight.grad is not None
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_wide_action_bank_roundtrips_and_requires_matching_delay_budget(tmp_path):
    from trace_transport.receiver_discrete import WIDE_ACTIONS
    model=make_categorical(42,wide=True)
    path=tmp_path/'wide.pt';save(path,model,{})
    restored=load(path)
    assert restored.feedback_actions==WIDE_ACTIONS
    with pytest.raises(ValueError,match='delay budget'):
        episode(NetworkConfig(),seed=1,model=restored,receipt_feedback=True)
    cfg=NetworkConfig(feedback_delay_budget=.25,feedback_bandwidth_bps=64000)
    result,score,_=episode(cfg,seed=1,model=restored,receipt_feedback=True)
    baseline,base_score,_=episode(cfg,seed=1,baseline=(8,.25),receipt_feedback=True)
    assert result==baseline and (score==base_score).all()
