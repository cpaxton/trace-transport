import pytest
pytest.importorskip('torch')
import torch
from trace_transport.learning import generalized_advantages,time_generalized_advantages


def test_time_gae_matches_legacy_at_reference_tick_and_masks_terminal_bootstrap():
    r=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
    v=torch.tensor([[.1,.2],[.3,.4],[.5,.6]])
    done=torch.tensor([[False,False],[True,False],[False,True]])
    b=torch.tensor([8.,100000.])
    a,returns=time_generalized_advantages(r,v,done,b,torch.full_like(r,.02),gamma=.99,lam=.95)
    expected,er=generalized_advantages(r,v,done,b,gamma=.99,lam=.95)
    assert torch.allclose(a,expected,atol=1e-5) and torch.allclose(returns,er,atol=1e-5)
    assert returns[-1,1]==r[-1,1]


def test_delayed_reward_trace_depends_on_elapsed_time_not_tick_count():
    # Same 200-ms interval before a terminal reward, with 20-ms versus 2-ms ticks.
    outputs=[]
    for dt in [.02,.002]:
        steps=round(.2/dt)
        rewards=torch.zeros(steps+1,1);rewards[-1]=1.
        terminal=torch.zeros_like(rewards,dtype=torch.bool);terminal[-1]=True
        a,_=time_generalized_advantages(rewards,torch.zeros_like(rewards),terminal,torch.zeros(1),
                                       torch.full_like(rewards,dt),gamma=.999,lam=.98)
        outputs.append(a[0,0])
    assert torch.allclose(outputs[0],outputs[1],atol=1e-5)


def test_time_gae_rejects_missing_or_invalid_transition_timing():
    r=torch.ones(2,1);done=torch.zeros_like(r,dtype=torch.bool)
    with pytest.raises(ValueError):time_generalized_advantages(r,r,done,torch.zeros(1),torch.zeros_like(r))
    with pytest.raises(ValueError):time_generalized_advantages(r,r,done,torch.zeros(1),torch.ones(1))
