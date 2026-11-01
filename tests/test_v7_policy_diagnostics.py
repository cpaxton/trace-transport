import json
from pathlib import Path
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from scripts.v7_policy_diagnostics import (actions,generators,gae_first,logit_gradient,
    difference_ci,build_tasks,conditional_rollout,run_episode,model,GAMMA,REFERENCE_SECONDS)
from trace_transport.reactive_sender import MacroWorld,KEEP
from trace_transport.reactive_sender_training import Policy,Critic


def test_sampling_is_local_reproducible_and_masks_reverse_roles():
    actor=Policy().eval();obs=MacroWorld(700006,production=.2,drain=.1).observation()
    a,_=actions(actor,obs,'sampled',generators(101,700006,len(obs['eligible'])))
    torch.manual_seed(987);torch.rand(100)
    b,_=actions(actor,obs,'sampled',generators(101,700006,len(obs['eligible'])))
    assert np.array_equal(a,b)
    assert (a[~obs['eligible']]==KEEP).all()
    assert (a[1::2]==KEEP).all()


def test_initial_greedy_replays_teacher():
    a=run_episode(700006,Policy().eval(),'greedy',0,production=.3,drain=.1)
    b=run_episode(700006,None,'teacher',0,production=.3,drain=.1)
    assert a['record']==b['record']


def test_full_lambda_one_telescopes_and_truncation_bootstraps():
    rewards=[1.,2.,3.];values=[.7,.9,1.2,0.];elapsed=[.1,.2,.1]
    discount=1.;ret=0.
    for r,dt in zip(rewards,elapsed):
        ret+=discount*r;discount*=GAMMA**(dt/REFERENCE_SECONDS)
    assert gae_first(rewards,values,elapsed,3,lam=1.)==pytest.approx(ret-values[0],rel=1e-6)
    expected=rewards[0]+GAMMA**(elapsed[0]/REFERENCE_SECONDS)*values[1]-values[0]
    assert gae_first(rewards,values,elapsed,1,lam=1.)==pytest.approx(expected,rel=1e-6)


def test_conditional_gradient_matches_softmax_finite_difference():
    logits=np.array([.3,1.1,-.4]);q=np.array([1.,-2.,4.])
    def score(x):
        p=np.exp(x-x.max());return np.dot(p/p.sum(),q)
    p=np.exp(logits-logits.max());p/=p.sum();eps=1e-5
    numerical=[(score(logits+np.eye(3)[i]*eps)-score(logits-np.eye(3)[i]*eps))/(2*eps) for i in range(3)]
    assert np.allclose(logit_gradient(p,q),numerical)
    with pytest.raises(ValueError):logit_gradient(p,[1.,float('nan'),3.])


def test_independent_arm_ci_and_fail_closed_coverage():
    assert difference_ci([3.,3.],[1.,1.],42)==[2.,2.]
    with pytest.raises(ValueError):difference_ci([3.],[1.,1.],42)
    cfg=json.loads(Path('experiments/v7-policy-diagnostics-v1.json').read_text())
    tasks=build_tasks(cfg,Path('/tmp/models'),'smoke')
    assert len(tasks)==22
    assert sum(t['type']=='credit' for t in tasks)==4
    cfg['profiles']['smoke']['sampling_seeds']=[101,101]
    with pytest.raises(ValueError,match='duplicate'):build_tasks(cfg,Path('/tmp/models'),'smoke')


def test_counterfactual_preserves_snapshot_and_returns_complete_trace():
    world=MacroWorld(700006,production=.3,drain=.1);actor=Policy().eval();critic=Critic().eval()
    original=world.observation()['history'].copy();time=world.env.now
    row=conditional_rollout(world,actor,critic,0,2,123,include_trace=True)
    assert world.env.now==time and np.array_equal(world.observation()['history'],original)
    trace=row['trace'];discount=1.;ret=0.
    for r,dt in zip(trace['rewards'],trace['elapsed']):
        ret+=discount*r;discount*=GAMMA**(dt/REFERENCE_SECONDS)
    assert row['discounted_return']==pytest.approx(ret)
    assert trace['values'][-1]==0. and np.isfinite(row['discounted_return'])
    with pytest.raises(ValueError,match='eligible'):conditional_rollout(world,actor,critic,1,2,123)


def test_credit_rejects_actor_only_and_source_mismatch(tmp_path):
    from trace_transport.reactive_sender import CONTRACT,FEATURES
    path=tmp_path/'actor.pt'
    torch.save(dict(format_version='reactive-sender-v7',contract=CONTRACT,features=list(FEATURES),kind='current-mlp',policy=Policy().state_dict(),metadata={}),path)
    with pytest.raises(ValueError,match='full private'):model(path,need_critic=True)
    with pytest.raises(ValueError,match='source mismatch'):model(path,expected_core={'src/example.py':'abc'})
