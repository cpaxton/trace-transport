from dataclasses import replace
import json
from pathlib import Path
import pytest

torch=pytest.importorskip('torch')
from trace_transport.reactive_sender_training import Config,train,load
from trace_transport.learning import time_generalized_advantages
from scripts.v7_long_credit import configurations


def test_registered_factorial_and_invalid_coverage():
    c=json.loads(Path('experiments/v7-long-credit-v1.json').read_text())
    runs=configurations(c,'pilot');assert len(runs)==8
    assert {cfg.decisions for _,cfg in runs}=={50000}
    assert {cfg.seed for _,cfg in runs}=={42,43}
    assert {(cfg.rollout,cfg.gae_lambda) for _,cfg in runs}=={(16,.98),(16,.998),(64,.98),(64,.998)}
    c['arms'].pop()
    with pytest.raises(ValueError,match='2x2'):configurations(c,'pilot')


def test_longer_trace_retains_delayed_signal_and_terminal_stops_leakage():
    r=torch.zeros(64,1);r[-1]=1.;v=torch.zeros_like(r);end=torch.zeros_like(r,dtype=torch.bool);end[-1]=True
    dt=torch.full_like(r,.1)
    short,_=time_generalized_advantages(r,v,end,torch.zeros(1),dt,lam=.98)
    long,_=time_generalized_advantages(r,v,end,torch.zeros(1),dt,lam=.998)
    assert long[0]>100*short[0]
    end[31]=True
    stopped,_=time_generalized_advantages(r,v,end,torch.zeros(1),dt,lam=.998)
    assert stopped[0]==0.


def test_training_forwards_and_records_configured_lambda(tmp_path,monkeypatch):
    import trace_transport.reactive_sender_training as module
    seen=[];original=module.time_generalized_advantages
    def capture(*args,**kwargs):
        seen.append(kwargs['lam']);return original(*args,**kwargs)
    monkeypatch.setattr(module,'time_generalized_advantages',capture)
    train(tmp_path/'run',Config(decisions=4,environments=1,rollout=64,gae_lambda=.998,epochs=1),[700006],{})
    _,meta=load(tmp_path/'run/final.pt')
    assert seen and set(seen)=={.998}
    assert meta['training']['rollout']==64 and meta['training']['gae_lambda']==.998
    assert meta['critic']['post_update_mse']>=0
    assert meta['collected_macro_steps']>0


@pytest.mark.parametrize('value',[-.1,1.1,float('nan'),float('inf')])
def test_invalid_trace_rejected(value):
    with pytest.raises(ValueError,match='credit horizon'):replace(Config(),gae_lambda=value)
