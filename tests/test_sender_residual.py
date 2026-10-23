import pytest
pytest.importorskip('torch')
import torch

from trace_transport.sender_residual import make_model,episode,save,load
from trace_transport.learning import load_checkpoint
from trace_transport.simulator import NetworkConfig


def test_zero_residual_exactly_matches_strong_baseline_and_has_distinct_schema(tmp_path):
    torch.set_num_threads(1)
    model=make_model(42)
    cfg=NetworkConfig(feedback_bandwidth_bps=64000,loss=.03,ack_loss=.05)
    result,score,_=episode(cfg,seed=42,model=model)
    baseline,baseline_score,_=episode(cfg,seed=42,baseline='bdp-aimd')
    assert result==baseline
    assert (score==baseline_score).all()
    path=tmp_path/'sender.pt';save(path,model,{})
    restored=load(path)
    assert restored.config==model.config
    with pytest.raises(ValueError,match='incompatible'):
        load_checkpoint(path)


def test_sender_cannot_win_by_trading_ack_savings_for_more_total_traffic():
    from trace_transport.sender_experiment import sender_comparison
    baseline=[dict(scenario='test',seed=s,completed=1,transfers=1,wrong=0,
                   elapsed=1.,data_bytes=100,feedback_bytes=10,score=1.) for s in [1,2,3]]
    learned=[{**r,'data_bytes':110,'feedback_bytes':5} for r in baseline]
    result=sender_comparison(learned,baseline)
    assert result['metrics']['feedback_bytes']['mean_delta']<0
    assert result['metrics']['total_wire_bytes']['mean_delta']>0
    assert not result['diagnostic_win']


def test_residual_sender_supports_staggered_short_and_bulk_messages():
    cfg=NetworkConfig(loss=.01,ack_loss=.02,deadline=10)
    texts=['bulk 🌐'*500,'short','later']
    result,_,_=episode(cfg,seed=42,texts=texts,start_times=[0.,.1,.2],baseline='fixed')
    assert all(r.completed for r in result)
    assert [r.received_text for r in result]==texts
