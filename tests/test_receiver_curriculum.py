from dataclasses import asdict
import random

import pytest

from trace_transport.receiver_curriculum import DOMAIN, sample_receiver_episode
from trace_transport.simulator import NetworkConfig


def test_receiver_domains_are_seeded_and_vary_both_paths_and_flow_counts():
    def draw():
        rng = random.Random(42)
        return [sample_receiver_episode(rng) for _ in range(50)]
    a,b=draw(),draw()
    assert a==b
    assert {flows for _,flows,_ in a} == {1,2,3,4}
    assert any(cfg.feedback_capacity_changes for cfg,_,_ in a)
    assert any(cfg.outages for cfg,_,_ in a)
    for cfg,flows,size in a:
        assert DOMAIN['reverse_bps'][0] <= cfg.feedback_bandwidth_bps <= DOMAIN['reverse_bps'][1]
        assert DOMAIN['forward_bps'][0] <= cfg.bandwidth_bps <= DOMAIN['forward_bps'][1]
        assert size in DOMAIN['message_bytes']
        assert len(cfg.capacity_changes) == len(cfg.feedback_capacity_changes)
        assert 1<=flows<=4


def test_reverse_capacity_schedule_is_validated():
    with pytest.raises(ValueError,match='capacity changes'):
        NetworkConfig(feedback_capacity_changes=((1,1000),(.5,2000)))


def test_degradation_has_persistent_hold_and_recovers_original_capacity():
    cfgs=[cfg for cfg,_,_ in [sample_receiver_episode(random.Random(seed)) for seed in range(20)]
          if cfg.capacity_changes]
    assert cfgs
    for cfg in cfgs:
        f,r=cfg.capacity_changes,cfg.feedback_capacity_changes
        assert len(f)==len(r)==6
        assert f[2][1]==f[3][1]  # impairment persists over an interval
        assert f[3][0]>f[2][0]
        assert f[-1][1]==pytest.approx(cfg.bandwidth_bps)
        assert r[-1][1]==pytest.approx(cfg.feedback_bandwidth_bps)
        assert cfg.loss_changes[2][1] >= cfg.loss
        assert cfg.rtt_changes[2][1] >= cfg.base_rtt
        assert cfg.loss_changes[-1][1]==cfg.loss
        assert cfg.rtt_changes[-1][1]==cfg.base_rtt
