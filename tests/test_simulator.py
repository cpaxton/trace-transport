from dataclasses import replace
import numpy as np
import pytest

from trace_transport import NetworkConfig, transfer_text
from trace_transport.cli import run_simulation
from trace_transport.environment import FEATURE_INDEX, TransportEnv
from trace_transport.simulator import FifoLink


@pytest.mark.parametrize("text", ["", "one packet", "🌐こんにちは\r\n\x00" * 100, "packet history " * 2000])
def test_exact_transfer_under_impairments(text):
    config = NetworkConfig(loss=.15, ack_loss=.2, duplicate=.15, jitter=.04, deadline=15)
    result = transfer_text(text, config=config, seed=7)
    assert result.completed
    assert result.received_text.encode() == text.encode()
    assert result.acknowledged_bytes == len(text.encode())


def test_seeded_runs_are_reproducible():
    config = NetworkConfig(loss=.2, ack_loss=.1, jitter=.03)
    assert transfer_text("hello" * 1000, config=config, seed=8) == transfer_text("hello" * 1000, config=config, seed=8)


@pytest.mark.parametrize("loss,ack_loss,receiver_complete", [(1, 0, False), (0, 1, True)])
def test_deadline_does_not_report_false_success(loss, ack_loss, receiver_complete):
    result = transfer_text("not acknowledged", config=NetworkConfig(loss=loss, ack_loss=ack_loss, deadline=1))
    assert not result.completed
    assert result.receiver_complete == receiver_complete
    assert result.elapsed_seconds == 1


def test_shared_queue_and_staggered_senders_recover_after_outage():
    texts = ["first 🌐" * 1000, "second" * 1300, "third" * 800]
    results, report = run_simulation(
        texts, NetworkConfig(bandwidth_bps=500_000, queue_bytes=8192,
                             outages=((.1, .4),), capacity_changes=((.8, 2_000_000),), deadline=15),
        "aimd", seed=10, start_times=[0, .1, .3],
    )
    assert all(r.completed and r.received_text == text for r, text in zip(results, texts))
    assert report["network"]["forward"]["outage_drops"] > 0
    assert report["network"]["forward"]["queue_drops"] > 0


def test_fifo_serialization_capacity_and_capacity_change():
    link = FifoLink(8000, 2000, changes=((.5, 16000),))
    assert link.enqueue(0, 1000) == pytest.approx(.75)
    assert link.enqueue(0, 1000) == pytest.approx(1.25)
    assert link.enqueue(0, 1) is None
    assert link.enqueue(.75, 1000) == pytest.approx(1.75)


def test_custom_metadata_consumes_feedback_bandwidth():
    cfg = NetworkConfig(loss=0, ack_loss=0, jitter=0)
    a = TransportEnv(["hello"], cfg)
    b = TransportEnv(["hello"], cfg, receiver_metadata={"custom": "x" * 100})
    for env in [a, b]:
        env.reset(seed=1)
        while not env.simulator.finished:
            env.step([[0]])
    ar, br = a.simulator.results()[0], b.simulator.results()[0]
    assert br.feedback_wire_bytes > ar.feedback_wire_bytes
    assert br.elapsed_seconds > ar.elapsed_seconds


def test_observations_do_not_leak_hidden_capacity_or_drop_rate():
    a = TransportEnv(["hello"], NetworkConfig(bandwidth_bps=500_000, loss=.5))
    b = TransportEnv(["hello"], NetworkConfig(bandwidth_bps=20_000_000, loss=0))
    ao, _ = a.reset(seed=1)
    bo, _ = b.reset(seed=1)
    np.testing.assert_array_equal(ao, bo)
    assert not ao[0, FEATURE_INDEX["rtt_valid"]]


def test_reset_clears_reward_counters_and_invalid_actions_fail():
    env = TransportEnv(["hello" * 100])
    env.reset(seed=1)
    first = env.step([[0]])
    env.reset(seed=1)
    second = env.step([[0]])
    for index in range(4):
        np.testing.assert_array_equal(first[index], second[index])
    with pytest.raises(ValueError, match="shape"):
        env.step([0])
    with pytest.raises(ValueError, match="finite"):
        env.step([[float("nan")]])


def test_shrinking_window_does_not_deadlock():
    env = TransportEnv(["window" * 2000], NetworkConfig(loss=.1, deadline=20), action_dimensions=2)
    env.reset(seed=3)
    while not env.simulator.finished:
        obs, reward, _, _, _ = env.step([[0, -1]])
        assert np.isfinite(obs).all() and np.isfinite(reward).all()
    assert env.simulator.results()[0].completed


@pytest.mark.parametrize("kwargs", [dict(loss=2), dict(base_rtt=-1), dict(bandwidth_bps=0),
                                   dict(capacity_changes=((1, 0),)), dict(outages=((2, 1),)),
                                   dict(queue_bytes=0), dict(jitter=float("nan"))])
def test_invalid_network_configuration(kwargs):
    with pytest.raises(ValueError):
        NetworkConfig(**kwargs)
