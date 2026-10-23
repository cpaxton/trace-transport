import random

import numpy as np
import pytest

from trace_transport.cli import run_simulation
from trace_transport.curriculum import sample_episode
from trace_transport.rewards import reward_spec, transport_reward
from trace_transport.simulator import NetworkConfig
from trace_transport.benchmark import summarize_runs


def reward(profile, **changes):
    kwargs = dict(progress=0, payload_bytes=65_536, overhead=0, inflation=0, dt=.02,
                  completed=False, timeout=False)
    kwargs.update(changes)
    return transport_reward(reward_spec(profile), **kwargs)


def test_completion_reward_penalizes_waiting_and_keeps_late_completion_meaningful():
    assert reward("legacy", dt=1) == 0
    assert reward("completion", dt=1) == pytest.approx(-.05)
    assert reward_spec("legacy").discount_gamma ** 1000 < .00005
    assert 2 * reward_spec("completion").discount_gamma ** 1000 > .7
    finishing = reward("completion", progress=65_536, completed=True, dt=2)
    partial_then_timeout = reward("completion", progress=40_000, timeout=True, dt=20)
    assert finishing > 2.8
    assert partial_then_timeout < -2


def test_completion_progress_is_normalized_by_message_size():
    assert reward("completion", progress=1024, payload_bytes=1024) == pytest.approx(
        reward("completion", progress=65_536, payload_bytes=65_536))


def test_training_curriculum_is_reproducible_and_covers_failure_modes():
    def samples():
        rng = random.Random(42)
        return [sample_episode(rng, profile="robust", max_flows=4, min_bytes=256,
                               max_bytes=1024, deadline=20) for _ in range(100)]
    first = samples()
    assert first == samples()
    assert {tag for _, _, _, tag in first} == {"clean", "lossy", "changing", "stress"}
    assert {len(texts) for texts, _, _, _ in first} == {1, 2, 3, 4}
    assert any(config.loss > .1 for _, config, _, _ in first)
    assert any(config.outages and config.capacity_changes for _, config, _, _ in first)
    assert any(config.corruption > 0 and config.ack_corruption > 0 for _, config, _, _ in first)


def test_evaluation_declares_scoring_and_rejects_mixed_objectives():
    cfg = NetworkConfig(loss=0, ack_loss=0)
    _, legacy = run_simulation(["hello"], cfg, "fixed", reward_profile="legacy")
    _, completion = run_simulation(["hello"], cfg, "fixed", reward_profile="completion")
    assert legacy["reward"]["version"] == "transport-v1"
    assert completion["reward"]["discount_gamma"] == .999
    with pytest.raises(ValueError, match="different reward"):
        summarize_runs([legacy, completion])


@pytest.mark.parametrize("architecture", ["transformer", "mlp", "gru"])
def test_all_history_models_mask_padding_and_roundtrip(architecture, tmp_path):
    torch = pytest.importorskip("torch")
    from trace_transport.learning import ModelConfig, load_checkpoint, make_model, save_checkpoint
    torch.set_num_threads(1)
    model = make_model(ModelConfig(history=8, width=16, layers=2, heads=2, architecture=architecture))
    history = torch.randn(3, 8, model.config.features)
    valid = torch.zeros(3, 8, dtype=torch.bool)
    valid[:, -3:] = True
    changed = history.clone()
    changed[:, :-3] = 1e6
    original = model(history, valid)
    for actual, expected in zip(model(changed, valid), original):
        torch.testing.assert_close(actual, expected)
    path = tmp_path / f"{architecture}.pt"
    save_checkpoint(path, model)
    restored = load_checkpoint(path)
    assert restored.config.architecture == architecture
    for actual, expected in zip(restored(history, valid), original):
        torch.testing.assert_close(actual, expected)
    actions, latent, log_prob, values = restored.sample(history, valid)
    assert torch.isfinite(log_prob).all() and torch.isfinite(values).all()
    assert actions.abs().max() <= 1


@pytest.mark.parametrize("architecture", ["transformer", "mlp", "gru"])
def test_robust_ppo_trains_with_variable_flow_counts(architecture, tmp_path):
    torch = pytest.importorskip("torch")
    from trace_transport.learning import ModelConfig, TrainConfig, train
    path = tmp_path / f"robust-{architecture}.pt"
    model = train(str(path), TrainConfig(
        updates=2, environments=4, flows=4, rollout_steps=8, epochs=1, minibatch=32,
        min_message_bytes=256, max_message_bytes=1024, episode_seconds=.3,
        seed=42, profile="robust", reward_profile="completion",
    ), ModelConfig(history=8, width=16, layers=1, heads=2, architecture=architecture))
    assert torch.isfinite(model.actor.weight).all()
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["training"]["reward"]["version"] == "transport-v2"
    assert checkpoint["training"]["active_transitions"] > 0
    _, result = run_simulation(["learned model 🌐"], NetworkConfig(loss=0, ack_loss=0),
                               "learned", checkpoint=path, reward_profile="completion")
    assert result["flows"][0]["completed"]


def test_legacy_checkpoint_without_architecture_still_loads(tmp_path):
    torch = pytest.importorskip("torch")
    from trace_transport.learning import HistoryTransformer, ModelConfig, load_checkpoint, save_checkpoint
    path = tmp_path / "legacy.pt"
    model = HistoryTransformer(ModelConfig(history=8, width=16, heads=2, layers=1))
    save_checkpoint(path, model)
    saved = torch.load(path, weights_only=True)
    del saved["model_config"]["architecture"]
    torch.save(saved, path)
    assert load_checkpoint(path).config.architecture == "transformer"


def test_kl_limit_stops_updates_and_retains_intermediate_weights(tmp_path):
    torch = pytest.importorskip("torch")
    import json
    from trace_transport.learning import ModelConfig, TrainConfig, train
    path = tmp_path / "guarded.pt"
    train(str(path), TrainConfig(updates=2, environments=2, flows=2, rollout_steps=16,
                                epochs=4, minibatch=8, min_message_bytes=256, max_message_bytes=1024,
                                episode_seconds=.5, target_kl=1e-10, checkpoint_every=1, seed=8),
          ModelConfig(history=8, width=16, layers=1, heads=2))
    rows = [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()]
    assert any(row["kl_stopped"] for row in rows)
    for update in [1, 2]:
        checkpoint = torch.load(tmp_path / f"guarded.u{update:04d}.pt", weights_only=True)
        assert checkpoint["training"]["update"] == update
    assert all(np.isfinite(row["approximate_kl_max"]) for row in rows)
