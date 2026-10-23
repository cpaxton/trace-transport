import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trace_transport.learning import (
    HistoryTransformer, ModelConfig, TrainConfig, TransformerController,
    empty_history, generalized_advantages, load_checkpoint, save_checkpoint, train,
)
from trace_transport import NetworkConfig, transfer_text


def small_config(**kwargs):
    return ModelConfig(history=8, width=16, heads=2, layers=1, **kwargs)


def test_padding_is_masked_and_actions_have_finite_corrected_probabilities():
    torch.set_num_threads(1)
    model = HistoryTransformer(small_config())
    history = torch.randn(2, 8, model.config.features)
    valid = torch.zeros(2, 8, dtype=torch.bool)
    valid[:, -2:] = True
    changed = history.clone()
    changed[:, :-2] = 100_000
    a, v = model(history, valid)
    b, w = model(changed, valid)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(v, w)
    action, latent, logp, value = model.sample(history, valid)
    reevaluated, _, _ = model.evaluate(history, valid, latent)
    torch.testing.assert_close(logp, reevaluated)
    assert torch.isfinite(logp).all() and torch.isfinite(value).all()
    assert (action.abs() <= 1).all()


def test_advantages_stop_at_terminal_boundaries():
    rewards = torch.tensor([[1.], [2.]])
    values = torch.zeros_like(rewards)
    terminals = torch.tensor([[True], [False]])
    advantages, returns = generalized_advantages(rewards, values, terminals, torch.tensor([100.]), gamma=1, lam=1)
    torch.testing.assert_close(advantages, torch.tensor([[1.], [102.]]))


def test_checkpoint_roundtrip_and_transformer_transfer(tmp_path):
    model = HistoryTransformer(small_config(action_dimensions=2))
    path = tmp_path / "model.pt"
    save_checkpoint(path, model)
    restored = load_checkpoint(path)
    for original, loaded in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(original, loaded)
    result = transfer_text("checkpoint 🌐" * 100, controller="transformer", checkpoint=str(path),
                           config=NetworkConfig(loss=.1, ack_loss=.1, deadline=10), seed=7)
    assert result.completed and result.received_text == result.sent_text


def test_ppo_updates_weights_logs_metrics_and_saves_usable_policy(tmp_path):
    path = tmp_path / "trained.pt"
    model = train(str(path), TrainConfig(
        updates=2, environments=2, flows=2, rollout_steps=8, epochs=1, minibatch=16,
        min_message_bytes=256, max_message_bytes=1024, episode_seconds=.4, seed=8,
    ), small_config())
    assert model.actor.weight.abs().sum() > 0
    assert torch.isfinite(model.actor.weight).all()
    records = [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()]
    assert [r["update"] for r in records] == [1, 2]
    assert records[-1]["active_transitions"] > 0
    result = transfer_text("trained", controller="transformer", checkpoint=str(path),
                           config=NetworkConfig(loss=0, ack_loss=0))
    assert result.completed
