"""Receiver-only experiment: frozen AIMD sender, local history, shared team reward.

Checkpoint schema is deliberately separate from sender checkpoints. CPU reference
only; actor inference time is measured in wall time, not modeled network time.
"""
from dataclasses import asdict
import math
from pathlib import Path

import numpy as np
import torch

from .controllers import AimdController
from .environment import TransportEnv
from .learning import ModelConfig, HistoryTransformer, empty_history, append_history

RECEIVER_FEATURES = (
    'received_fraction', 'received_bytes', 'total_bytes', 'chunk_count',
    'complete', 'message_seen', 'arrival_count', 'duplicate_pending_count',
    'reordered_count', 'last_offset', 'last_interarrival', 'arrival_age',
    'pending_count', 'pending_bytes', 'oldest_pending_age', 'ack_count',
    'ack_age', 'batch_target', 'delay_target', 'connection_age',
    'receive_capacity', 'remaining_bytes',
)


def observations(sim):
    rows = []
    for r, f in zip(sim.receivers, sim.feedback_schedulers):
        total = r.total or 0
        age = sim.now
        rows.append([
            r.received_bytes / max(1, total), math.log1p(r.received_bytes / 16384),
            math.log1p(total / 16384), math.log1p(len(r.chunks) / 32),
            float(r.complete), float(r.message_id is not None), math.log1p(f.arrivals / 32),
            math.log1p(f.duplicates / 8), math.log1p(f.reordered / 8),
            math.log1p(f.previous_offset / 16384), min(10., f.interarrival / .02),
            min(10., (sim.now - f.last_arrival) / .1), len(f.pending) / 8,
            sum(a.metadata['length'] for a, _ in f.pending.values()) / 4096,
            min(10., max((sim.now - t for _, t in f.pending.values()), default=0.) / .025),
            math.log1p(f.emitted / 32), min(10., (sim.now - f.last_emit) / .1),
            f.batch_size / 8, f.delay / max(f.max_delay,1e-9), math.log1p(age),
            math.log1p(r.max_message_bytes / 16384), math.log1p(max(0, total-r.received_bytes) / 16384),
        ])
    return np.asarray(rows, dtype=np.float32)


def make_model(seed=0, history=8):
    torch.manual_seed(seed)
    return HistoryTransformer(ModelConfig(history=history, width=32, heads=4, layers=1, action_dimensions=2))


def save(path, model, training):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format_version='receiver-categorical-v1' if hasattr(model,'feedback_actions') else 'receiver-v1', feature_names=list(RECEIVER_FEATURES),
                    model_config=asdict(model.config), state_dict=model.state_dict(), training=training,
                    action_bank=[list(a) for a in model.feedback_actions] if hasattr(model,'feedback_actions') else None), path)


def load(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('format_version') not in ('receiver-v1','receiver-categorical-v1') or data.get('feature_names') != list(RECEIVER_FEATURES):
        raise ValueError('incompatible receiver checkpoint')
    if data['format_version']=='receiver-categorical-v1':
        from .receiver_discrete import CategoricalReceiver,ACTIONS,WIDE_ACTIONS
        if data.get('action_bank') not in ([list(a) for a in ACTIONS],[list(a) for a in WIDE_ACTIONS]):
            raise ValueError('incompatible categorical action schema')
        model=CategoricalReceiver(ModelConfig(**data['model_config']),actions=tuple(tuple(a) for a in data['action_bank']))
    else:
        model = HistoryTransformer(ModelConfig(**data['model_config']))
    model.load_state_dict(data['state_dict'])
    return model.eval()


def episode(config, *, seed, payload_bytes=32768, flows=2, model=None,
            baseline=None, stochastic=False, receipt_feedback=False):
    if model and hasattr(model,'feedback_actions') and max(a[1] for a in model.feedback_actions)>max(.025,getattr(config,'feedback_delay_budget',0.)):
        raise ValueError('policy ACK delay bank exceeds configured endpoint delay budget')
    # Exact Unicode including CRLF. Same workload for all receiver policies.
    text = ('receiver 🌐\r\n' * (payload_bytes // 15 + 1)).encode()[:payload_bytes].decode('utf-8', errors='ignore')
    env = TransportEnv([text] * flows, config, reward_profile='completion', receiver_feedback=True,
                       receipt_feedback=receipt_feedback)
    sender_obs, _ = env.reset(seed=seed)
    sender = AimdController(flows)
    sender.reset(sender_obs)
    history, valid = empty_history(flows, model.config) if model else (None, None)
    trajectory = []
    score = np.zeros(flows)
    while not env.simulator.finished:
        if model:
            append_history(history, valid, observations(env.simulator))
            with torch.no_grad():
                h, v = torch.from_numpy(history.copy()), torch.from_numpy(valid.copy())
                action, latent, logp, values = model.sample(h, v)
                if not stochastic:
                    mean, _ = model(h, v)
                    action = mean.argmax(dim=-1) if hasattr(model,'feedback_actions') else mean.tanh()
            actions = action.numpy()
        else:
            batch, delay = baseline or (1, 0.)
            # Baselines set exact targets, avoiding continuous bin boundaries.
            actions = None
        active = np.array([not s.complete for s in env.simulator.senders])
        for i, feedback in enumerate(env.simulator.feedback_schedulers):
            if actions is not None:
                if hasattr(model,'feedback_actions'):
                    feedback.batch_size,feedback.delay=model.feedback_actions[int(actions[i])]
                else:
                    feedback.set_action(actions[i])
            else:
                feedback.batch_size, feedback.delay = batch, delay
        sender_obs, reward, terminated, truncated, _ = env.step(sender.act(sender_obs))
        score += reward
        if model and stochastic:
            trajectory.append((h, v, latent, logp, values, torch.from_numpy(reward),
                               torch.from_numpy(terminated | truncated), torch.from_numpy(active)))
    return env.simulator.results(), score, trajectory
