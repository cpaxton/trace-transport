"""Small masked history transformer and a minimal shared-policy PPO trainer.

The reference simulator runs on CPU. Only policy inference and optimization move
to the selected torch device. This is not the planned GPU simulation backend.
"""

from collections import Counter, deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise ImportError("Learned controllers require PyTorch: run uv sync --locked --extra learn") from exc

from .environment import FEATURE_NAMES, TransportEnv
from .simulator import NetworkConfig
from .curriculum import sample_episode
from .rewards import reward_spec


@dataclass(frozen=True)
class ModelConfig:
    features: int = len(FEATURE_NAMES)
    history: int = 64
    width: int = 64
    heads: int = 4
    layers: int = 2
    action_dimensions: int = 1
    architecture: str = "transformer"

    def __post_init__(self):
        if self.features != len(FEATURE_NAMES):
            raise ValueError("checkpoint observation schema does not match this version")
        if self.width <= 0 or self.heads <= 0 or (self.architecture == "transformer" and self.width % self.heads):
            raise ValueError("width must be positive and divisible by heads")
        if self.history <= 0 or self.layers <= 0 or self.action_dimensions not in (1, 2):
            raise ValueError("invalid model dimensions")
        if self.architecture not in ("transformer", "mlp", "gru"):
            raise ValueError("architecture must be transformer, mlp, or gru")


class AttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, tokens, allowed):
        batch, length, width = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None])
        attention = attention.transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + self.projection(attention)
        return tokens + self.mlp(self.norm2(tokens))


class HistoryTransformer(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.input = nn.Linear(cfg.features, cfg.width)
        self.position = nn.Parameter(torch.zeros(cfg.history, cfg.width))
        nn.init.normal_(self.position, std=.02)
        self.blocks = nn.ModuleList([AttentionBlock(cfg.width, cfg.heads) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.width)
        self.actor = nn.Linear(cfg.width, cfg.action_dimensions)
        self.critic = nn.Linear(cfg.width, 1)
        self.log_std = nn.Parameter(torch.full((cfg.action_dimensions,), -.7))
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def forward(self, history, valid):
        if history.shape[1:] != (self.config.history, self.config.features):
            raise ValueError("history shape does not match model configuration")
        length = history.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=history.device).tril()
        allowed = causal[None] & valid[:, None, :]
        # Give padded queries their own diagonal so softmax never has an empty
        # row. Real queries cannot attend to padded keys at any layer.
        diagonal = torch.eye(length, dtype=torch.bool, device=history.device)[None]
        allowed = allowed | (diagonal & ~valid[:, :, None])
        tokens = self.input(history.masked_fill(~valid[:, :, None], 0)) + self.position
        for block in self.blocks:
            tokens = block(tokens, allowed)
        state = self.norm(tokens[:, -1])
        mean = self.actor(state)
        value = self.critic(state).squeeze(-1)
        return mean, value

    @staticmethod
    def _log_prob(distribution, latent):
        correction = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
        return (distribution.log_prob(latent) - correction).sum(-1)

    def sample(self, history, valid):
        mean, value = self(history, valid)
        distribution = torch.distributions.Normal(mean, self.log_std.clamp(-5, 1).exp())
        latent = distribution.sample()
        return latent.tanh(), latent, self._log_prob(distribution, latent), value

    def evaluate(self, history, valid, latent):
        mean, value = self(history, valid)
        distribution = torch.distributions.Normal(mean, self.log_std.clamp(-5, 1).exp())
        # Base Gaussian entropy is used as an exploration regularizer.
        return self._log_prob(distribution, latent), distribution.entropy().sum(-1), value


class HistoryMLP(HistoryTransformer):
    """Flattened, masked history with an explicit padding indicator."""

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        modules = []
        size = config.history * (config.features + 1)
        for _ in range(config.layers):
            modules.extend([nn.Linear(size, config.width), nn.GELU()])
            size = config.width
        self.encoder = nn.Sequential(*modules)
        self.norm = nn.LayerNorm(config.width)
        self.actor = nn.Linear(config.width, config.action_dimensions)
        self.critic = nn.Linear(config.width, 1)
        self.log_std = nn.Parameter(torch.full((config.action_dimensions,), -.7))
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def forward(self, history, valid):
        masked = history.masked_fill(~valid[:, :, None], 0)
        inputs = torch.cat([masked, valid[:, :, None].to(masked.dtype)], dim=-1).flatten(1)
        state = self.norm(self.encoder(inputs))
        return self.actor(state), self.critic(state).squeeze(-1)


class HistoryGRU(HistoryMLP):
    """A GRU over the same finite history window, with padding removed."""

    def __init__(self, config):
        super().__init__(config)
        del self.encoder
        self.gru = nn.GRU(config.features, config.width, config.layers, batch_first=True)

    def forward(self, history, valid):
        # History buffers are right-aligned, with contiguous valid suffixes.
        length = history.shape[1]
        lengths = valid.sum(1)
        indices = (torch.arange(length, device=history.device)[None] + length - lengths[:, None]).clamp(max=length - 1)
        compact = history.masked_fill(~valid[:, :, None], 0).gather(
            1, indices[:, :, None].expand(-1, -1, history.shape[-1]))
        outputs, _ = self.gru(compact)
        state = self.norm(outputs[torch.arange(len(history), device=history.device), (lengths - 1).clamp(min=0)])
        return self.actor(state), self.critic(state).squeeze(-1)


def make_model(config):
    constructors = {"transformer": HistoryTransformer, "mlp": HistoryMLP, "gru": HistoryGRU}
    return constructors[config.architecture](config)


def empty_history(batch: int, config: ModelConfig):
    return (np.zeros((batch, config.history, config.features), dtype=np.float32),
            np.zeros((batch, config.history), dtype=bool))


def append_history(history, valid, observations):
    history[:, :-1] = history[:, 1:].copy()
    valid[:, :-1] = valid[:, 1:].copy()
    history[:, -1] = observations
    valid[:, -1] = True


def save_checkpoint(path, model: HistoryTransformer, *, training: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format_version": 1, "feature_names": list(FEATURE_NAMES),
        "model_config": asdict(model.config), "state_dict": model.state_dict(),
        "training": training or {},
    }, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("incompatible checkpoint format or feature schema")
    model = make_model(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


class LearnedController:
    def __init__(self, checkpoint, flows: int, *, device="cpu"):
        # These small reference policies are faster without large CPU thread pools.
        if str(device) == "cpu":
            torch.set_num_threads(1)
        self.model = load_checkpoint(checkpoint, device)
        self.device = device
        self.flows = flows
        self.action_dimensions = self.model.config.action_dimensions

    def reset(self, observations):
        self.history, self.valid = empty_history(self.flows, self.model.config)

    def act(self, observations):
        append_history(self.history, self.valid, observations)
        with torch.inference_mode():
            mean, _ = self.model(torch.as_tensor(self.history, device=self.device),
                                 torch.as_tensor(self.valid, device=self.device))
        return mean.tanh().cpu().numpy()


# Retain the original public import and controller spelling for old callers.
TransformerController = LearnedController


def generalized_advantages(rewards, values, terminal, bootstrap, *, gamma=.99, lam=.95):
    """Finite-transfer deadlines are terminal for this task; never bootstrap them."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for step in reversed(range(len(rewards))):
        continuation = (~terminal[step]).to(rewards.dtype)
        delta = rewards[step] + gamma * next_value * continuation - values[step]
        carry = delta + gamma * lam * continuation * carry
        advantages[step] = carry
        next_value = values[step]
    return advantages, advantages + values


@dataclass(frozen=True)
class TrainConfig:
    updates: int = 20
    environments: int = 4
    flows: int = 2
    rollout_steps: int = 64
    epochs: int = 4
    minibatch: int = 128
    learning_rate: float = 3e-4
    seed: int = 0
    device: str = "cpu"
    threads: int = 1
    min_message_bytes: int = 16_384
    max_message_bytes: int = 65_536
    episode_seconds: float = 4.0
    profile: str = "legacy"
    reward_profile: str = "legacy"
    target_kl: float | None = None
    checkpoint_every: int = 0

    def __post_init__(self):
        for name in ("updates", "environments", "flows", "rollout_steps", "epochs", "minibatch", "threads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 1 <= self.min_message_bytes <= self.max_message_bytes <= 16 * 1024 * 1024:
            raise ValueError("training message sizes must lie in [1, 16 MiB]")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.episode_seconds) or self.episode_seconds <= 0:
            raise ValueError("episode_seconds must be positive and finite")
        if self.profile not in ("legacy", "robust"):
            raise ValueError("profile must be legacy or robust")
        reward_spec(self.reward_profile)
        if self.target_kl is not None and (not math.isfinite(self.target_kl) or self.target_kl <= 0):
            raise ValueError("target_kl must be positive and finite")
        if type(self.checkpoint_every) is not int or self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be a nonnegative integer")


def train(output: str, config: TrainConfig | None = None, model_config: ModelConfig | None = None):
    cfg, model_cfg = config or TrainConfig(), model_config or ModelConfig()
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = random.Random(cfg.seed)
    device = torch.device(cfg.device)
    model = make_model(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    categories = Counter()

    def pad_observation(obs, info):
        padded = np.zeros((cfg.flows, model_cfg.features), dtype=np.float32)
        padded[:len(obs)] = obs
        alive = np.zeros(cfg.flows, dtype=bool)
        alive[:len(obs)] = info["active"]
        return padded, {**info, "active": alive}

    def new_environment():
        texts, settings, seed, category = sample_episode(
            rng, profile=cfg.profile, max_flows=cfg.flows,
            min_bytes=cfg.min_message_bytes, max_bytes=cfg.max_message_bytes, deadline=cfg.episode_seconds,
        )
        categories[category] += 1
        env = TransportEnv(texts, settings, action_dimensions=model_cfg.action_dimensions,
                           reward_profile=cfg.reward_profile)
        obs, info = env.reset(seed=seed)
        obs, info = pad_observation(obs, info)
        return env, obs, info

    instances = [new_environment() for _ in range(cfg.environments)]
    envs = [item[0] for item in instances]
    history, valid = empty_history(cfg.environments * cfg.flows, model_cfg)
    append_history(history, valid, np.concatenate([item[1] for item in instances]))
    active = np.concatenate([item[2]["active"] for item in instances])
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(".jsonl")
    start_time = time.perf_counter()
    transitions = 0
    completed = deque(maxlen=100)
    with log_path.open("w", encoding="utf-8") as log:
        for update in range(1, cfg.updates + 1):
            histories, masks, latents, log_probs, values, rewards, terminals, active_masks = ([] for _ in range(8))
            model.eval()
            for _ in range(cfg.rollout_steps):
                h = torch.as_tensor(history.copy(), device=device)
                m = torch.as_tensor(valid.copy(), device=device)
                with torch.no_grad():
                    action, latent, log_prob, value = model.sample(h, m)
                histories.append(h)
                masks.append(m)
                latents.append(latent)
                log_probs.append(log_prob)
                values.append(value)
                active_masks.append(torch.as_tensor(active.copy(), device=device))
                actions = action.cpu().numpy().reshape(cfg.environments, cfg.flows, model_cfg.action_dimensions)
                next_obs, step_rewards, step_done, next_active = [], [], [], []
                reset_indices = []
                for index, env in enumerate(envs):
                    obs, reward, terminated, truncated, info = env.step(actions[index, :len(env.texts)])
                    padded_reward = np.zeros(cfg.flows, dtype=np.float32)
                    padded_reward[:len(reward)] = reward
                    padded_done = np.ones(cfg.flows, dtype=bool)
                    padded_done[:len(reward)] = terminated | truncated
                    step_rewards.append(padded_reward)
                    step_done.append(padded_done)
                    if env.simulator.finished:
                        completed.extend(float(r.completed) for r in env.simulator.results())
                        envs[index], obs, info = new_environment()
                        reset_indices.append(index)
                    else:
                        obs, info = pad_observation(obs, info)
                    next_obs.append(obs)
                    next_active.append(info["active"])
                rewards.append(torch.as_tensor(np.concatenate(step_rewards), device=device))
                terminals.append(torch.as_tensor(np.concatenate(step_done), device=device))
                for index in reset_indices:
                    section = slice(index * cfg.flows, (index + 1) * cfg.flows)
                    history[section] = 0
                    valid[section] = False
                append_history(history, valid, np.concatenate(next_obs))
                transitions += int(active.sum())
                active = np.concatenate(next_active)
            with torch.no_grad():
                _, bootstrap = model(torch.as_tensor(history, device=device), torch.as_tensor(valid, device=device))
            advantages, returns = generalized_advantages(
                torch.stack(rewards), torch.stack(values), torch.stack(terminals), bootstrap,
                gamma=reward_spec(cfg.reward_profile).discount_gamma,
            )
            keep = torch.stack(active_masks).flatten()
            h = torch.stack(histories).flatten(0, 1)[keep]
            m = torch.stack(masks).flatten(0, 1)[keep]
            z = torch.stack(latents).flatten(0, 1)[keep]
            old_log_prob = torch.stack(log_probs).flatten()[keep]
            target = returns.flatten()[keep]
            adv = advantages.flatten()[keep]
            if len(adv) > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            model.train()
            losses = []
            kls, clip_fractions = [], []
            kl_stopped = False
            for _ in range(cfg.epochs):
                order = torch.randperm(len(adv), device=device)
                for offset in range(0, len(adv), cfg.minibatch):
                    batch = order[offset:offset + cfg.minibatch]
                    new_log_prob, entropy, value = model.evaluate(h[batch], m[batch], z[batch])
                    log_ratio = new_log_prob - old_log_prob[batch]
                    ratio = log_ratio.exp()
                    approximate_kl = float(((ratio - 1) - log_ratio).mean().detach())
                    kls.append(approximate_kl)
                    clip_fractions.append(float(((ratio - 1).abs() > .2).float().mean().detach()))
                    if cfg.target_kl is not None and approximate_kl > 1.5 * cfg.target_kl:
                        kl_stopped = True
                        break
                    unclipped = ratio * adv[batch]
                    clipped = ratio.clamp(.8, 1.2) * adv[batch]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = .5 * (value - target[batch]).square().mean()
                    loss = policy_loss + value_loss - .01 * entropy.mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite PPO loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), .5)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                if kl_stopped:
                    break
            record = {
                "update": update, "active_transitions": transitions,
                "elapsed_seconds": time.perf_counter() - start_time,
                "mean_loss": sum(losses) / max(1, len(losses)),
                "mean_reward": float(torch.stack(rewards)[torch.stack(active_masks)].mean()),
                "recent_completion_rate": sum(completed) / len(completed) if completed else None,
                "architecture": model_cfg.architecture,
                "parameters": sum(p.numel() for p in model.parameters()),
                "reward": reward_spec(cfg.reward_profile).metadata(),
                "sampled_episode_categories": dict(categories),
                "approximate_kl_max": max(kls, default=0.0),
                "clip_fraction": sum(clip_fractions) / max(1, len(clip_fractions)),
                "optimizer_steps": len(losses), "kl_stopped": kl_stopped,
                "latent_std": float(model.log_std.clamp(-5, 1).exp().mean().detach()),
            }
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
            save_checkpoint(output_path, model, training={"config": asdict(cfg), **record})
            if cfg.checkpoint_every and update % cfg.checkpoint_every == 0:
                snapshot = output_path.with_name(f"{output_path.stem}.u{update:04d}{output_path.suffix}")
                save_checkpoint(snapshot, model, training={"config": asdict(cfg), **record})
    return model
