"""Versioned, shared training and evaluation objectives."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RewardSpec:
    version: str
    discount_gamma: float
    progress_scale: float | None
    overhead_scale: float | None
    queue_cost_per_second: float
    time_cost_per_second: float
    completion_bonus: float
    timeout_penalty: float
    overhead_weight: float = .1

    def metadata(self):
        return asdict(self)


PROFILES = {
    "legacy": RewardSpec("transport-v1", .99, 16_384, 16_384, 1.0, 0.0, 1.0, 1.0),
    "completion": RewardSpec("transport-v2", .999, None, None, .01, .05, 2.0, 2.0),
}


def reward_spec(profile):
    try:
        return PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown reward profile: {profile}") from exc


def transport_reward(spec, *, progress, payload_bytes, overhead, inflation, dt, completed, timeout):
    progress_scale = spec.progress_scale or max(payload_bytes, 1)
    overhead_scale = spec.overhead_scale or max(payload_bytes, 512)
    return (progress / progress_scale
            - spec.queue_cost_per_second * (inflation / .1) * dt
            - spec.time_cost_per_second * dt
            - spec.overhead_weight * overhead / overhead_scale
            + spec.completion_bonus * completed - spec.timeout_penalty * timeout)
