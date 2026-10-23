"""A small vector-observation, multi-sender RL environment.

The reset/step interface resembles Gymnasium, but done/reward values are per-flow
arrays. This is deliberately not registered as a single-agent Gymnasium Env.
"""

import math
import numpy as np

from .simulator import NetworkConfig, PacketSimulator
from .rewards import reward_spec, transport_reward

FEATURE_NAMES = (
    "sent_wire_rate", "acked_payload_rate", "retransmitted_rate", "declared_lost_rate",
    "mean_rtt", "minimum_rtt", "smoothed_rtt", "rtt_trend", "bytes_in_flight",
    "pacing_rate", "congestion_window", "receiver_credit", "unsent_bytes",
    "feedback_age", "connection_age", "previous_rate_action", "previous_window_action",
    "rtt_valid", "feedback_seen", "peer_received_bytes", "complete", "started",
)
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def logscale(value: float, scale: float) -> float:
    return math.log1p(max(0.0, value) / scale)


def sender_snapshot(sender, feedback_bytes=0):
    return (sender.wire_sent, sender.acknowledged_bytes, sender.retransmitted_bytes,
            sender.declared_lost_bytes, len(sender.rtt_samples), sender.srtt,
            feedback_bytes, sender.complete)


def sender_observation(sender, now, start, previous, dt):
    """Shared local telemetry encoding for simulated and socket-based senders."""
    samples = sender.rtt_samples[previous[4]:]
    mean_rtt = sum(samples) / len(samples) if samples else sender.srtt
    age = max(0.0, now - start)
    feedback_age = age if sender.last_ack_at is None else now - sender.last_ack_at
    return np.asarray([
        logscale((sender.wire_sent - previous[0]) / dt, 128_000),
        logscale((sender.acknowledged_bytes - previous[1]) / dt, 128_000),
        logscale((sender.retransmitted_bytes - previous[2]) / dt, 128_000),
        logscale((sender.declared_lost_bytes - previous[3]) / dt, 128_000),
        mean_rtt / .1, sender.min_rtt / .1, sender.srtt / .1,
        (sender.srtt - previous[5]) / .1,
        logscale(sender.flight_bytes, 16_384), logscale(sender.rate, 128_000),
        logscale(sender.cwnd, 16_384), logscale(sender.receiver_credit, 65_536),
        logscale(sender.unsent_bytes, 65_536), logscale(feedback_age, .1),
        logscale(age, 1), *sender.previous_action,
        float(bool(samples)), float(sender.last_ack_at is not None),
        logscale(sender.peer_metadata.get("received", 0), 65_536),
        float(sender.complete), float(now >= start),
    ], dtype=np.float32)


class TransportEnv:
    observation_size = len(FEATURE_NAMES)

    def __init__(self, texts: list[str], config: NetworkConfig | None = None, *, action_dimensions: int = 1,
                 start_times: list[float] | None = None, receiver_metadata: dict | None = None,
                 reward_profile: str = "legacy", receiver_feedback: bool = False):
        if action_dimensions not in (1, 2):
            raise ValueError("action_dimensions must be one or two")
        self.texts = list(texts)
        self.config = config or NetworkConfig()
        self.action_dimensions = action_dimensions
        self.start_times = start_times
        self.receiver_metadata = receiver_metadata
        self.receiver_feedback = receiver_feedback
        self.reward_spec = reward_spec(reward_profile)
        self.simulator: PacketSimulator
        self._previous = []

    def _snapshot(self, i: int):
        sender = self.simulator.senders[i]
        return sender_snapshot(sender, self.simulator.feedback_bytes[i])

    def reset(self, *, seed: int = 0):
        self.simulator = PacketSimulator(self.texts, self.config, seed=seed,
                                        start_times=self.start_times, receiver_metadata=self.receiver_metadata,
                                        receiver_feedback=self.receiver_feedback)
        self._previous = [self._snapshot(i) for i in range(len(self.texts))]
        self._payload_totals = [0] * len(self.texts)
        return self._observe(self.config.control_interval), self._info()

    def _observe(self, dt: float):
        rows = []
        for i, sender in enumerate(self.simulator.senders):
            rows.append(sender_observation(sender, self.simulator.now, self.simulator.starts[i],
                                           self._previous[i], dt))
        return np.asarray(rows, dtype=np.float32)

    def _info(self):
        sim = self.simulator
        return {
            "time": sim.now,
            "active": np.asarray([not s.complete and sim.now >= start and sim.now < self.config.deadline
                                  for s, start in zip(sim.senders, sim.starts)], dtype=bool),
            "feature_names": FEATURE_NAMES,
        }

    def step(self, actions):
        if self.simulator.finished:
            raise RuntimeError("episode has finished; call reset before stepping again")
        actions = np.asarray(actions, dtype=np.float64)
        expected = (len(self.texts), self.action_dimensions)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"expected finite actions with shape {expected}")
        before = self.simulator.now
        self.simulator.advance(actions)
        dt = max(1e-9, self.simulator.now - before)
        obs = self._observe(dt)
        rewards = []
        terminated = []
        truncated = []
        for i, sender in enumerate(self.simulator.senders):
            old = self._previous[i]
            progress = sender.acknowledged_bytes - old[1]
            sent = sender.wire_sent - old[0]
            # Charge all sender headers, redundant payload, and receiver feedback.
            # Exact application bytes introduced in this interval are tracked
            # independently below, rather than inferred from ACK timing.
            previous_payload = self._payload_totals[i]
            payload_delta = sender.payload_sent - previous_payload
            new_payload = payload_delta - (sender.retransmitted_bytes - old[2])
            overhead = sent - new_payload + self.simulator.feedback_bytes[i] - old[6]
            inflation = max(0.0, sender.srtt - sender.min_rtt)
            timeout = self.simulator.now >= self.config.deadline and not sender.complete
            active_dt = max(0.0, self.simulator.now - max(before, self.simulator.starts[i]))
            reward = transport_reward(
                self.reward_spec, progress=progress, payload_bytes=len(sender.payload),
                overhead=overhead, inflation=inflation, dt=active_dt,
                completed=sender.complete and not old[7], timeout=timeout,
            )
            rewards.append(0.0 if old[7] else reward)
            terminated.append(sender.complete)
            truncated.append(timeout)
        self._payload_totals = [s.payload_sent for s in self.simulator.senders]
        self._previous = [self._snapshot(i) for i in range(len(self.texts))]
        return (obs, np.asarray(rewards, dtype=np.float32), np.asarray(terminated),
                np.asarray(truncated), self._info())
