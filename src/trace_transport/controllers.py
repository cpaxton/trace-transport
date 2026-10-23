"""Controllers use only endpoint observations; they never inspect network state."""

import math
import numpy as np

from .environment import FEATURE_INDEX as I


class FixedController:
    def __init__(self, flows: int, action_dimensions: int = 1):
        self.flows = flows
        self.action_dimensions = action_dimensions

    def reset(self, observations):
        pass

    def act(self, observations):
        return np.zeros((self.flows, self.action_dimensions), dtype=np.float32)


class AimdController(FixedController):
    """A rate AIMD reference, not a TCP implementation.

    Increase by 2,000 bytes/s per feedback-bearing interval; reduce by 20% on
    estimated loss or substantial RTT inflation, with an RTT-sized cooldown.
    """

    def reset(self, observations):
        self.cooldowns = np.zeros(self.flows, dtype=np.int64)

    def act(self, observations):
        actions = super().act(observations)
        for i, row in enumerate(observations):
            self.cooldowns[i] = max(0, self.cooldowns[i] - 1)
            rate = math.expm1(float(row[I["pacing_rate"]])) * 128_000
            loss = row[I["declared_lost_rate"]] > 0
            inflated = row[I["smoothed_rtt"]] > max(.2, 1.5 * row[I["minimum_rtt"]])
            if self.cooldowns[i] == 0 and (loss or inflated):
                actions[i, 0] = -1
                self.cooldowns[i] = max(1, math.ceil(row[I["smoothed_rtt"]] * .1 / .02))
            elif row[I["acked_payload_rate"]] > 0:
                actions[i, 0] = min(1.0, math.log1p(2_000 / max(rate, 1)) / .2)
        return actions


def make_controller(name: str, flows: int, *, checkpoint: str | None = None,
                    device: str = "cpu", action_dimensions: int = 1):
    if name == "fixed":
        return FixedController(flows, action_dimensions)
    if name == "aimd":
        return AimdController(flows, action_dimensions)
    if name in ("transformer", "learned"):
        if not checkpoint:
            raise ValueError("a learned controller requires --checkpoint")
        from .learning import LearnedController
        return LearnedController(checkpoint, flows, device=device)
    raise ValueError(f"unknown controller: {name}")
