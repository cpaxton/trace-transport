"""Seeded training domains; independent of the frozen evaluation manifests."""

import math

from .simulator import NetworkConfig


def sample_episode(rng, *, profile, max_flows, min_bytes, max_bytes, deadline):
    seed = rng.randrange(2**31)
    if profile == "legacy":
        config = NetworkConfig(
            bandwidth_bps=10 ** rng.uniform(math.log10(500_000), math.log10(5_000_000)),
            base_rtt=rng.uniform(.01, .2), jitter=rng.uniform(0, .02),
            loss=rng.uniform(0, .05), ack_loss=rng.uniform(0, .05),
            queue_bytes=rng.choice([8_192, 16_384, 32_768]), deadline=deadline,
        )
        flows, category = max_flows, "legacy"
    elif profile == "robust":
        category = rng.choices(["clean", "lossy", "changing", "stress"], [2, 3, 3, 2])[0]
        flows = max_flows if category == "stress" else rng.randint(1, max_flows)
        rate = 10 ** rng.uniform(math.log10(500_000), math.log10(5_000_000))
        impaired = category != "clean"
        high_loss = category == "stress"
        loss_range = (.06, .15) if high_loss else (0, .08)
        changes = ()
        outages = ()
        # Times scale down for short software tests; normal runs use 20s deadlines.
        if category in ("changing", "stress"):
            first = rng.uniform(.03, .08) * deadline
            second = rng.uniform(.1, .2) * deadline
            changes = ((first, rate * rng.uniform(.3, .7)), (second, rate * rng.uniform(1.2, 2.5)))
            start = rng.uniform(.01, .04) * deadline
            outages = ((start, start + rng.uniform(.01, .04) * deadline),)
        config = NetworkConfig(
            bandwidth_bps=rate, feedback_bandwidth_bps=10 ** rng.uniform(5.7, 6.7),
            base_rtt=rng.uniform(.01, .2), jitter=rng.uniform(0, .025) if impaired else 0,
            loss=rng.uniform(*loss_range) if impaired else 0,
            ack_loss=rng.uniform(*loss_range) if impaired else 0,
            corruption=rng.uniform(0, .02) if impaired else 0,
            ack_corruption=rng.uniform(0, .02) if impaired else 0,
            queue_bytes=rng.choice([4_096, 8_192, 16_384, 32_768]),
            deadline=deadline, capacity_changes=changes, outages=outages,
        )
    else:
        raise ValueError(f"unknown training profile: {profile}")
    texts = ["x" * rng.randint(min_bytes, max_bytes) for _ in range(flows)]
    return texts, config, seed, category
