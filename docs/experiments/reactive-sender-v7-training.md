# Caliban: 50k-decision reactive sender training

**Running**, actual start 2026-10-27 01:31:44 UTC (October 26 local evening).
This follows the completed [2k engineering pilot](reactive-sender-v7.md).

[Registered configuration](../../experiments/reactive-sender-v7-training.json),
[actual on-device plan](../../artifacts/reactive-sender-v7-training-plan.json),
[source-bound gate](../../artifacts/reactive-sender-v7-training-gate.json),
[frozen source](../../artifacts/reactive-sender-v7-training-source.zip),
[operational snapshot](../../artifacts/reactive-sender-v7-training-status.json).

## Fixed experiment

- Six independent from-scratch runs: current MLP / physical-history MLP, seeds
  42/43/44, **50,000 eligible producer macro decisions each** (300k total).
- Same model, optimizer, initialization rules, raw reward, physical-time GAE and
  quality criteria as the small pilot. No receiver/reverse-role learning.
- Training coverage expands to 48 cases, including the original 24. The retained
  calibration bank remains unused. All six final models evaluate on 24 new,
  prospective validation cases; the earlier pilot validation is not reused.
- No checkpoint/model selection or reward/guard tuning on validation. The original
  reserved acceptance cases remain unopened. Expanding coverage means this is not
  an isolated budget-only causal comparison.

The earlier heads were provably greedy-teacher over their bounded hidden states,
so that short smoke could not assess adaptive behavior. This training exposure is
justified by source/role/timing checks, small development intervention signals and
training-only weight bounds, not by selecting a validation winner. The same
intervention states are replayed for source/config binding, not independent data.
This explicitly amends the earlier resource gate; it does not relax performance
acceptance or assert a development/validation learning gain.

The control contract stays producer-only target-rate bias 0.8/1/1.2, a minimum
100-ms decision period adjusted by local RTT, fixed teacher window/ACK/scheduling,
20-ms modeled inference delay, one-second physical histories and a training-only
centralized critic. Repeated target bias cannot compound current rate. Greedy
initialization is the teacher; training samples teacher/alternatives at 0.6/0.2/0.2.
Actors receive neither network capacity, scenario family, future events nor critic
state. Receiver learning can return in a separate study once sender control has
useful evidence.

## Operations

Caliban CPU-only runtime, four environment workers, six-core affinity 6–11, low
priority, 8 GiB memory / 512 MiB swap, **12-hour cap**. Unit:
`trace-reactive-sender-v7-training`. Source/config/gates are immutable. No new
packages, system changes, robot motion or real-network experiments are involved.
The completed small study remains unchanged. Latest/initial/final checkpoints and
JSON are mirrored atomically under a 128 MiB budget and 1 GiB disk-free floor.
This initial trainer does not implement optimizer resume; snapshots are retained.

```bash
.venv/bin/python -m scripts.caliban_deploy status
ssh caliban systemctl --user status trace-reactive-sender-v7-training --no-pager
```

The recorded snapshot verifies 50k actual budgets, an active service, completed
training episodes and nonzero policy gradients. Counts/rewards in that snapshot
are training statistics, not validation performance. Network conditions are
simulated on Caliban; inference timing does not establish live protocol quality.
No final or accepted improvement is available yet.


## Interim update, 2026-10-27 UTC

Four completed models (all current MLP seeds and history MLP seed 42) match the
reference validation metrics exactly: 0% verified-goodput gain, all quality guards
pass, zero wrong deliveries. History seed 43 is training; seed 44 is queued.
[Snapshot](../../artifacts/reactive-sender-v7-training-interim.json). This does not
establish action identity or a final ranking; remaining runs continue unchanged.
