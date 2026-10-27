# Caliban: 50k-decision reactive sender training

**Completed, no learned advantage.** Actual start/end: 2026-10-27
01:31:44–04:44:41 UTC (October 26 evening to October 27 local midnight).
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
The completed study and final diagnostics are reported below.


## Interim update, 2026-10-27 UTC

Four completed models (all current MLP seeds and history MLP seed 42) match the
reference validation metrics exactly: 0% verified-goodput gain, all quality guards
pass, zero wrong deliveries. History seed 43 is training; seed 44 is queued.
[Snapshot](../../artifacts/reactive-sender-v7-training-interim.json). This does not
establish action identity or a final ranking; remaining runs continue unchanged.


## Complete result, 2026-10-27 UTC

All six runs finish, totaling **300,011 producer macro decisions**. All six match
the deterministic reference validation metrics exactly: 701,212 verified B/s,
24.25% interactive misses, zero wrong deliveries, and all relative quality guards
passing. This is **0% gain**, not an accepted win. Passing nonregression guards
does not imply good absolute latency or complete bulk delivery in every case.

[Full results](../../artifacts/reactive-sender-v7-training-results.json),
[reference records](../../artifacts/reactive-sender-v7-training-reference.json),
[final head audit](../../artifacts/reactive-sender-v7-training-heads.json),
[curated final actors](../../artifacts/reactive-sender-v7-training-actors.zip).
Actor exports exclude critic/optimizer/RNG; the archive manifest records original
checkpoint hashes and verifies every policy tensor unchanged. Raw checkpoints and
training logs remain on Caliban. Last-update metadata in the exports is preserved;
it is not the current service state.

Five heads have a strictly positive sufficient teacher-logit lower bound over
bounded Tanh activations, so their greedy deployment remains the teacher for all
supported histories. Seed-42 current MLP does not satisfy that bound; its identical
validation metrics do not establish action identity. Negative bounds do not prove
adaptive behavior. Parameters/gradients can change without changing greedy choice.

The job completed without a reported failure and is inactive. No unchanged larger
run is launched. The next useful diagnosis is a repeated-case RL credit/signal
check and sampled-versus-greedy evaluation, with explicit action-value uncertainty
and fresh prospective testing before another performance claim. More compute alone
is not supported by this result. Receiver learning remains a separate later task.
