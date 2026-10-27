# V7 producer-only training on Caliban

**Completed engineering pilot.** Actual start/end: 2026-10-27
00:59:44–01:19:47 UTC (October 26 local evening).
This is simulated network training using Caliban's CPU; it is not a live-network
or robot deployment and no accepted improvement is claimed.

[Actual run plan](../../artifacts/reactive-sender-v7-plan.json),
[registered config](../../experiments/reactive-sender-v7-pilot.json),
[development gate](../../artifacts/reactive-sender-v7-gate.json),
[frozen source](../../artifacts/reactive-sender-v7-source.zip),
[operational snapshot](../../artifacts/reactive-sender-v7-status.json).

## What trains

Two small local actors (current-observation MLP and physical-history MLP), each at
optimizer seeds 42/43/44, learn cooperatively across all bulk-producer sender roles.
Each run receives 2,000 eligible producer macro decisions, with at most one batch
of overshoot. Reverse senders, ACK control, reliability and queue scheduling remain
deterministic. This deliberately isolates sender learning before returning to
joint sender/receiver learning.

Actions bias the teacher target rate by 0.8/1.0/1.2; they do not multiply the current
rate repeatedly. Teacher windows remain unchanged. Producers reconsider at 100 ms
or twice their own measured RTT, capped at 500 ms. The 20-ms modeled inference/
dispatch delay is quantized to a kernel boundary. Physical-time history has 64 bins
over one second, integrates decoded ACK/wire/loss rates, and uses past instantaneous
state. Interval-rate spreading is a retrospective approximation, not per-packet
timestamp recovery. No family, true link capacity, future schedule or critic state
enters actors. A centralized training-only critic also sees control-state summaries.

The algorithm is cooperative macro MAPPO with physical-time GAE, raw whole-object
v6 task reward, no shaping and no subtracted teacher. Micro rewards are discounted
inside each macro transition. Initial sampled probabilities are 0.6 teacher and
0.2 each alternative; greedy initialization is exactly the teacher. Policy, critic
and sampling seeds are separate, with critic initialization matched across kinds.
Sixteen-macro-step rollouts provide frequent updates/checkpoints.

## Gates and scope

Teacher parity, reverse-role rejection, target convergence, rate-volume aggregation,
physical-time return equivalence and checkpoint roundtrip tests pass locally and
on Caliban. The development intervention audit compares the same producer state
with independent post-state packet-stream replicas, not CRN coupling. Six states
from four training families include two small positive, no-miss/no-verified-regression
signals above the descriptive two-replica noise threshold. Both MLPs fit these tiny
examples exactly. Two replicas and a finite horizon are not statistical proof of
headroom, optimal labels or a learning win. The gate was replayed after a collector/
initialization-only revision to bind it to final source, not as independent data.
Actual start/end date of that engineering audit: 2026-10-27 UTC.

Warm p95 inference for a 16-endpoint batch is 0.78 ms (current MLP) and 0.83 ms
(history MLP) on Caliban, below the fixed 20-ms modeled budget. This is a descriptive
benchmark, not a real-time guarantee. A failed sensitivity/timing gate would have
limited training to 512 decisions/run; the actual on-device plan records 2,000 for
all six. The larger planned 50k-per-seed experiment is not launched here.

Training uses 24 newly allocated cases, with 12 distinct calibration cases unused
and 12 new validation cases. Allocations were checked against seed-labeled fields
in prior public config/artifact JSON and committed before intervention collection.
Every final checkpoint is evaluated against the same deterministic reference;
there is no validation checkpoint/model selection. The original reserved test sets
remain unopened. Small case counts and training budgets limit any inference.

The first operational snapshot shows a completed seed, nonzero actor gradients,
finite weights and persistent checkpoints. Training reward/episode statistics do
not establish held-out improvement; final validation is recorded separately.

## Resources and monitoring

Unit: `trace-reactive-sender-v7`, four CPU environment workers, six-core affinity
(6–11), 8 GiB memory limit, 512 MiB swap, low priority, 12-hour cap, file logs and
no terminal/tmux attachment. JSON and initial/latest/final checkpoints are mirrored
atomically under a 128 MiB budget while preserving 1 GiB free disk. Latest files
are retained for inspection; optimizer-resume support is not implemented in this
initial pilot. Source and the installed CPU runtime were not hot-patched, and old
studies were left unchanged.

```bash
.venv/bin/python -m scripts.caliban_deploy status
ssh caliban systemctl --user status trace-reactive-sender-v7 --no-pager
```

Raw runs/checkpoints stay local to the devices. Public artifacts contain frozen
source, exact configuration, small gate results and labeled operational snapshots.
No v7 advantage is claimed; the completed engineering outcome is recorded below.


## Completed small pilot and meaningful training budget

[All six small-run results](../../artifacts/reactive-sender-v7-pilot-results.json)
match the teacher exactly on validation: zero relative gain, no accepted win.
A [training-weight-only audit](../../artifacts/reactive-sender-v7-initial-head-audit.json)
certifies that all six greedy policies always choose the teacher for any finite
input. For a Tanh hidden vector, the lower bound on teacher versus alternative
logit is `b_teacher - b_other - sum(abs(W_teacher - W_other))`; every bound is
strictly positive. This is a sufficient certificate, not an estimate from
validation examples. A nonpositive bound would not itself prove adaptation.

Therefore this short smoke cannot judge the learned controller's potential. A
separate [registered 50k configuration](../../experiments/reactive-sender-v7-training.json)
keeps models/optimizer/reward/quality gates fixed, expands training coverage to 48
cases and allocates 24 fresh prospective validation cases. It runs from scratch,
not by silently changing the live small pilot. The stronger budget follows source,
role, timing and training-weight diagnostics; it is not justified by selecting a
validation winner. No longer-run improvement is claimed before its results.
