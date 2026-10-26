# Reactive v6: isolate contribution before scaling

Started 2026-10-26 UTC. This is a diagnostic study, not a retraining or held-out
acceptance result. It implements the first gates in the
[improvement plan](../research/improvement-plan-2026-10-26.md).

## Configuration

- Final seed-42 v6 checkpoint from the 200k-decision pilot; fixed, no reselection.
- Sender-only and receiver-only ablations on all 24 original validation cases,
  compared with the existing joint and adaptive records.
- Four original cases replayed with instrumentation. Count only receiver actions
  applied on active endpoints, report probabilities/entropy, and distinguish
  setting changes, pending timer replacements and immediate control-triggered
  flushes. Assert complete metric-record equality with standard joint evaluation.
- Separate 12-case development bank, one case per physical family, selected from
  seeds >=150000; no overlap with the original validation bank. Compare delay-fast,
  AIMD and BDP controllers with ACK settings (1,0), (8,5ms), (32,25ms). Initial
  pacing 512000 B/s and EDF scheduling remain fixed. Each run produces for 12 s
  and drains for 3 s. These are our custom controllers, not BBR/Cubic reproductions.
- A finite-control hindsight selector is a descriptive optimistic comparator,
  neither a deployable policy nor a proven bound. It filters per-case deadlines,
  completion, wire, integrity, accepted progress, image/map/confirmed bytes and
  fairness; it does not imply statistical noninferiority after selection.

Source is frozen separately from the active workspace. Runtime reuses the pinned
CPU-only environment without modifying it. One bounded service runs at a time,
six-core affinity, 8 GiB RAM, swap 512 MiB, file logs, two-hour runtime cap. Initial
launch stopped before any simulation because the log directory already existed;
the guard now admits a directory containing only its service log. The second
immutable source bundle is the study source.

## Review correction: recorded versus actual pilot configuration

PR review identified that the old pilot runner serialized `progress_shaping=false`
before training a copy with `progress_shaping=true`. Both the actual training
`config.json` and final checkpoint metadata confirm **true**. The published plan
and embedded result plan are corrected with an explicit
[erratum](../../artifacts/reactive-v6-pilot-plan-erratum.json) recording the old
configuration, original plan hash, actual config hash, and checkpoint hash. Raw
run files and the original frozen source remain unchanged; measurements are not
recomputed or altered. Future runners serialize the same config passed to training.
A regression test inspects the plan at the training call. The validation bank now
also fails closed when a required family cannot be filled.

## Caliban interruption

The long v4 study stopped on 2026-10-25 at 23:15:59 UTC with
`TimeoutError('CPU environment worker stalled')`. Read-only status on October 26
shows the service failed, seed 42 complete at 5,000,031 decisions, seed 43 partial
at 3,071,329, and no seed-44 result. It has not been restarted. The cause of the
worker stall is not yet diagnosed; this does not establish an OOM or device fault.
The incomplete three-seed study and known infeasible v4 completion guard prevent
an acceptance claim. [Status artifact](../../artifacts/caliban-v4-interruption-20261026.json).

## Results

Role ablations are complete: sender-only reproduces joint records exactly on all
24 cases; receiver-only reproduces the adaptive reference exactly. There is no
observable receiver contribution to these greedy-deployment outcomes.

The four instrumented cases contain 14,677 active receiver decisions, all teacher
choice 0; no setting changes with pending receipts, timer replacements or
control-triggered flushes. Mean teacher probability is 0.506–0.531, entropy
2.094–2.171 nats. The categorical distribution still spreads substantial
probability over alternatives; greedy deployment ignores that mass. All four
instrumented metric records match standard joint evaluation exactly.

Development sweep completed 2026-10-26 14:14:44 UTC; role/action/sweep study began
14:03:08 UTC. The sampled-receiver audit ran 14:14:59–14:16:14 UTC the same day.
All reserved tests remain unopened.

[Full diagnostic results](../../artifacts/reactive-v6-diagnostics-results.json),
[compact report](../../artifacts/reactive-v6-diagnostics-report.json),
[frozen source](../../artifacts/reactive-v6-diagnostics-source.zip),
[exported actors](../../artifacts/reactive-v6-final-actors.pt).

### Separate development control sweep

| Control | ACK batch/wait | Verified B/s | Interactive misses |
| --- | --- | ---: | ---: |


The applied-action audit counts initial installation of the teacher ACK settings
as setting changes at the first tick; these are not learned deviations. The
reported `changes_with_pending` and `control_flushes` distinguish transport effects
from these empty-backlog initialization events.

A separate, predeclared deployment diagnostic samples receiver actions on those
same four cases with sampling seeds 1, 2 and 3, retaining the greedy sender. All
three outcomes will be reported without choosing a winning seed or changing the
policy deployment contract. This checks the train/sample versus test/argmax gap;
it is not a held-out stochastic-policy performance claim.
| delay-fast | 1 / 0 ms | 720,100 | 14.84% |
| delay-fast | 8 / 5 ms | 615,538 | 13.84% |
| delay-fast | 32 / 25 ms | 208,213 | 14.38% |
| aimd | 1 / 0 ms | 910,450 | 18.73% |
| aimd | 8 / 5 ms | 751,616 | 18.44% |
| aimd | 32 / 25 ms | 304,014 | 19.46% |
| bdp | 1 / 0 ms | 1,348,722 | 21.23% |
| bdp | 8 / 5 ms | 1,193,301 | 21.29% |
| bdp | 32 / 25 ms | 597,333 | 20.90% |

No alternative fixed controller passes every quality guard against the reference.
AIMD/immediate ACK offers +26.4% goodput but worsens deadlines; BDP/immediate ACK
offers +87.3% but worsens deadlines and completion. ACK batching generally reduces
aggregate throughput on this bank, but can help individual cases.

The descriptive per-case admissible hindsight selection gives 1,603,015 versus
720,100 B/s (+122.6%), with aggregate misses 14.71% versus 14.84%. Gains concentrate
in wired LAN (3,642,709→11,217,579 B/s) and bufferbloat (262,144→2,752,512 B/s),
with smaller gains in mobile fade and capacity opportunities. This uses future
whole-run outcomes to choose differently for each case. It is selection-biased,
not a deployable policy, not an optimal bound, and not evidence that local history
can identify the winner. Single-case family coverage is especially weak. No
post-selection confidence interval or accepted-win claim is attached to it.

### Sampled receiver, greedy sender

The four-case aggregate results are:

| Deployment mode | Verified B/s | Interactive misses |
| --- | ---: | ---: |
| Adaptive | 137,557 | 9.37% |
| Joint greedy | 138,581 | 9.80% |
| Sample seed 1 | 128,683 | 10.23% |
| Sample seed 2 | 126,635 | 10.10% |
| Sample seed 3 | 141,995 | 9.97% |

All modes have zero wrong deliveries. Two sampling seeds reduce goodput and one
increases it; all have worse aggregate interactive misses than joint greedy.
This does not establish an improved stochastic deployment policy. The four
reused cases and three sampling seeds cannot establish broad generalization.
[All sampled results](../../artifacts/reactive-v6-stochastic-results.json),
[predeclared plan](../../artifacts/reactive-v6-stochastic-plan.json),
[frozen source](../../artifacts/reactive-v6-stochastic-source.zip).

## Decision and next gate

Freeze ACK learning for the next selector diagnostic. Prioritize discovering
whether local observations identify safe sender/controller choices on independent
cases, before increasing transformer size or joint RL budget. The
[larger Caliban study](selector-coverage-v1.md) collects the data for tiny-fit and
history-model comparisons; it is running and is not yet a selector win. Lack of
receiver value in these deployed policies does not prove that dynamic ACK control
or receiver grants could never help a different task.
