# Learning signal and benchmark diagnosis, 2026-10-25

The next change targets measurement and credit assignment, not simply longer
training or more independent noise. Caliban's v4 experiment remains frozen.

## Findings from completed seed 42

The initial policy's two wired-LAN cases supply 55.9% of verified bulk bytes.
That is not itself a scoring bug when the objective is aggregate bytes/s, but it
can hide regressions on the robot-relevant conditions. We now report every family
separately and require future v5 candidates to preserve both accepted bulk progress
and completed-object bytes within each family, as well as the global constraints.

Neither Wi-Fi contention case finishes a bulk object during the 12-s measurement
window. In case 121007, even an optimistic 1,114,779-byte forward service budget
is smaller than the 1,181,696 wire bytes needed for one smallest object per host.
This lower bound includes only payload plus minimum headers/CRCs, ignoring all
metadata beyond an empty dictionary, ACK airtime, status, loss and propagation.
Therefore the old all-host-completed-object guard is impossible in at least one
case. Passing this service bound does not prove feasibility; other cases may also
be constrained. No cases or old scores are removed or reclassified as successes.

A new baseline replay distinguishes packet service from object completion:

| Wi-Fi seed | CRC-accepted bulk payload per host during measurement | Complete bulk bytes |
| --- | --- | ---: |
| 121007 | 216,064 / 224,256 B | 0 |
| 121017 | 5,632 / 2,560 / 2,560 B | 0 |

Thus all five hosts receive some bulk data, despite the legacy `starved_hosts=5`
label. The combined accepted-payload rate is 18,795 B/s, not useful completed-object
goodput. Class-specific counts after drain also expose the lack of bulk completion
that can be obscured by many completed short status messages.

[Stratified curve and feasibility bounds](../../artifacts/learning-diagnostics-v1-results.json),
[baseline replay and class counts](../../artifacts/progress-baseline-v5-results.json).
These reuse v4 validation cases for diagnosis and are not held-out evidence.
The audit checks that the reconstructed sampler matches the frozen sampler AST.

## V5 corrections and learning changes

The v5 profile keeps v4's physical distribution and exact-byte reliability kernel.
Its new host guard checks zero CRC-accepted bulk payload during measurement;
whole-object zero-completion counts remain separately visible. A mean per-case
progress Jain-index guard and per-family nonregression checks accompany it.
This is a new acceptance version, not a retrospective change to v4. Partial
payload never counts as completed-object goodput, confirmed delivery, or a
successful image/text transfer. A single accepted packet per host also does not
establish strong fairness, hence the extra fairness/progress reports.

Optional potential shaping gives earlier training feedback:

```
Phi(state) = unique accepted bulk bytes belonging to incomplete objects
             / (262144 * production_seconds)
r_shaped = r_raw + gamma(dt) * Phi(next) - Phi(current)
gamma(dt) = 0.999 ** (dt / 0.02)
Phi(terminal) = 0
```

A completed object leaves the potential ledger; it is still credited exactly once
by the original raw completion reward. Duplicates add no potential. With zero
initial/terminal potential and matching discounts, the shaping terms telescope
out of the full discounted return. Tests verify that equality on an actual
simulated trajectory, including incomplete objects at termination. This follows
the potential-based shaping principle of [Ng, Harada and Russell](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
It changes optimization feedback, not the completed-object objective, and cannot
make an infeasible completion task feasible. Finite rollouts, function approximation
and PPO updates can still benefit or regress; improvement is not guaranteed.

V5 shaping requires time-aware GAE and raw task reward (no shadow-teacher
subtraction). A learner-only option masks the explicit ACK choice exactly equal
to the teacher's setting; action zero already represents that behavior. Sampling
prior normalization accounts for the mask, and checkpoint export/load restores
it. Other controllers retain every legal protocol action. Both arms of the new
pilot use this option, so it does not confound their shaping comparison.

## Controlled pilot

The local pilot runs two 200,000-opportunity arms from identical actor/critic
initialization: raw reward versus potential-shaped reward. Both use 64-step
histories, width 32, one layer, two CPU workers, 512-step rollouts, time-aware GAE,
receiver alias masking and the same initial teacher. Longer rollouts span at least
one second at the fastest 2-ms control tick, rather than truncating at 256 ms.

Only the final checkpoints are evaluated on 24 new stratified validation cases
starting at seed 134000. No intermediate-snapshot selection or reserved acceptance
tests are used. Case seeds/initial worlds match; policy-dependent packet draws and
opportunity counts can diverge after actions differ. This is not CRN coupling.
The single-seed pilot is exploratory even if numerical gates pass.

The source and a separate CPU-only dependency environment are frozen under /tmp,
so continued local development does not modify the experiment. The named service
has six-core affinity, 8-GiB memory max and a two-hour runtime limit. Caliban's
larger v4 experiment continues independently. The final completed counts and validation are reported below; no accepted learning
win is established.

[Acceptance plan](../../experiments/goodput-v5-plan.json),
[actual pilot configuration](../../artifacts/progress-ablation-v5-plan.json),
[frozen source](../../artifacts/progress-ablation-v5-source.zip),
[runner](../../scripts/progress_ablation.py).

## Next priorities after this comparison

Replicate receiver interventions across multiple future draws, retain local
histories, and test whether useful action differences are predictable from those
histories before training a receiver-specific action-value model. [COMA](https://ojs.aaai.org/index.php/AAAI/article/view/11794)
motivates separating an agent's action contribution with a counterfactual critic;
our current shared world advantage and cloned-state diagnostic are not COMA.

If completion rewards remain too sparse, investigate independently verified,
application-useful chunks with explicit whole-object manifests, or longer finite
transfer tests. Do not count arbitrary partial PNG bytes as a completed image or
silently shrink image payloads to manufacture a win. Inference latency and a
replicated timing/credit ablation remain deployment and learning requirements.

## Completed pilot result

Both arms completed on 2026-10-25 UTC: raw 200,001 opportunities / 329.25 s;
shaped 200,002 / 329.70 s. Both used 35,472 world steps and completed 18 episodes.
The actor/critic initial parameters were exactly equal. Final-only validation
finished at 17:27:16 UTC (72 case/controller evaluations, including the reference).

| Mode | Verified object goodput B/s | Interactive misses | Gain vs adaptive |
| --- | ---: | ---: | ---: |
| Adaptive | 855,268 | 10.48% | — |
| Raw reward | 848,555 | 9.77% | −0.78% |
| Potential-shaped | 860,900 | 9.80% | +0.66% |

Shaped versus adaptive has a paired goodput-difference interval +228 to +13,312 B/s
and passes the v5 quality guards, but misses the 10% gain target. Shaped versus
raw has +1.45% goodput, with interval +114 to +28,562 B/s, but fails deadline
noninferiority and per-family accepted-progress nonregression relative to raw.
Therefore this is not an established overall improvement over the raw arm or an
accepted learning win. The single seed, 24-case validation bank and multiple
reported comparisons limit inference; reserved acceptance cases remain unopened.

The four-case greedy audit still finds only teacher-following receiver proposals
(30,166 valid-history proposals, including inactive rows). Do not attribute the
small aggregate gain to useful receiver learning. Role ablations are recorded
separately. [Results](../../artifacts/progress-ablation-v5-results.json),
[greedy audit](../../artifacts/progress-ablation-v5-final-action-audit.json),
[final shaped actors](../../artifacts/progress-ablation-v5-shaped-actors.pt).

## Next concrete fix: receiver control authority

A settings change in v4/v5 affects subsequent packet arrivals but does not
reschedule an already-pending ACK timer. For example, selecting an immediate ACK
while no new packet arrives leaves that ACK waiting for the old timer. This limits
what a history-based reactive receiver can do, regardless of its training budget.

The separate `goodput-v6` variant applies changed settings to the current backlog.
It flushes eligible ACKs immediately or, when the wait changes, replaces the timer using the original
oldest pending arrival. A generation counter invalidates stale timers. Unchanged
settings preserve their original event ordering; newly emitted wire bytes remain
inside the raw reward's accounting. Mandatory immediate first-completion ACKs are
still enforced. This does not give the policy control over every feedback event.

Tests verify immediate action without a new arrival, extension beyond the old
timer without restarting pending age, no duplicate valid flush, exact parity for
unchanged settings, and the serialized v6 contract. No v6 training-performance
result is claimed. The completed v5 experiment used its frozen earlier semantics,
and Caliban's v4 source is unchanged.

[Reactive-control plan](../../experiments/goodput-v6-plan.json),
[implementation](../../src/trace_transport/goodput_reactive.py),
[tests](../../tests/test_reactive_receiver.py).

The completed 24-case role ablations reproduce the joint shaped records exactly
with only the learned sender enabled. Receiver-only records reproduce the adaptive
reference exactly. This identifies no useful receiver contribution to the shaped
pilot's measured gain. [Role ablations](../../artifacts/progress-ablation-v5-roles.json).
