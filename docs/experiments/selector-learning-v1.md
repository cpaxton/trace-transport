# Selector learnability: fitting is not generalization

Actual date: 2026-10-26 UTC. Coverage collection and offline fitting are complete;
executed probe-then-select validation is **complete, negative**. No held-out acceptance or
learned transport win is claimed. This is supervised controller selection, not
another joint PPO run.

## Completed collection

The [Caliban coverage study](selector-coverage-v1.md) completed at
2026-10-26 14:49:22 UTC: 48 development cases and 24 independent validation cases,
nine full control trials per case, and 72 causal one-second baseline probes.
[Compact results](../../artifacts/selector-coverage-v1-results.json),
[curated synthetic dataset](../../artifacts/selector-coverage-v1-data.zip).
The archive retains exact dataset bytes and a SHA256 manifest. It contains
synthetic probes/outcomes, not private device data or raw training logs.

The strict per-case admissible hindsight gain is **47.2% on development but only
4.08% on validation**. The teacher is the hindsight target in 32/48 development
and 18/24 validation cases. This does not support generalizing the previous
12-case +122.6% headroom estimate. It is not an optimal bound: choices are a finite
set of static controls, and per-case filtering is stricter than some aggregate
benchmark guards. No acceptance threshold was changed to obtain a result.

## Tiny-fit gate

Four development examples, one for each of the first four distinct target classes,
were selected before examining validation results. All model types reached 100%
training accuracy in 400 full-batch updates. Start/end: 14:48:46–14:48:54 UTC.
[Results](../../artifacts/selector-tiny-fit-v1-results.json).
The tiny fit read an earlier snapshot with complete development data and
incomplete unused validation. Its entire development block exactly matches the
final published dataset; the compact coverage report records the content hash.
Whole-file hashes differ because validation was unfinished.
This checks supervised optimizer/representation plumbing on four points; it does
not validate PPO credit assignment, policy safety or generalization.

## Fixed offline comparison

Start/end: 14:52:28–14:53:29 UTC. Each model fits all 48 development cases for
400 full-batch AdamW updates, learning rate .001, weight decay .0001, gradient
norm cap 1. Seeds 1/2/3, final epochs only, no validation checkpoint selection or
hyperparameter tuning. Normalization uses only valid development observations.
Targets are the highest-byte admissible finite controls, falling back to the
teacher. Cross entropy weights cases equally, not by throughput. Future outcomes
and families construct diagnostic targets but never enter model inputs.

| Model | Development accuracy | Validation target accuracy, seeds 1/2/3 | Inadmissible validation choices, seeds 1/2/3 |
| --- | --- | --- | --- |
| Current-observation MLP | 100% | 58.3 / 54.2 / 58.3% | 5 / 5 / 5 |
| Flattened-history MLP | 100% | 45.8 / 54.2 / 45.8% | 7 / 6 / 8 |
| GRU | 100% | 45.8 / 45.8 / 45.8% | 7 / 7 / 8 |
| Transformer | 100% | 54.2 / 58.3 / 54.2% | 5 / 4 / 4 |
| Always teacher | — | 75.0% | 0 |

Model sizes are reported in the results; they are small but not exactly parameter
matched. The models use a final observation, 64-step flattened history, 48-wide
GRU, or 32-wide one-layer/four-head transformer respectively. Validity masks hide
prefix padding. At different control periods, 64 samples cover different physical
time spans; the available history is not uniformly one second.

[All model results](../../artifacts/selector-learning-v1-results.json),
[frozen plan](../../artifacts/selector-learning-v1-plan.json),
[frozen fitting source](../../artifacts/selector-learning-v1-source.zip),
[all twelve final models](../../artifacts/selector-learning-v1-models.zip).

Candidate-lookup throughput is **not executed performance**. The probe occurs
under the teacher, whereas stored candidate outcomes run a constant candidate
from time zero. Looking up a predicted class cannot retroactively change the
probe's traffic or reconstruct a mixed-controller trajectory. All lookups are
explicitly labeled diagnostic, and inadmissible choices are reported even when
throughput rises. A byte gap can be negative for an unsafe high-throughput choice;
it is not a constrained-utility regret metric. There is no demonstrated advantage
for history or the transformer in this small one-shot classification study.

## Executed validation, completed

[Execution plan](../../artifacts/selector-execute-v1-plan.json),
[frozen source](../../artifacts/selector-execute-v1-source.zip),
[runner](../../scripts/selector_execute.py).

All twelve final models, the teacher, and a declared low-RTT BDP rule run on all
24 validation cases; no winning model is selected after looking at results.
Every endpoint separately observes its own sender history during a one-second
teacher probe, then chooses one of the nine sender/ACK bundles once. Its rate,
window, cooldown and measurements are retained; ACK changes use v6 pending-timer
semantics. Other endpoints choose independently using the same weights, without
family tags, future state or free coordination. BDP starts with the endpoint's
existing estimator state rather than a shadow trajectory. These local mixed
choices differ from the globally uniform candidates used for training labels.
**Role coverage is limited:** training probes observe only endpoint 0 (a bulk
producer), while this exploratory execution applies the classifier to every
endpoint, including reverse interactive senders. Those reverse-role feature
contexts were not represented by separate training probes. Therefore a rollout
regression can reflect role-transfer and weak global-label mismatch; it is not
an isolated test of the value of memory or a general failure of RL. A future
role-matched or sender-only experiment must be registered separately, with local
action-aligned labels and prospective validation, rather than changing this
running experiment after seeing outcomes.

The probe consumes actual simulated time and traffic. Model inference wall time
is measured but is not charged as simulated delay; this remains a simulator
experiment, not a real-time deployment claim. Models are compared with a freshly
executed local teacher rather than silently pooling Caliban and local timings.
The simulator uses the same pinned CPU Torch/NumPy versions. Caliban collection
uses Python 3.12.3; fitting/execution uses Python 3.12.10 on x86-64. A separate
[runtime inventory](../../artifacts/selector-learning-v1-runtime.json) records
the unchanged fitting/execution environment observed after fitting.

The first serial engineering run was retired for inefficient CPU utilization;
a parallel bootstrap attempt failed before any case because its worker was not
pickleable through runpy. Raw outputs are retained locally. A new immutable
source uses a canonical module worker; four CPU workers run under the six-core,
8 GiB, two-hour service `trace-selector-execute-parallel-v2`, with file logs.
These engineering attempts are not negative model-performance studies.

A further launch completed the teacher and rule modes, then its strict checkpoint
loader rejected a `TorchVersion` object in metadata before executing any learned
model. All twelve checkpoints were re-exported with JSON-primitive plan metadata;
original files are retained and every state/normalization tensor was verified
identical. The [portability record](../../artifacts/selector-learning-v1-portability.json)
links old/new hashes, and regression tests require safe `weights_only=True`
roundtrips. The published model archive contains the portable exports.

Completed service: `trace-selector-execute-portable` (15:11:24–15:26:41 UTC). It reuses only those
[completed baseline trials](../../artifacts/selector-execute-v1-baselines.json),
with their [frozen source](../../artifacts/selector-execute-v1-baseline-source.zip).
Resume checks exact policy AST, baseline constant, simulator source hashes and
case memberships; learned outcomes cannot be resumed through this path. All
learned trials execute freshly. The plan records reused-artifact hashes and the
portable model hashes. No outcome, threshold, model or case was selected to repair
the engineering failures.

## Interpretation and next gate

Memorizing 48 contexts is easy; safe validation choices are not. A one-second
prefix also cannot predict independently randomized future changes. Classification
accuracy ignores whether a tiny apparent advantage justifies switching risk.
The current findings argue against enlarging this transformer or increasing the
same training budget. Executed validation confirms that none of these policies passes the full guards;
role-transfer and weak-label limitations prevent broader conclusions about RL.

Future proposals, not launched here: collect decisions at multiple times during
persistent regimes; test short-horizon reactive choices with training-only
counterfactual labels and explicit packet-RNG limitations; calibrate conservative
deferral on development folds; compare fixed physical-time histories; or define
an opt-in robot image-freshness scheduling task with strong deterministic controls.
Any change gets a new plan and frozen acceptance rules. The original reserved
sets remain unopened; this study does not weaken the 10% goodput target.


### Interim executed results

[Timestamped partial results](../../artifacts/selector-execute-v1-partial-results.json)
contain only completed modes, with remaining models still running. The first
completed current-observation MLP seeds fail deadline noninferiority despite
throughput near the teacher. This is a partial negative result, not a final ranking
of all architectures. The simple low-RTT rule also fails the throughput target.
No reported completed mode establishes an accepted win. The same fixed list of
models/cases continues; no validation-driven retraining is launched here.


## Complete executed result

Actual end 2026-10-26 15:26:41 UTC. All 14 modes (12 learned models, teacher,
simple rule) finish all 24 cases. [Full results](../../artifacts/selector-execute-v1-results.json),
[compact report](../../artifacts/selector-execute-v1-report.json). The earlier
partial snapshot is retained as a historical record; it is superseded by these
complete results.

| Model | Goodput gain range, three seeds | Interactive miss-rate range | Seeds passing all quality guards |
| --- | ---: | ---: | ---: |
| Current MLP | -0.37% to +1.66% | 23.41–23.95% | 0/3 |
| History MLP | -3.33% to +0.81% | 21.14–25.11% | 0/3 |
| GRU | -2.91% to +6.59% | 20.09–23.23% | 0/3 |
| Transformer | -13.10% to -12.19% | 20.42–24.69% | 0/3 |

Teacher: 1,149,952 B/s, 17.56% misses. The simple rule loses 10.70% goodput.
All modes have zero wrong deliveries. Ranges describe the three fixed training
seeds, not confidence intervals. No learned mode passes every guard, reaches the
accepted-win criterion, or establishes a memory advantage. No weights or guards
were changed in response to these validation results.

The next proposed study should isolate role-aligned repeated decisions and
short-horizon utility, with conservative teacher fallback calibrated only on
new development data. It needs a new prospective validation bank; reusing this
bank for more tuning would not provide fresh evidence.
