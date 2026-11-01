# Fresh-future precision replication

Registered 2026-11-01. Status: **engineering running**, actual start 2026-11-01T23:53:46.195136+00:00.
[Actual engineering plan](../../artifacts/v7-precision-replication-engineering-plan.json).
Pilot is queued after successful engineering completion. All four targeted tests
passed on Caliban; no performance result or training promotion is claimed. This follows the completed
[local-predictability study](v7-local-predictability.md), whose privileged action
table did not generalize between future banks. Its apparent headroom remains
unresolved, rather than an established failure of all learned protocols.

## Fixed experiment

[Configuration](../../experiments/v7-precision-replication-v1.json) reuses **all
24 test snapshots from all 12 prior diagnostic test cases**, without selecting
states by outcome. The simulator, teacher prefixes/continuations, one-decision
rate action, hold times, inference delay, observations, task duration, reward and
discount remain unchanged. Before simulating each snapshot, require an exact
fingerprint match of producer, time, local history and validity mask to the
original. Any mismatch fails closed; no replacement snapshots are allowed.

Use **64 independent fitting futures and 64 independent scoring futures per
action**, eight times the original replica count. Replica indices 16–79 have
fresh seeds disjoint from the original 0–7 and the engineering replication 8–9.
Banks and actions remain independent, not common random numbers. There are
24 × 3 × 128 = **9,216** new terminal continuations in the pilot. The two-case
engineering check adds 48 continuations before the pilot starts. It checks
plumbing only and cannot promote or reject based on reward/performance.

Keep the original nine supervised predictors and training-selected constant
(bias 0.8) frozen. Preregister their file SHA-256s and original dataset/result
hashes. No new predictor fit, architecture change, normalization, label tuning,
constant selection or RL occurs. Verify original simulator/predictor source
hashes before startup. Private weights and datasets stay outside Git.

## Readouts and interpretation

Score teacher, frozen constant, all nine frozen predictors, and the privileged
new-fitting-table choice on the new independent scoring bank. Reuse the original
case-balanced summaries and case/replica bootstrap; keep both snapshots clustered
within their case. Report comparisons to teacher, constant and current-only
models, all three model seeds, per-snapshot action values/standard errors,
fit/score action-max agreement, and selection-biased same-score-bank maximum.

The primary question is whether privileged fitting-table choices retain positive
return difference on independent scoring futures as simulation noise shrinks.
The secondary question is whether the fixed local predictors do. A positive
privileged-table result alone establishes neither local predictability nor an
oracle upper bound. Report task-metric deltas separately from reward. Nine
predictors are fixed descriptive robustness runs; exploratory intervals have no
multiple-comparison correction and never select a winner for deployment.

This is a replication of future noise **on the same cases and states**, not fresh
case generalization or adaptive-policy execution. More replicas reduce simulation
noise but do not add independent network cases, improve case coverage, or prove
absence of a small effect. If intervals remain broad or rankings unstable, report
inconclusive evidence. Negative results concern these bounded teacher-relative
single decisions, not other controls, observations or learned scheduling. No
result triggers automatic retraining, policy promotion, or reserved acceptance.

## Execution

Run engineering then pilot in a fresh immutable source directory and one service:
four workers, six CPU equivalents, 8 GiB, single-threaded numerical libraries,
file logs, and a 24-hour cap. Preserve prior studies unchanged. Write new data,
partial future groups and per-case heartbeat/progress directly to private
persistent storage. Record actual start/end times, completion coverage, failures
and curated findings here, in the experiment register and paper when available.

CLI: `python -m scripts.v7_precision_replication --config CONFIG --reference-root
OLD_PILOT --output NEW_OUTPUT --profile engineering|pilot`, run from a frozen
bundle with its verified source manifest. The reference directory contains the
old data/result JSON and nine original predictor files.
