# Fresh-future precision replication

Registered 2026-11-01. Status: **complete; no reliable independent-bank gain**.
Engineering ran 2026-11-01 23:53:46–2026-11-02 00:05:31 UTC; pilot ran
2026-11-02 00:05:34–10:11:15 UTC. Results recorded 2026-11-13.
[Engineering plan](../../artifacts/v7-precision-replication-engineering-plan.json),
[engineering results](../../artifacts/v7-precision-replication-engineering-results.json),
[pilot results](../../artifacts/v7-precision-replication-pilot-results.json).
All four targeted tests passed on Caliban. This follows the completed
[local-predictability study](v7-local-predictability.md); its bounded rate-control
headroom remains unresolved. No new training or adaptive-policy execution occurred.
The frozen design follows.

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

## Completed replication

All 12 cases and 24 original snapshots completed with 9,216 fresh terminal
continuations. The engineering check added 48 continuations. Models and constant
selection stayed frozen. The pilot finished within the 24-hour cap in about
10 hours 6 minutes. The raw data, source archive and original weights remain
in private persistent storage.

| Choice rule | Independent-scoring reward difference vs teacher | Exploratory 95% interval |
| --- | --- | --- |
| Training-selected constant | +0.01450 | [−0.04871, +0.08780] |
| Privileged fitting-table choice | −0.00099 | [−0.05352, +0.04382] |
| Current-only, seeds 1/2/3 | +0.02736 / +0.03240 / +0.01617 | All include zero |
| History MLP, seeds 1/2/3 | +0.02018 / +0.02297 / +0.02097 | All include zero |
| Transformer, seeds 1/2/3 | +0.01508 / −0.00019 / +0.01809 | All include zero |

Units are conditional discounted reward per snapshot, not throughput percentages.
These are reused diagnostic cases and isolated actions under teacher continuation.
No mode establishes a transport win. The privileged choice scores −0.01549 versus
the frozen constant; its independent-bank value remains uncertain. Fit/score
maximizing actions agree at 5/24 snapshots (20.8%). The same-scoring-bank maximum
is +0.05426, which remains an optimistic selection-biased statistic rather than a
learnable bound or validated headroom. Replication intervals remain broad despite
eight times as many futures; only 12 independent network cases were used.

Every learned predictor's mean verified-byte difference is negative (roughly
−12,000 to −18,464 bytes per conditional continuation). Eight have negative mean
interactive-miss differences, and one transformer has a small positive difference.
These are task-metric point estimates, without acceptance tests or confidence
claims. Reward and verified goodput are distinct; the reward improvements often
coincide with fewer misses and fewer verified bytes.

Several fixed predictors' reward point estimates changed sign versus the earlier
8-future study. This reflects fresh simulation draws, not improved training. The
replication supplies no reliable advantage for privileged action selection or the
fixed local predictors and does not justify another unchanged RL sweep. It also
does not prove small transient rate-control gains are impossible: the teacher
resumes after the next eligible decision, interventions are brief, and case/model
coverage remains limited. No reserved transport acceptance evaluation occurred.

## Possible next questions (unrun)

- Predict future observable delivery/RTT directly from local history, compared
  against persistence and moving-average baselines. Use case splits and report
  calibration before asking a controller to act on the predictions.
- Evaluate sustained rate decisions over a declared duration. The current
  counterfactual concerns a single short decision followed by the teacher, so it
  cannot rule out gains from a sequence of coordinated decisions. This requires
  a newly registered task/control comparison, with the strong baseline retained.
- Explore robot-object scheduling: prioritize commands, complete image objects
  and map deltas under changing service, with explicit application latency goals
  and exact-delivery checks. Compare priority, deadline and fair-queueing rules
  before training. This changes the research question and has not been run.

[Replay audit](../../artifacts/v7-precision-replication-replay.json) reproduces all
report fields from saved data/frozen models, verifies all 24 snapshot identities,
checks 9,216 unique futures disjoint from the earlier study, and records the
private raw-data hash. This is verification of the completed run.
