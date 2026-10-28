# V7 longer credit pilot

Registered 2026-10-28. Status: **engineering running**, actual start 2026-10-28T22:57:17.703736+00:00.
[Actual plan](../../artifacts/v7-long-credit-engineering-plan.json).
The pilot is queued in the same bounded service and starts only after successful
engineering completion; no training-performance promotion rule or results yet. This follows the exploratory
[credit smoke](v7-policy-diagnostics.md), not a confirmed training defect.

## Fixed comparison

[Configuration](../../experiments/v7-long-credit-v1.json) specifies a 2×2 study:

| Arm | Rollout transitions/world/update | GAE lambda per 20 ms |
| --- | --- | --- |
| Baseline | 16 | 0.98 |
| Longer rollout | 64 | 0.98 |
| Longer retention | 16 | 0.998 |
| Both | 64 | 0.998 |

Train each arm from scratch for 50,000 eligible producer decisions with seeds
42 and 43: eight runs, approximately 400,000 decisions total (actual small
overshoot recorded). All use the current-observation MLP, same critic, four
worlds, same 48 training cases, optimizer, minibatch, epochs, reward, action space,
20-ms modeled inference cost, and fixed receiver. Macro steps are about 100 ms,
so windows are roughly 1.6 versus 6.4 seconds. GAE decay remains physical-time
based; lambda's own half-life grows from 0.69 to 6.92 seconds. Gamma stays 0.999
per 20 ms; the effective eligibility trace includes both gamma and lambda.
The actor's one-second history schema and physical task horizon are unchanged.

This controls interaction budget, not optimizer work. Longer collection windows
produce fewer update cycles and different policy/data trajectories, and may
increase on-policy gradient variance. Common seeds are not common random numbers.
Report actual decisions, updates, episodes, action distributions and in-sample
critic target MSE/explained variance; the latter are not full-return critic accuracy.
No value-architecture changes or optimizer tuning are mixed into this comparison.

## Evaluation and limits

Run all eight fixed final checkpoints, with no model selection or early
promotion. Evaluate greedy and sampled deployment (seeds 101 and 102 separately)
on the previous 24 validation cases, alongside teacher and initialized sampled
baselines. This is explicitly reused development validation, not prospective
held-out acceptance. Report verified goodput, exact completion, wrong delivery,
interactive misses, wire cost and fairness; reward does not establish performance.
Keep all four factorial arms and both optimizer seeds visible. Compare arms within
a seed and deployment mode; do not pool action sampling seeds as independent
network cases. Two optimizer seeds support a pilot, not a broad superiority claim.
Any improvement should motivate a preregistered fresh-case replication rather than
turning these reused cases into acceptance evidence. Reserved test sets remain closed.

## Execution

Before the pilot, run the engineering profile: the same four arms with 32 decisions,
one optimizer seed and two reused validation cases. This is plumbing only; no
performance-based promotion rule. Source hashes, warm inference p95 <=20 ms and
unit/integration tests must pass. Freeze code/config before launch. The runner
writes the full actual plan, timestamps, final-checkpoint hashes, raw per-case
metrics and progress, and mirrors JSON/checkpoints to a private persistent study
directory. Keep all checkpoints, environments and raw training runs out of Git.

Use `scripts.v7_long_credit` with `--config`, `--root`, `--persistent`, and
`--profile engineering|pilot` from a frozen bundle working directory with its
source manifest. Run through `scripts.run_limited`: six CPU equivalents, 8 GiB,
single-threaded BLAS/Torch, file logs, one active experiment, 12-hour maximum.
Serialize engineering then pilot; never mutate or resume an old study directory.
Update this record with actual dates and outcomes. No claimed win at registration.
