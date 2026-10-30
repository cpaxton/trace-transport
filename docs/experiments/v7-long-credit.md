# V7 longer credit pilot

Registered 2026-10-28. Status: **complete, negative; no accepted learning win**.
Engineering ran 2026-10-28 22:57:17–23:00:57 UTC; pilot ran
2026-10-28 23:00:59–2026-10-29 03:40:23 UTC. Both services' stages completed
successfully, without a runtime-cap failure. [Engineering plan](../../artifacts/v7-long-credit-engineering-plan.json),
[engineering results](../../artifacts/v7-long-credit-engineering-results.json),
[pilot results and actual plan](../../artifacts/v7-long-credit-pilot-results.json).
This follows the exploratory [credit smoke](v7-policy-diagnostics.md), not a
confirmed training defect. The frozen preregistered design follows below.

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

## Completed pilot result

All eight fixed runs finished: **400,016 eligible producer decisions**, including
small recorded overshoots. Evaluation contains 648 full episodes: 24 reused
validation cases × (teacher/initial sampled 101/initial sampled 102 plus three
modes for each of eight trained models). Engineering added 20 episodes. No model
selection, new tuning, or reserved acceptance evaluation occurred.

| Rollout | Lambda | Greedy goodput change vs teacher (both optimizer seeds) | Sampled goodput range vs teacher (two optimizer × two sampling seeds) |
| --- | --- | --- | --- |
| 16 | .98 | 0% | −0.45% to +0.55% |
| 16 | .998 | 0% | −1.02% to −0.24% |
| 64 | .98 | 0% | −0.12% to +0.36% |
| 64 | .998 | 0% | −1.15% to −0.40% |

These ranges describe separate runs, not pooled estimates or confidence intervals.
All eight greedy models exactly match the teacher's aggregate metrics: 701,212
verified bulk B/s, 24.25% interactive misses, 96.07% message completion and zero
wrong deliveries. Relative quality guards pass; goodput gain is zero. Equal
metrics do not certify that every action equals teacher.

All 16 sampled evaluations have zero wrong deliveries but fail the deadline guard
and at least one per-family progress guard. Interactive misses range from 24.43%
to 25.40%, versus teacher 24.25%. Sampled trained-vs-initial goodput ranges from
−0.65% to +1.67%; these changes do not establish a quality-preserving win. Every
comparison to teacher returns win=false. Longer collection with unchanged lambda
has mixed small effects, while higher lambda yields negative goodput changes in
all eight sampled evaluations. This does not prove higher lambda is generally
harmful; this pilot has only two optimizer seeds and reused development cases.

For seed 42, 16-step arms made 456 updates and 64-step arms 114–115; for seed 43,
426 versus 107. Thus interaction budgets are matched, while optimizer work differs
substantially. Both seed-42 longer arms finished 192 episodes; seed-43 arms 180.
In-sample critic diagnostics use different bootstrapped targets across lambda
settings and are not evidence of improved full-return calibration.

The tested rollout/retention changes alone did not unlock a deployable advantage
at 50k decisions. The small earlier action-value discrepancies therefore remain
exploratory leads, not an established explanation or a solved bug. Do not promote
another unchanged longer run from this result. A useful next investigation would
measure actor action diversity and full-return critic calibration across more
states, then test a single targeted change with fixed optimizer work or explicit
update-budget controls. Such follow-ups remain unrun.
