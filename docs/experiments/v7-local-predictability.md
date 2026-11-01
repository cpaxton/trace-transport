# Is local history predictive of useful rate decisions?

Registered 2026-10-31. Status: **complete; no demonstrated local-predictability
advantage, headroom inconclusive**. Engineering ran 2026-10-31 18:59:12–19:43:11
UTC; pilot ran 2026-10-31 19:43:14–22:53:51 UTC. Both finished successfully.
[Engineering plan](../../artifacts/v7-local-predictability-engineering-plan.json),
[engineering results](../../artifacts/v7-local-predictability-engineering-results.json),
[pilot results](../../artifacts/v7-local-predictability-pilot-results.json).
All seven targeted tests passed on Caliban. This follows the negative
[longer-credit pilot](v7-long-credit.md); no new RL or deployment occurred.
A negative result concerns these observations, actions, teacher prefixes and
continuations, and finite data. It does not rule out learned protocols generally.
The frozen design follows.

## Frozen design

[Configuration](../../experiments/v7-local-predictability-v1.json) selects 24
training cases (two per synthetic family) and 12 test cases (one per family),
using the earlier 48/24 banks. These are reused diagnostic cases, not fresh
transport acceptance. Whole network cases stay in one split: all snapshots,
producers, fitting futures and scoring futures from a case stay together.
There is no epoch/model selection on the test cases or their outcomes.

Follow the deterministic teacher, issuing explicit bias-1 decisions at eligible
producer slots so the existing inference delay and decision holds are respected.
At the first eligible slot at/after 2 and 6 seconds (deadline +0.75 s), choose a
producer by a fixed rotating index among eligible producers. Store only its
64-bin, one-second local history and validity mask as predictor inputs. Record
snapshot metadata for analysis separately. No seed, family, absolute time,
critic state, hidden bandwidth, queue schedule or future outcome enters a model.
Missing eligible states are explicitly skipped and reported, never replaced by
performance-selected states.

Clone each snapshot and force each of the three existing biases (0.8, 1.0, 1.2).
Continue with the same teacher until the true 15-second task end. The forced
action lasts until the next eligible teacher decision resets it; this is not a
persistent rate-policy comparison. Physical inference delay, hold and discounting
are unchanged. Discount full macro rewards with gamma .999 per 20 ms. Separately
record verified bulk bytes, interactive misses and wire bytes after the snapshot.
These reward values do not establish a transport throughput/quality win.

For each action use eight independent fitting futures and eight independent
scoring futures. Future packet RNGs are independently reseeded across arms,
replicas and banks; already scheduled events and identity RNG state remain
conditional on the prefix. This is not common random numbers. Save replica seeds.
The full pilot has up to 72 snapshots and 3,456 terminal continuations; actions
at different snapshots are not sequentially composed into an adaptive policy.

## Predictors and controls

Fit the two action advantages Q(0.8)-Q(1) and Q(1.2)-Q(1) from the training fitting
bank only. Use case-balanced squared error, training-only feature/target scaling,
500 fixed full-batch AdamW steps (lr .001, decay .001, clipped norm 1), and seeds
1/2/3 for every fixed architecture. The teacher's predicted advantage is exactly
zero. There is no critic bootstrap in the labels.

- Current-observation MLP: the latest valid local observation, no earlier bins.
- History MLP: flattened local history plus validity mask.
- Small transformer: one 32-wide, four-head layer with learned positions,
  64-wide feedforward, no dropout, and the same local history. All its inputs
  precede the decision, so attending across the history does not reveal future data.
- Teacher: always bias 1.
- Constant action: choose the action with highest case-balanced training-fit
  return, then fix it for every test snapshot.
- Snapshot fitting-table choice: use each test snapshot's fitting futures to
  choose an action, then evaluate it on independent scoring futures. This is a
  privileged noisy action-value diagnostic, not a deployable policy or upper bound.

Only the scoring futures evaluate selected actions. All nine learned predictors
and all controls are reported. The report also shows the same-scoring-bank
maximum explicitly as selection-biased hindsight, not usable headroom. Predictor
fit functions never read scoring-bank labels, including training scoring labels.

## Interpretation and uncertainty

Report case-balanced scoring return differences against teacher and the fixed
constant action, plus history-vs-current differences for matching model seeds.
Bootstrap 2,000 times over cases and within-action future replicas; cluster both
snapshots of a case together. Report action counts, skipped states, train-fit
error, fit/score action-ranking agreement, and byte/miss/wire differences. With
12 test cases and eight futures per bank, intervals and rankings remain noisy.
Nine fits are descriptive robustness runs, not nine opportunities to select a
winner; intervals are exploratory and have no multiplicity adjustment.

Evidence would be a history predictor improving independent-bank value over both
constant controls, consistently across model seeds, without adverse task-metric
tradeoffs. Improvement over current-only inputs would specifically support the
usefulness of history. These are conditional decision tests, not policy rollouts.
A noisy privileged-table advantage alone does not establish learnability. Poor
fit/score agreement or broad intervals mean inconclusive evidence. Even a
negative result cannot prove that local information is fundamentally insufficient
or rule out other observation windows, models, state distributions or scheduling.
No outcome triggers automatic RL promotion or deployment.

## Execution

Run a two-case engineering profile first (48 terminal continuations with two
replicas/bank), then the fixed pilot only after successful end-to-end completion.
This is a plumbing gate, not a performance gate. Local and Caliban tests must pass.
Freeze source/config before collection and verify the source manifest at startup.
Run a single service with four workers, six CPU equivalents, 8 GiB, file logs and
a 24-hour cap. Write snapshots, models and partial/completed cases directly to a
private persistent study directory; keep these raw data and weights out of Git.
Curate actual dates, coverage, uncertainty and results into the register and paper
when available. Prior studies and source directories remain untouched.

## Completed pilot

All 24 training and 12 test cases were processed. Two training snapshots were
skipped (the later snapshot of cases 700017 and 700023 had no eligible producer
before the fixed deadline); no test snapshots were skipped. The result is 46
training and 24 test snapshots with **3,360** terminal continuations, plus the
engineering check's 48. Nine fixed predictors were fit; no test selection occurred.

| Choice rule | Independent scoring return difference vs teacher | 95% exploratory interval |
| --- | --- | --- |
| Teacher | 0 | [0, 0] |
| Training-selected constant (bias 0.8) | −0.0339 | [−0.1757, +0.0964] |
| Privileged snapshot fitting-table choice | −0.0408 | [−0.1835, +0.0745] |
| Current-only models, seeds 1/2/3 | −0.0194 / −0.0309 / −0.0701 | All include zero |
| History MLP, seeds 1/2/3 | −0.1027 / −0.0793 / −0.0938 | All include zero |
| Transformer, seeds 1/2/3 | −0.0559 / −0.0541 / −0.0905 | All include zero |

These are case-balanced, conditional discounted-reward differences per snapshot,
not percentages, transfer goodput or composed-policy performance. All history and
transformer point estimates are worse than their seed-matched current-only control
and the training-selected constant; all those comparison intervals also include
zero. The nine models nearly fit their training labels (normalized training MSE
below 0.0002), but do not demonstrate useful out-of-case prediction. Finite model
capacity is therefore not an obvious explanation for training-fit failure; this
does not establish that any predictor can generalize from only 24 training cases.

The fitting-bank and scoring-bank maximizing actions agree at only 6/24 test
snapshots (25%). Selecting the maximum on the same scoring bank gives apparent
mean headroom +0.0638, but the independently scored privileged fitting-table
choice gives −0.0408 with a wide interval. Thus optimistic hindsight estimates
are not evidence of a stable exploitable signal in this study. The privileged
choice is noisy and is not a true upper bound; these observations do not prove
that useful actions never exist.

Mean interactive-miss differences for the nine predictors are all positive
(+0.46 to +2.23 misses per conditional continuation); these are counts, not
percentage points, and no confidence intervals or acceptance guards are claimed
for these task-metric differences. Verified-byte changes are mixed. A reward-only
positive result, had one occurred, would still require transport-quality checks.

The evidence does not justify more unchanged RL or asserting the overall premise
is false. It leaves small one-decision rate-control headroom unresolved, with
only eight futures per bank, two teacher-trajectory snapshots per case, a limited
observation window and finite predictors. A sharper follow-up could replicate
fixed states with many more futures before changing observations or actions;
that follow-up has not been run. Adaptive-policy execution and original reserved
transport acceptance remain unopened. Private raw datasets/models remain outside
Git; source hashes and public result summaries identify the frozen experiment.

[Post-run replay audit](../../artifacts/v7-local-predictability-replay.json) verifies
all 12 mode summaries exactly from the saved predictors and private dataset,
correct split coverage, and uniqueness of all 3,360 future replica seeds.
The audit records the raw-data hash; it does not introduce a new experiment.
