# Is local history predictive of useful rate decisions?

Registered 2026-10-31. Status: **planned; no results**. Follow-up to the negative
[longer-credit pilot](v7-long-credit.md). This isolates local action predictability
before further RL. A negative outcome concerns these observations, actions,
prefixes, continuation and finite data, not all possible learned protocols.

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
