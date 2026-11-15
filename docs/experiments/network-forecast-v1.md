# Network forecasting before learned scheduling

Registered 2026-11-14. Status: **complete; registered primary forecast criterion
failed**. Engineering ran 15:25:34–15:26:47 UTC and pilot ran
**2026-11-14 15:26:49–15:42:07 UTC**. All cases/fixed fits completed successfully.
[Engineering status](../../artifacts/network-forecast-engineering-status.json),
[actual pilot plan](../../artifacts/network-forecast-pilot-plan.json),
[seed-allocation audit](../../artifacts/network-forecast-seed-audit.json),
[full results](../../artifacts/network-forecast-pilot-results.json).
All eight targeted tests passed on Caliban. No RL or scheduler deployment occurred.
The fixed design follows; results and limitations appear below.

## Task and labels

Use the existing v6 network/workload and deterministic v7 teacher throughout.
Collect one complete 12-second-production / 3-second-drain trajectory per case,
without branching or modifying transport physics. The observer keeps a new
**two-second, 64-bin** causal history. At 0.5-second intervals from seconds 2 to
10, sample every even-index producer that currently has outstanding bulk data.
Selection uses current local backlog, not future service or scenario outcomes.

Predict three horizons: 0.5, 1 and 2 seconds. Targets are:

1. Newly acknowledged payload bytes, encoded as log1p(ACK bytes / horizon /
   128000). Preserve exact integer byte counts for evaluation. This is service
   under the fixed teacher and offered workload, not hidden physical capacity.
2. Sender SRTT at the end of the window, encoded as log1p(SRTT / 0.1). Score and
   train only when at least one fresh RTT sample arrived in that window. Missing
   RTT labels remain masked; a retained estimate during silence is not a new
   measurement. The target is the sender estimator, not instantaneous path RTT.
3. Whether **zero payload bytes were newly acknowledged** during the window.
   This is ACK-progress silence, not a physical-outage label. Queueing, load,
   churn, feedback loss and outages can all influence it.

Require horizon alignment within 1 microsecond and save actual elapsed times.
Predictions use the declared horizon, never future elapsed time. Record future
backlog fractions for interpretation, and report an additional >=90%-backlogged
subset descriptively. Do not exclude samples based on future backlog or use it
as an input. Later scheduling may change offered load and invalidate these
conditional forecasts; this study does not validate that distribution shift.

Inputs are only producer-local history and validity mask. Exclude case IDs,
scenario/family labels, absolute anchor times, endpoint IDs, future labels and
hidden network schedules. Omit the raw `age` column; in the current v6 adapter
that column encodes the control tick rather than episode age. Other local ages
(e.g. feedback age) remain causal inputs. This is a new forecast schema, not a
retagged protocol actor checkpoint.

## Prospective split and fixed fits

[Config](../../experiments/network-forecast-v1.json): allocate new, disjoint
synthetic case banks, balanced across all 12 existing families: 96 train (8 per
family), 24 calibration (2), 48 test (4). Check IDs against prior public JSON
integer values. A separate six-case engineering profile shares no case IDs.
These banks are prospective for this experiment, not reserved transport
acceptance. They remain uncalibrated synthetic conditions, not real Wi-Fi traces.
All anchors/endpoints from a network case stay in one split. Overlapping windows
never count as independent network cases.

Fit four baselines from training data only: a constant distribution, latest-bin
persistence, two-second moving-average ACK rate, and an ACK-rate EWMA with
250-ms half-life. Dynamic baselines persist latest SRTT and predict silence when
their ACK-rate point forecast is zero. Their interval residual quantiles (.1/.5/.9)
are fitted with case-balanced training weights. The constant silence probability
is the training frequency. Baselines, including their limitations, remain visible.

Fit current-observation MLP, flattened-history MLP and a small transformer (one
32-wide, four-head layer). Each emits nonnegative sorted .1/.5/.9 quantiles for
ACK-rate and SRTT log targets plus a silence logit at all horizons. Train raw
quantiles and clamp them at zero for inference, so zero-byte windows can be
covered by forecast intervals. Use seeds
1/2/3, 1,000 AdamW steps, lr .001, decay .001, batch 128 with replacement and
clipped gradient norm 1. Batch RNG is separate from initialization and matched
across architectures within a seed. Training-only feature scaling and case/mask
weights prevent dense endpoints from dominating. Minimize equally weighted
pinball losses and binary cross-entropy; no checkpoint selection or tuning.

Report all nine fits plus three fixed architecture ensembles (average quantiles
and probabilities). Calibration chooses the strongest of the four baselines by
one-second ACK-byte MAE, without changing or selecting a learned model. No other
calibration fitting, interval widening or post-hoc model choice occurs.

## Evaluation and predeclared evidence

Primary: transformer ensemble versus calibration-selected baseline on test
one-second ACK-byte MAE. Evidence of a forecasting opportunity requires >=10%
MAE reduction, upper paired difference CI below zero, and empirical coverage
between 70% and 90% for the nominal 80% ACK interval. These are development
criteria, not an accepted transport win or an automatic scheduler promotion.
Compare the same ensemble against current-only to test the added value of history.

Report all horizons, exact byte MAE, log pinball score, interval coverage/width,
fresh-label SRTT MAE/coverage, silence Brier score, prevalence and descriptive
reliability bins. Average endpoints/windows within cases; use 2,000 paired
bootstrap draws over cases within synthetic families. Keep repeated windows and
model seeds from becoming extra independent cases. Report sample/case coverage
and undefined RTT populations; silence calibration does not establish outage
causality. All secondary model/horizon intervals are exploratory without
multiplicity correction. A positive forecast result still needs closed-loop
validation with realistic scheduling objectives.

## Execution and next stage

Run the six-case / 10-step engineering profile first, then the registered pilot
only after successful end-to-end completion. This is a plumbing gate without
performance-based promotion. Freeze source/config and verify their manifest.
Write compressed case arrays, predictions and model weights to private persistent
storage. Run a single Caliban service: four collection workers, six CPU
equivalents, 8 GiB, one numerical thread, file logs, 24-hour cap. Preserve prior
studies. Record actual dates and results in the register and paper when available.

CLI (learning dependencies required): `uv run --locked --extra learn python -m scripts.network_forecast --config CONFIG --output NEW_DIRECTORY
--profile engineering|pilot` from a frozen source-bundle directory.

The next planned research stage is object scheduling for commands, image objects
and map deltas using forecast service budgets. Its baselines should include
priority, deadline and fair-queueing rules, with exact-delivery checks and
application latency/freshness goals. That scheduler has not been implemented or
trained by this forecast study; forecast quality must be assessed first.

## Completed prospective test

All 168 pilot cases completed: 96 training (4,306 samples), 24 calibration (980),
48 test (2,297). Samples are overlapping endpoint/time snapshots, not independent
cases. Nine models each fit 1,000 fixed steps; all nine plus three ensembles and
four baselines are reported. The six engineering cases are disjoint. All raw case
arrays, private weights, prediction arrays and fitted baseline parameters remain
in private persistent storage, with hashes in the public result.

Calibration selected **EWMA** for the primary one-second ACK-byte comparison.

| Predictor | Test 1s ACK-byte MAE | Nominal 80% interval coverage | Test 1s fresh-label SRTT MAE |
| --- | --- | --- | --- |
| EWMA | 39,071 B | 84.5% | 24.0 ms |
| Current-only ensemble | 51,395 B | 74.7% | 18.5 ms |
| History MLP ensemble | 82,448 B | 45.6% | 25.6 ms |
| Transformer ensemble (primary) | 47,241 B | 78.7% | 19.0 ms |

Transformer ACK-byte error is **20.9% higher**, rather than the required >=10%
reduction. Its paired difference is +8,170 B, family-stratified case-bootstrap
95% interval [−927, +20,816] B; the interval includes zero. Interval coverage
passes the registered range, but gain and paired-difference criteria fail.
`forecast_evidence=false`; there is no accepted transport win. The transformer ensemble versus
current-only ACK-error difference is −4,154 B [−13,192, +9,915] B, also uncertain.
No learned ensemble beats EWMA's ACK-byte point MAE at any of the three horizons;
at 2 s the moving-average baseline has the best baseline point estimate. The
constant baseline's much larger ACK error is not an appropriate sole comparator.

There is a secondary latency lead: the current-only ensemble reduces one-second
SRTT error from 23.99 to 18.52 ms (22.8%). A **post-hoc**, unadjusted paired
interval for its MAE difference is [−11.01, −0.86] ms. This exploratory comparison
was calculated after seeing test results; it does not replace the registered
primary. All three ensemble SRTT comparisons are included in the
[secondary artifact](../../artifacts/network-forecast-secondary.json). The
transformer's SRTT improvement has an interval crossing zero. Fresh cases and a
newly registered latency-specific criterion are needed before relying on the
selected current-only lead. These errors concern future sender SRTT conditional
on fresh RTT samples, not physical delay during periods without samples.

ACK-silence prevalence is only 1.37% on one-second test windows (case-weighted).
The constant-frequency baseline already has Brier score 0.01354, versus 0.01289
for current-only and 0.01311 for transformer. Large improvements over a weak
binary persistence/silence heuristic do not establish a strong classifier. All
calibration diagrams and other horizons are descriptive secondary results.

The test establishes neither superior ACK-service forecasting nor a need for a
transformer. It leaves an exploratory small current-state latency predictor worth
replicating. Do not promote a scheduler or report an application win from these
numbers. A later scheduling experiment could keep the EWMA service budget and
separately test a freshly validated learned-delay component. That experiment has
not run; original reserved transport acceptance remains unopened.

[Saved-artifact replay audit](../../artifacts/network-forecast-replay.json)
verifies every case hash, disjoint splits, all nine model files, all 32 saved
prediction arrays, exact metric reproduction and calibration-only baseline
selection. No retraining occurred during verification.
