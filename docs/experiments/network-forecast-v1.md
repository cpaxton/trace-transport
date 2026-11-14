# Network forecasting before learned scheduling

Registered 2026-11-14. Status: **pilot running**. Engineering completed 2026-11-14T15:25:34.071275+00:00 to
2026-11-14T15:26:47.062101+00:00; pilot actually started 2026-11-14T15:26:49.773163+00:00.
[Engineering status](../../artifacts/network-forecast-engineering-status.json),
[actual pilot plan](../../artifacts/network-forecast-pilot-plan.json),
[seed-allocation audit](../../artifacts/network-forecast-seed-audit.json).
All eight targeted tests passed on Caliban. Engineering is a plumbing check;
no pilot forecast-quality result is available yet. The completed
[precision study (PR #9)](https://github.com/cpaxton/trace-transport/pull/9)
did not establish useful independent-bank action values. This experiment asks
whether local history predicts observable future service directly. Forecast
accuracy and application utility are separate evidence; no RL is trained here.

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
