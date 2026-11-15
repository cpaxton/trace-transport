# Confirm the exploratory SRTT forecast lead

Registered 2026-11-15. Status: **planned; no results**. Follow-up to
[network-forecast-v1](network-forecast-v1.md), whose primary ACK-byte comparison
failed. The current-only ensemble's 22.8% SRTT error reduction was selected after
looking at the old test results; a fresh test is required to confirm it.

## Fixed prospective experiment

[Config](../../experiments/rtt-replication-v1.json) allocates **96 new test cases**,
eight per existing synthetic family, plus two separate engineering cases. IDs
are checked against prior public config/artifact integer values. These are
synthetic fresh-case results, not new robot traces or transport acceptance.

Freeze all nine original neural forecast files, their normalization and original
baseline file by SHA-256. The primary model is **current-ensemble**, the average
of original current-only seeds 1/2/3. There is no neural retraining, checkpoint
selection or tuning on new cases. Keep the teacher, two-second local histories,
anchors at 2–10 seconds, horizons .5/1/2 seconds, backlog gating, RTT-label masking
and all transport semantics exactly as in the original experiment. Verify the
collector/predictor source and all core simulator hashes against the old plan.

Fit stronger RTT baselines on the **original training data only**:

- Raw latest-SRTT persistence. Preserve the raw median; residuals only provide
  intervals around it.
- Original persistence with its original training-fitted residual quantiles.
- RTT EWMA over valid positive historical estimates, half-life 250 ms.
- Bounded trend: fit a least-squares slope to valid positive RTT bins in the
  two-second history, extrapolate from latest SRTT, and clip the change to
  +/- max(latest SRTT, 10 ms), then clip predictions at zero.
- Ridge regression on standardized current local features, targeting future log
  SRTT. Case/mask weights, fixed ridge penalty .1, unpenalized intercept; training
  data alone determines normalization and residual quantiles.

All newly fitted baselines get case-balanced .1/.5/.9 residual quantiles from the
old training data, except the preserved original persistence residuals and raw
persistence's fixed median. This is supervised baseline fitting, not retraining
the neural candidates. Neither old test arrays nor new case labels are supplied
to baseline fitting or selection.

Use **original calibration cases** to choose the strongest of the five baselines
by one-second conditional SRTT MAE, before collecting any new test cases. Report
all baselines, nine frozen models and three fixed ensembles (17 modes). The
original EWMA ACK-budget forecast remains attached to deterministic RTT baselines
for contextual summaries; ACK/gap results are secondary and do not redefine the
failed forecast-v1 primary.

## Predeclared outcome and limits

Primary: frozen current-only ensemble versus the selected baseline on **new-case
one-second fresh-label SRTT MAE**. Confirmation requires >=10% relative MAE
reduction, upper paired difference CI below zero, and lower point MAE than raw
persistence. Use 2,000 family-stratified case bootstrap draws (seed 718); keep
all endpoints and overlapping windows within their case. Include only cases
with fresh labels in that horizon's comparison and report their count. Do not
impute RTT during unobserved windows.

Report all other models/horizons descriptively, plus interval coverage/width and
case-balanced 95th-percentile absolute SRTT error. No multiplicity adjustment is
claimed for secondary results. If ridge wins, distinguish useful linear
predictability from evidence for a neural advantage. A negative primary stays
negative regardless of another favorable mode. No result becomes a throughput,
reliability, physical-outage, one-way delay or application-deadline win.

The target remains **future sender SRTT conditional on a fresh RTT sample**.
Missing-label selection and the fixed teacher matter: prediction improvement may
not transfer to queue scheduling or real Wi-Fi. A positive result could justify a
separate closed-loop scheduler experiment using EWMA service budgets and a frozen
latency component. That scheduler has not run, and no automatic policy promotion
is performed.

## Execution

Freeze source/config before collection. Run engineering then replication in one
Caliban user service: four collection workers, six CPU equivalents, 8 GiB,
single-threaded numerical libraries, file logs, 12-hour maximum. Engineering
checks plumbing only and does not gate on performance. Keep original studies,
raw arrays, model files and baseline parameters private and unchanged. New
outputs are written directly to a fresh persistent directory, with case hashes,
prediction hashes, actual dates and failure records. Update the register and
paper with actual results, including negative/incomplete outcomes.

CLI from a frozen source bundle: `python -m scripts.rtt_replication --config CONFIG
--reference-root ORIGINAL_FORECAST_PILOT --output NEW_DIRECTORY --profile
engineering|replication`. The original reference includes saved weights, case
arrays, results and baseline parameters. Dependencies are the existing learning
extra. Original reserved transport acceptance sets remain unopened.
