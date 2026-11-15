# Corrected frozen SRTT replication (v2)

Registered 2026-11-15. Status: **complete; corrected primary failed**. V1 is
retired for invalid baseline units. Engineering ran 16:39:36–16:39:43 UTC;
replication ran **2026-11-15 16:39:48–16:47:42 UTC**. All 96 fresh cases and
17 modes completed; 4,374 endpoint/time samples are clustered by case.
[Correction](../../artifacts/rtt-baseline-units-correction.json),
[actual v2 plan](../../artifacts/rtt-replication-v2-plan.json),
[results](../../artifacts/rtt-replication-v2-results.json),
[exact replay audit](../../artifacts/rtt-replication-v2-replay.json).
The decoder matches the real observer at 2/100/500 ms. Original neural inputs,
weights, labels and the collector/model code are unchanged, checked against the
archived source. All 14 targeted corrected tests pass on Caliban. No scheduler,
receiver-policy change, neural retraining or transport gain is claimed.
The preregistered v2 design follows.

## Fixed prospective experiment

[Config](../../experiments/rtt-replication-v2.json) allocates **96 new test cases**,
eight per existing synthetic family, plus two separate engineering cases, all disjoint from v1 and prior studies. IDs
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
- Correctly decoded persistence with residual quantiles refit on original training data.
- RTT EWMA over valid positive historical estimates, half-life 250 ms.
- Bounded trend: fit a least-squares slope to valid positive RTT bins in the
  two-second history, extrapolate from latest SRTT, and clip the change to
  +/- max(latest SRTT, 10 ms), then clip predictions at zero.
- Ridge regression on standardized current local features, targeting future log
  SRTT. Case/mask weights, fixed ridge penalty .1, unpenalized intercept; training
  data alone determines normalization and residual quantiles.

All newly fitted baselines get case-balanced .1/.5/.9 residual quantiles from the
old training data, except raw
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

## Corrected fresh-case result

| Mode | One-second SRTT MAE | Nominal 80% coverage | Case-balanced p95 absolute error |
| --- | --- | --- | --- |
| Raw persistence | 14.69 ms | 79.1% | 77.46 ms |
| Adjusted persistence (selected comparator) | 14.69 ms | 79.1% | 77.46 ms |
| RTT EWMA | 15.32 ms | 80.5% | 86.91 ms |
| Bounded trend | 19.57 ms | 80.3% | 107.84 ms |
| Ridge current-state | 20.34 ms | 77.3% | 86.57 ms |
| Current-only ensemble (primary) | 17.95 ms | 78.6% | 79.41 ms |
| History ensemble | 22.99 ms | 65.9% | 104.93 ms |
| Transformer ensemble | 18.06 ms | 82.0% | 79.48 ms |

The primary current-only ensemble has **22.2% higher** MAE than the selected
comparator. Its paired MAE difference is +3.26 ms, 95% family-stratified case
bootstrap interval **[−0.01, +7.89] ms**, which includes zero. It fails both
required gain/CI conditions and has higher point error than raw persistence.
`rtt_evidence=false`; no learned RTT benefit is confirmed. Do not interpret the
higher point error alone as statistically established harm.

Every architecture ensemble has higher point MAE than corrected persistence.
Interval coverage is similar for current/transformer and the comparator, while
the history MLP undercovers. Tail metrics are descriptive and secondary. The
study concerns fresh-sample sender SRTT in synthetic teacher trajectories, not
physical latency during missing observations or application deadlines.

The corrected historical rescoring also removes the original lead: on the old
48 test cases, current-only stays at 18.52 ms while properly decoded persistence
is 14.77 ms (25.4% higher neural point error). These are separate data sets and
analyses; they are not pooled into additional confirmation cases. The original
23% improvement claim remains withdrawn. Original ACK-byte primary results stay
negative and unaffected by the RTT decoder correction.

The replay audit rechecks all frozen weights, case hashes, original-train-only
baseline fits, original-calibration selection and all 17 prediction/report modes
in the same one-thread numerical environment as the run. No raw run was changed.

This closes the RTT lead without a confirmed forecasting win. More unchanged
training is not justified by these results. Future application scheduling can
retain strong simple service/latency estimates and measure object-level utility;
a fresh design must establish scheduling headroom before training. No such
scheduler or original reserved transport acceptance test has run here.
