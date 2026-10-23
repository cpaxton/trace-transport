# Experiment register

Last updated: **2026-10-23**, America/New_York. This is the dated index of
experiments actually run, including negative and retired work. Training reward,
engineering acceptance, validation, and held-out performance are different kinds
of evidence. The linked artifacts contain configurations, seed splits and details.

For legacy entries, **registered** means the entry was added on this date; their
source artifacts do not preserve an execution date. It is not an inferred run
date. New experiments record their actual execution dates and status. Planned
work is listed separately and never presented as a result.

## Experiments run

| Date (kind) | Experiment | Status | Result and interpretation | Record |
| --- | --- | --- | --- | --- |
| 2026-10-23 (registered) | robust-v2 sender | Complete | Learned and fixed-rate both completed 200/200 reserved transfers. Reliability improvement over earlier learned policies; no general superiority claim. | [Results](../../artifacts/robust-v2-results.json), [guide](../training.md) |
| 2026-10-23 (registered) | stability-v1 | Complete, negative | KL stopping and intermediate snapshots exposed but did not resolve a bad training seed. Robust-v2 remained the default. | [Results](../../artifacts/stability-v1-results.json) |
| 2026-10-23 (registered) | receiver-v1 | Complete, negative | Learned 223/240 exact completions versus deterministic batching 240/240; more feedback and slower mean elapsed time than the selected reference. | [Results](../../artifacts/receiver-v1-results.json), [analysis](../receiver-experiments.md#receiver-v1-result) |
| 2026-10-23 (registered) | receipt recovery / receiver-v2 | Complete, protocol gain; no learning win | Frozen v1 weights improved 223→239/240 with receipt summaries. V2 learned candidates reached 551–552/560 generalization completions versus deterministic 553/560. | [Results](../../artifacts/receiver-v2-results.json), [analysis](../receiver-experiments.md#receipt-recovery-and-receiver-v2) |
| 2026-10-23 (registered) | receiver headroom | Complete, diagnostic only | Scenario-label oracle was 4.9% faster but used 25.6% more feedback and failed the original win criterion. Not deployable or a proven upper bound. | [Results](../../artifacts/receiver-headroom-results.json) |
| 2026-10-23 (registered) | receiver-v3 categorical control | Complete, negative learning result | Validation selected the initial snapshot. Initial, selected, and deterministic policies behaved identically and completed 240/240. | [Results](../../artifacts/receiver-v3-results.json), [analysis](protocol-v3.md#establish-receiver-control-headroom) |
| 2026-10-23 (registered) | feedback-budget-v1 | Complete, deterministic protocol gain | Budgeted timers plus sparse feedback: 240/240 diagnostic completions, 37.0% lower mean elapsed, 70.5% fewer feedback bytes, 32.4% fewer total wire bytes. Generalization improved 551→560/560. | [Results](../../artifacts/feedback-budget-v1-results.json), [analysis](protocol-v3.md#main-result-remove-a-protocol-constraint) |
| 2026-10-23 (registered) | receiver-wide-v1 | Complete, no learning win | Learned and deterministic sparse feedback completed 240/240 diagnostics; learned was slower (1.542 vs 1.491 s). Wider controls helped the protocol; learning did not beat the strong reference. | [Results](../../artifacts/receiver-wide-v1-results.json), [generalization](../../artifacts/receiver-wide-generalization-v1-results.json) |
| 2026-10-23 (registered) | sender-residual-v1 | Complete, exploratory; no clear win | 240/240 diagnostics. Learned was 5.3% faster than AIMD with a confidence interval including zero and 2.7% more total wire traffic. Stronger deterministic controls account for most generalization gains. | [Results](../../artifacts/sender-residual-v1-results.json), [generalization](../../artifacts/protocol-generalization-v1-results.json) |
| 2026-10-23 (registered) | mixed-workload-v1 | Complete, challenge set | All policies delivered all 400 short messages. Bulk completions: learned 44/50, AIMD 40/50, window-aware 43/50, fixed 50/50. Lower wire totals with incomplete bulk are not efficiency wins. | [Results](../../artifacts/mixed-workload-v1-results.json), [analysis](protocol-v3.md#a-more-demanding-application-workload) |
| 2026-10-23 (run) | joint-domain-v1 training sampler | Retired, not final evidence | Offered-load calculation undercounted data headers and reverse feedback. Partial runs were retained locally and excluded from the corrected study. | [Domain correction](../joint-network-brain.md#conditions-and-training) |
| 2026-10-23 (run) | joint-v1 CPU acceptance | Complete, engineering only | Six tiny 200-decision runs exercised training, role/history ablations, selection, export, and paired evaluation. Not a performance experiment. | [Acceptance](../../artifacts/joint-v1-acceptance.json), [scope](../../artifacts/joint-v1-acceptance-note.json) |
| 2026-10-23 (run) | joint-v1 corrected full study | Complete, negative learning result | Six runs reached ≥5 million endpoint decisions each. All three initial policies tied and outranked the six trained candidates sent to full validation. Selected initial policy: held-out misses 40.00% vs reference 40.71%; 1.74% relative reduction, below the required 20%, interval includes zero, completion-ratio guard fails. All 560 case/mode evaluations finished; OOD relative reduction was 0.76%, also below target. No wrong delivery observed. | [Full report](joint-v1.md), [raw results](../../artifacts/joint-v1-results.json), [runtime notes](../../artifacts/joint-v1-runtime-notes.json) |
| 2026-10-23 (run) | device-binary-v1 | Complete, engineering only | Herman Wi-Fi and Caliban Ethernet each recovered 9/9 synthetic text/PNG/map payloads with deterministic control and 9/9 with a trained sender-only actor: 36/36 total. Sequential, not randomized paired performance trials; receiver deterministic. | [Results](../../artifacts/device-binary-v1-results.json), [method](../joint-network-brain.md#device-engineering-checks) |
| 2026-10-23 (start/end) | goodput-v2.0 calibration + CPU pilot | Pilot complete, negative validation result | 24 deterministic controls calibrated; delay-target/512-kB/s/budgeted reference selected. Validation reference: 157,184 verified bulk B/s and 4.26% interactive misses. Pilot completed 300,011 decisions in 251 s. Seven-snapshot validation selected the initial teacher; all trained candidates fail deadline noninferiority. Repeated final-policy evaluation differed; final tests remain unopened. | [Pilot report](goodput-v2.md), [literature](../research/learning-opportunities.md); [validation artifacts](../../artifacts/goodput-v2.0-pilot-validation.json) |

## Planned experiments

| Date planned | Experiment | Objective and acceptance | Status |
| --- | --- | --- | --- |
| 2026-10-23 | goodput-v2 | Maximize verified useful delivery with continuously backlogged bulk traffic; a separate mixed mode preserves the 100-ms target. Proposed target: ≥10% goodput gain with positive paired confidence interval, deadline noninferiority, bounded wire overhead/queues, and zero wrong delivery. | [Plan](../goodput-v2-plan.md), [versioned specification](../../experiments/joint-goodput-v2-plan.json). Prototype implemented; first bounded pilot completed without a validation win under goodput-v2.0 above. Full acceptance evaluation is pending. |

## Recording the next run

Add an entry when an experiment starts, and update that same entry when it ends.
Record start/end dates, experiment/configuration version, hypothesis, baseline,
training budget, selection and held-out splits, result, limitations, and links to
raw/compact artifacts and checkpoint hashes. Keep failures, interruptions,
retirements, and negative outcomes visible. Changing an objective or acceptance
criterion starts a new version; it does not rename an old result into a win.

`trace-transport joint-status` and the console's `/joint` page show live local
progress; this register records reviewable milestones and conclusions. Raw runs
and device credentials stay outside Git. Publish updates through feature PRs.
