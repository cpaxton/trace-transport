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

## Follow-up experiments

| Actual start | End | Experiment | Configuration and evidence | Status |
| --- | --- | --- | --- | --- |
| 2026-10-23 UTC | 2026-10-24 UTC | goodput-v2.1 / learn42 | [Launcher/configuration](../../scripts/goodput_learn.py), [domain](../../src/trace_transport/goodput.py), [reference search](../../scripts/goodput_study.py). One million endpoint decisions, joint sender/receiver, paired teacher reward, CPU only. Calibration searched 87 named deterministic settings; selected delay-target pacing at 512,000 B/s with ACK batch 16 / 25 ms. Sixteen-case validation reference: 158,379 verified bulk B/s, 4.29% interactive misses. | Complete, negative validation result. 1,000,006 decisions; initial teacher selected. Final learned goodput −3.56%, deadline guard fails. [Validation](../../artifacts/goodput-v2.1-validation.json), [configuration](../../artifacts/goodput-v2.1-training-config.json). Reserved test and OOD seeds unopened. |

The v2.1 inference diagnostic reproduced all 16 case records exactly with one
versus two evaluation workers after fixing inference batch shapes. This is an
engineering repeatability check, not evidence of improved network performance.
The v2.0 negative result remains recorded separately above.

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

## Changing-network role comparison

Actual start/end 2026-10-24 UTC. Complete, negative exploratory comparison:
adaptive deterministic, learned sender, learned receiver, and joint learning;
128 paired case/controller evaluations on two new 16-case banks. Coupled-harsh:
joint goodput −4.15%, interactive misses 42.24% versus 39.30%; zero-bulk
host-cases 4 versus 1. Receiver-only matched adaptive; the action audit found
teacher-following receiver proposals throughout four sampled cases. No wrong
delivery observed. [Report and limitations](changing-comparison-v1.md),
[configuration](../../artifacts/changing-comparison-v1-plan.json),
[results](../../artifacts/changing-comparison-v1-results.json).
Reserved acceptance seeds remain unopened.

## Distribution-v3 audit and retraining

Actual start/end: 2026-10-24 UTC. Completed a synthetic distribution and
legal-control sensitivity audit before retraining. Twelve explicit strata cover
stable and wired controls, shared-airtime Wi-Fi, fades/interference, short service
opportunities, propagation-only changes, reverse bottlenecks, loss-cause changes,
outage recovery, source churn and large queues. [Coverage contract and gaps](distribution-v3.md),
[sampler](../../src/trace_transport/goodput_distribution.py),
[audit](../../scripts/goodput_distribution_audit.py). One initial engineering
attempt stopped because the metrics helper rejected v3 records; that schema
support was corrected and the attempt preserved locally. No scientific conclusion
was drawn from the failed attempt. Audit: all twelve strata represented in 1,000 configurations, with 216 legal-control evaluations. The deadline-constrained calibration selects the wider-bandwidth adaptive reference. [Coverage](../../artifacts/distribution-v3-coverage.json), [results](../../artifacts/distribution-v3-audit-results.json).

V3 joint pilot actual start/end: 2026-10-24 UTC. Complete, no established learning
win. Both actors trained for 300,008 endpoint decisions in 403 s; 120 validation
case/controller evaluations completed. Joint goodput +1.85% with interval
including zero; deadline and zero-completed-bulk-host guards fail. Receiver-only
records match adaptive despite 51.6% nonteacher training samples. A four-case
audit again finds all greedy receiver proposals teacher-following.
[Configuration](../../artifacts/goodput-v3-pilot-plan.json). [Results and limitations](distribution-v3.md#completed-v3-pilot-2026-10-24-utc),
[validation](../../artifacts/goodput-v3-pilot-validation.json). These are one-seed
pilot validation findings; reserved acceptance seeds remain unopened.

Device-link-snapshot-v1 actual start/end 2026-10-24 UTC: complete, engineering
inventory only. Two ten-packet idle Herman probes and one successful ten-packet
Caliban probe; link/PHY rates are not capacity measurements. First Caliban ping
failed because ping did not resolve its SSH alias; the corrected capture used the
SSH-resolved hostname. No saturation or device changes. The sub-millisecond wired
RTT and reported gigabit interface expose coverage gaps for a future long study.
[Results](../../artifacts/device-link-snapshot-v1-results.json),
[limitations](distribution-v3.md#small-device-inventory-check).

## Caliban v4 parallel study and review corrections

| Actual start | Actual end | Experiment | Status and evidence |
| --- | --- | --- | --- |
| 2026-10-24T17:53:26.524886+00:00 | 2026-10-24T17:54:54.916570+00:00 | Caliban v4 worker/model benchmark | Complete, engineering only. 1/2/4 workers: 68.1/72.2/77.2 opportunities/s; actor-pair CPU p50/p95 18.8/20.1 ms. [Results](../../artifacts/caliban-v4-benchmark.json). |
| 2026-10-24 UTC | 2026-10-24 UTC | Matched workstation v4 benchmark | Complete, engineering only. Best short throughput 164.9/s; CPU inference p50/p95 7.9/8.2 ms. [Results](../../artifacts/v4-local-benchmark.json). |
| 2026-10-24T17:50:39.893480+00:00 | 2026-10-24T17:51:50.199796+00:00 | Cloned-state receiver action diagnostic | Complete, exploratory. Eight states attempted, seven found pending ACKs; all 22 actions compared over one-second continuations. Different choices maximize return in different cases; selected maxima on one future RNG realization do not establish expected or deployable gains. [Results](../../artifacts/receiver-counterfactual-v4-results.json). |
| 2026-10-24T17:58:42.427973+00:00 | Pending | Caliban v4 multi-seed study | Running calibration/training pipeline. Seeds 42/43/44, planned five million opportunities each, 64-step history, width 64, two layers; four workers selected by benchmark. Six CPU cores, 8-GiB memory ceiling, 72-hour limit, bounded persistent checkpoint/source mirror. [Plan](../../artifacts/caliban-v4-plan.json), [frozen source](../../artifacts/caliban-v4-source.zip), [operation and limitations](caliban-v4.md). No trained result yet; reserved test/OOD ranges unopened. |

Review correction applies to the completed v2.1/v3 training signals: cloned teacher
RNGs can diverge under different packet schedules. The subtraction was a same-case
return baseline, not demonstrated CRN coupling or guaranteed variance reduction.
Original metrics are unchanged. V4 longer training disables subtraction and uses
raw task rewards with the learned PPO critic. The review also prompted explicit
CLI profile selection, named inference bounds, and named observation indices.

Elapsed-time GAE engineering check, actual start/end 2026-10-24 UTC: implemented
and unit/integration verified; no performance experiment or learned win claimed.
Optional `mappo-time-gae-v1` scales discount/trace decay by actual transition time;
Caliban's frozen study remains on `mappo-step-gae-v1`. Tests cover 2/20-ms prefix
trace equivalence, terminal masks and tiny v4 training. [Method and planned
comparison](caliban-v4.md#local-follow-up-elapsed-time-credit-assignment),
[tests](../../tests/test_time_gae.py). Larger ablation remains planned.

Caliban v4 update, 2026-10-25 UTC: seed 42 completed 5,000,031 opportunities in 58,974 s (16.38 h), then entered snapshot validation. Ten snapshot summaries are available; final-checkpoint validation/selection and seeds 43/44 remain pending. No available trained snapshot passes the deadline guard. The largest goodput point estimate is +2.82%, with difference CI including zero; this is selection data, not held-out evidence. Twelve optimizer checkpoints are persisted; root disk free ~1.37 GiB. [Partial results](../../artifacts/caliban-v4-seed42-validation-partial.json).

Caliban v4 seed 42 validation completed, 2026-10-25 UTC: 11 checkpoint summaries.
Selection is `d002509210.pt`, a +2.82% goodput point estimate whose interval
includes zero and whose deadline guard fails. Final policy: −2.71% goodput and
22.52% interactive misses versus reference 21.94%. No sustained improvement or
accepted win. [Completed curve](../../artifacts/caliban-v4-seed42-results.json).

User-authorized checkpoint maintenance, actual start/end 2026-10-25 UTC: removed
27 redundant completed-seed/benchmark checkpoint files (72.65 MiB total, including
26.19 MiB on persistent system disk). Retained seed42 initial, selected-best and
final weights plus every validation summary; discarded weights cannot support new
post-hoc ablations. Kept all active seed43 optimizer/RNG state. Stopped the owned
service to avoid mirror races, then restarted the identical frozen source with
explicit resume. Collector worlds/histories reset; optimizer/RNG restoration is
not bitwise continuation of in-flight trajectories. Reference checks precede
continued seed43 training. [Removal hashes and retention](../../artifacts/caliban-v4-checkpoint-cleanup.json).

## V5 measurement and learning-signal follow-up

| Actual start | Actual end | Experiment | Status and evidence |
| --- | --- | --- | --- |
| 2026-10-25 UTC | 2026-10-25T16:50:21.769665+00:00 | Seed42 per-family/service-bound audit | Completed, post-hoc diagnosis. Wired cases supply 55.9% of initial bulk bytes; one Wi-Fi case provably cannot complete one smallest object per host in the measurement window even under an optimistic header-aware bound. Old scores/criteria are unchanged. [Results](../../artifacts/learning-diagnostics-v1-results.json). |
| 2026-10-25 UTC | 2026-10-25T16:57:23.869609+00:00 | V5 baseline progress replay | Completed, metric diagnosis only, two reused Wi-Fi validation cases. Zero completed bulk bytes but nonzero CRC-accepted bulk payload at all five hosts; accepted rate 18,795 B/s is not completed-object goodput. [Results](../../artifacts/progress-baseline-v5-results.json). |
| 2026-10-25T17:09:06.245621+00:00 | 2026-10-25T17:27:16.473754+00:00 | V5 raw versus potential-shaped pilot | Completed, exploratory; no accepted win. Raw 200,001 and shaped 200,002 opportunities. Shaped +0.66% verified goodput vs adaptive and passes v5 quality guards, below 10% target. Shaped vs raw +1.45%, but fails deadline/per-family progress nonregression. Initial parameters identical. 72 validation case/controller evaluations; reserved tests unopened. [Plan](../../artifacts/progress-ablation-v5-plan.json), [frozen source](../../artifacts/progress-ablation-v5-source.zip), [method/limitations](learning-signal-v5.md). |

V5 changes the service-starvation guard explicitly and adds fairness/per-family
checks while retaining whole-object goodput and integrity requirements. Prefix
potential is training-only and cancels at termination. This does not reinterpret
v4's failed acceptance or prove any new learning gain.

V6 reactive receiver engineering check, actual start/end 2026-10-25 UTC:
implemented and tested, not a training-performance study. Changed ACK controls now
act on pending receipts without requiring a later packet arrival; old timers are
invalidated, pending age retained, and flush bytes charged to raw reward. Tests
also verify unchanged-setting outcome parity and the serialized protocol contract.
This is a new simulator profile; the completed v5 pilot and running Caliban v4
study retain their frozen semantics. [Plan](../../experiments/goodput-v6-plan.json),
[tests](../../tests/test_reactive_receiver.py). The v4 study remains a diagnostic
baseline; its old all-host completed-object guard is now known to be infeasible
in at least one validation case, so it cannot establish an accepted win under
that frozen criterion.

V5 final role/proposal audits, actual start/end 2026-10-25 UTC: sender-only matches
joint shaped records exactly across all 24 validation cases; receiver-only matches
the adaptive reference exactly. Four-case audit finds 30,166 valid-history receiver
proposals all teacher-following (including inactive rows). The small shaped-policy
gain is not evidence of useful receiver learning. [Roles](../../artifacts/progress-ablation-v5-roles.json),
[proposals](../../artifacts/progress-ablation-v5-final-action-audit.json).

Reactive v6 matched retraining pilot — **running**, actual start 2026-10-25 UTC;
end pending. [Frozen plan](../../artifacts/reactive-v6-pilot-plan.json),
[runner](../../scripts/reactive_pilot.py). One seed (42), 200,000 decisions,
history 64 / width 32 / one transformer layer, two environments; potential shaping,
time-aware GAE and receiver action alias masking match the completed v5 shaped arm.
Initial sender/receiver/critic tensors and the 24 validation seed memberships were
checked identical. Only pending-ACK protocol semantics change. Final-checkpoint
validation compares against its own v6 adaptive baseline; no reserved tests or
acceptance claim. CPU-only isolated source/runtime, six-core affinity, 8 GiB limit,
file logs; the frozen Caliban v4 study remains unchanged. No performance result yet.
