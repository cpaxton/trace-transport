# Pushing beyond receiver-v2

This batch separates three questions: whether the receiver training was weak,
whether its controls offered much headroom, and whether the protocol itself
prevented useful feedback timing. All measurements below are CPU simulations,
with zero observed wrong payloads. Neural inference cost is excluded from
simulated time. Wi-Fi service models remain uncalibrated.

## Main result: remove a protocol constraint

The old receiver could wait at most 25 ms, and the sender did not explicitly add
an intentional feedback-wait budget to its retransmission timeout. Slow reverse
paths accumulated ACK traffic and avoidable retransmissions. The new experiment
configures a 250-ms maximum feedback-wait budget at both endpoints, adds it to
sender RTO, and explores longer waits and larger batches. This budget is
configured out of band in the simulator; on-wire negotiation and deployment on
real devices are not implemented.

The existing unbudgeted mode remains unchanged. The delay field in receipts is
bounded by the configured limit and removed from RTT measurements. The first
completion receipt remains immediate, so small complete messages do not wait
for a partial batch timer. All ACK and data bytes consume link capacity.

Validation seeds 17000–17004 choose a constant setting from nine controls.
Diagnostic seeds 18000–18019 and generalization seeds 19000–19019 are separate.
The selected control is batch 8, maximum wait 250 ms.

| Diagnostic protocol | Exact completions | Mean elapsed | Feedback bytes | Total wire bytes |
| --- | ---: | ---: | ---: | ---: |
| Old batch 8 / 25 ms, old timers | 240/240 | 2.367 s | 1,388,086 | 15,608,314 |
| Same ACK policy, budgeted timers | 240/240 | 2.172 s | 1,146,262 | 12,328,675 |
| Batch 8 / 250 ms, budgeted timers | 240/240 | **1.491 s** | **409,714** | **10,558,204** |

The full protocol change reduces diagnostic mean elapsed time by **37.0%**,
feedback by **70.5%**, and total wire bytes by **32.4%**. The paired seed-block
95% interval for elapsed-time difference is [-1.067, -0.675] seconds. Both
policies complete every diagnostic transfer, so this time comparison is not
confounded by timeouts.

On the separate 560-transfer generalization suite, the old protocol completes
551, and both budgeted variants complete 560. Mean elapsed times are 4.332 s,
3.107 s, and 4.091 s respectively. These means include deadlines for failures;
they are not success-only latency statistics. Longer waits save feedback but
can slow fast paths: the timer-only arm is fastest on this suite. The sparse
arm uses 2,140,786 feedback bytes versus 5,491,166 for the old protocol, and
49,114,999 total wire bytes versus 59,422,901.

These are **deterministic protocol gains, not evidence of learned superiority**.
Raw records and all three ablation arms are in
[feedback-budget-v1-results.json](../../artifacts/feedback-budget-v1-results.json).

## Establish receiver control headroom

A validation sweep over 13 constant settings still selects batch 8 / 25 ms
within the old control range. A scenario-label oracle selecting a separate
constant setting for each case is 4.9% faster on fresh test seeds, but uses
25.6% more feedback and fails the original win condition. It is not a deployable
policy and is not an upper bound on a fully time-varying policy.
See [headroom results](../../artifacts/receiver-headroom-results.json).

The previous tanh-Gaussian policy started at a mediocre batch/wait combination
and its continuous action map made exact endpoint delays awkward to reach. The new categorical policy can
represent every constant setting exactly. It starts with a 90% prior on the
strong baseline, trains for 320 episodes per seed (42/43/44), and validates
snapshots every 40 episodes, including the initial snapshot. Validation seeds
8000–8004 select the **initial snapshot**. On seeds 9000–9019, learned,
initial, and deterministic policies have identical behavior and complete
240/240 transfers. Training did not improve this policy. The initial checkpoint
is retained for reproducibility, not recommended in place of its inexpensive
deterministic equivalent. See [receiver-v3 results](../../artifacts/receiver-v3-results.json).

## Learn over the wider feedback range

A second categorical experiment adds 100/250-ms waits and 16/32-entry batch
targets. Both endpoints use the same 250-ms delay budget. Training uses persistent
randomized degradation, three seeds, and 320 episodes each. The initial prior is
8% on batch 8 / 250 ms with the remainder shared by the other 20 controls, so
exploration can reach alternatives. Snapshots and all 21 constant baselines
compete on validation seeds 21000–21004; testing uses 22000–22019.

Validation selects seed 42 at episode 200; the best constant is batch 16 / 250 ms.
Both complete 240/240 diagnostic transfers. The learned policy averages 1.542 s
and 364,222 ACK bytes, compared with 1.530 s and 253,867 bytes for the best
constant. It does not win. On fresh generalization seeds 23000–23019, both
complete 559/560; the latency-focused batch-8/25-ms budgeted control completes
560/560 and is faster. The wider range gives the learner a real tradeoff, but
this training run does not exploit it better than tuned constants.
See [wide-policy results](../../artifacts/receiver-wide-v1-results.json) and
[generalization](../../artifacts/receiver-wide-generalization-v1-results.json).

## Expand to sender pacing and window control

A history transformer learns a bounded correction to a local AIMD-rate,
bandwidth-delay-window baseline. Its zero output exactly reproduces that
baseline. All sender variants use the same receipt batch-8/25-ms feedback.
This is sequential sender learning with receiver behavior fixed, not joint
endpoint learning. The correction is limited to ±0.5 in each action dimension;
for example, it cannot completely cancel a baseline rate action of -1. That
restriction limits conclusions about the unconstrained sender policy space.

Three seeds receive 320 episodes each; initial and 80-episode snapshots compete
on validation seeds 10000–10004. Seed 44 / episode 80 is selected. AIMD is the
validation-selected deterministic reference among AIMD, fixed-rate, and the
window-aware AIMD controller. On seeds 11000–11019, every sender completes
240/240 transfers. Relative to AIMD, the selected sender is 5.3% faster on
average, but its paired interval includes zero. ACK traffic falls 16.1%, while
data traffic rises 4.6% and **total traffic rises 2.7%**. A feedback-only win
criterion would hide that tradeoff. The analysis was corrected after examining
the test breakdown to use total wire bytes; this is exploratory analysis, not
a preregistered superiority claim. No clear win remains. Fixed-rate is fastest
on this final diagnostic sample despite not being selected on validation.

On new generalization seeds 13000–13019, selected/window-aware/fixed controllers
complete 560/560, versus 554/560 for plain AIMD. Their mean elapsed times are
3.955/3.973/2.718 s, versus 4.389 s for AIMD. Window-aware control accounts for
most of the reliability improvement; the neural correction has no clear
advantage over the stronger deterministic controls. Fixed-rate still has a
finite in-flight window and retransmission timers: it is not an unconstrained
packet flood. See [sender results](../../artifacts/sender-residual-v1-results.json)
and [generalization](../../artifacts/protocol-generalization-v1-results.json).

## A more demanding application workload

Two 128-KiB bulk transfers now share queues with sixteen staggered 1-KiB text
messages. Five network cases and five fresh seeds yield 450 transfers per
controller. This exploratory challenge set is evaluated with frozen policies;
it is not used to tune or select a checkpoint.

All controllers deliver all 400 short messages. The learned residual sender
completes 44/50 bulk transfers, plain AIMD 40/50, window-aware AIMD 43/50, and
fixed-rate 50/50. Short-message p95 latencies are 4.429/4.547/4.770/4.614 s.
Lower wire totals with incomplete bulk transfers are **not efficiency wins**.
This gives future experiments a concrete application target: protect short
message responsiveness without sacrificing bulk delivery. It also exposes the
limits of training only on simultaneous finite transfers.
See [mixed-workload results](../../artifacts/mixed-workload-v1-results.json).

## Reproduce and inspect

```bash
trace-transport console --open
# Select “Receipt summaries · latency focused” or “Receipt summaries · fewer ACKs”.
trace-transport benchmark --feedback-profile budgeted --reward completion
trace-transport benchmark --feedback-profile sparse --reward completion

.venv/bin/python scripts/receiver_headroom.py
trace-transport receiver-categorical
trace-transport sender-experiment
.venv/bin/python scripts/protocol_generalization.py
.venv/bin/python scripts/mixed_workload.py
.venv/bin/python scripts/feedback_budget_experiment.py
trace-transport receiver-categorical --wide
.venv/bin/python scripts/receiver_wide_generalization.py
```

The new feedback profiles are simulation controls. UDP currently supports
receipt summaries with immediate feedback, but not these wider scheduled
receiver policies. Configurations and seed splits are in
[protocol-v3.json](../../experiments/protocol-v3.json); candidate weights,
source fingerprints, raw paired records, and compressed training logs accompany
the repository. Model selection is always completed before the corresponding
test outcomes are read. Small validation sets and stylized network models limit
these results; real Wi-Fi calibration and application-aware training remain
necessary before deployment claims.
