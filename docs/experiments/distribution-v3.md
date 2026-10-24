# V3 environment distribution: adaptation first

Version: `goodput-domain-v3.0`. This is a transport-level synthetic distribution,
not a measured distribution of Wi-Fi conditions or deployment prevalence. V2.0
and v2.1 findings remain unchanged. The new installed training profile is
`goodput-v3`; old profiles retain their samplers and scoring.

## Coverage contract

| Stratum | Mechanism | Intended deployment analogue | Coverage limitation |
| --- | --- | --- | --- |
| steady | Constant service; varied queues and workload | Stable links/control cases | No independent cross traffic |
| wired-lan | 10–80 Mbps, 2–12 ms RTT, independent directions | Caliban over Ethernet | Does not cover full gigabit LAN capacity or OS bottlenecks |
| wifi-contention | Shared airtime; correlated forward/reverse service changes | Herman competing for airtime | No DCF/backoff or independent Wi-Fi stations |
| mobile-fade | Persistent multiplicative capacity drift with residual loss/delay | Changing reception during robot movement | No RF, rate adaptation, or link-layer retry simulation |
| burst-interference | Alternating persistent high/low residual loss | Interference bursts | Conditional packet draws, not calibrated RF traces |
| capacity-opportunities | Random short service increases and contractions | Temporary spare service | No advance warning; prediction requires observable temporal structure |
| propagation-step | Delay changes without capacity changes | Route/path-delay change | Not asserted to be a common local-Wi-Fi event |
| reverse-congestion | Reverse service contraction without forward contraction | ACK/control traffic bottleneck | FIFO service abstraction |
| loss-cause-switch | Random loss versus queue-driven loss | Different causes requiring different responses | Cause remains hidden; identifiability must be measured |
| outage-recovery | Bounded no-delivery interval, endpoint state retained | Brief disconnection/reassociation | No permanent host failure or route discovery |
| producer-churn | Source joins/leaves without deleting outstanding messages | Applications starting/stopping | Source churn, not actual host crash/rejoin |
| bufferbloat | Large queues relative to path bandwidth-delay product | Deep bottleneck buffering | Drop-tail only, no ECN/AQM |

Families have equal synthetic sampling weight; this provides coverage rather than
estimating their real frequency. Base service is 0.5–80 Mbps, base RTT 2–120 ms,
clients 1–4, and queues span roughly 0.25–4 initial BDP, with a distinct large-buffer
stratum. Hidden schedules use random 80-ms–2.5-s dwell times and random first
transition times. Scheduled service peaks may exceed the base-rate range.

Workloads vary status payload (64/128/512 B), rate (5/10/20 Hz), image objects
(16/64/128 KiB), map objects (64/256/512 KiB), and concurrent image slots (1/2/4).
Reliable arbitrary bytes remain the contract. Queues are bounded; producers refill
on confirmed credit. Bulk-only and fixed-rate camera production are still separate
extensions: this version trains a continuously backlogged mixed workload, not a
complete camera arrival model. Longer default 12-s sessions and 32-step (640-ms)
histories allow several regime changes; do not infer long-term forecasting from
this short window. The audit uses 6-s sessions for bounded control sensitivity.

## Realism evidence and remaining gaps

[Pantheon](https://www.usenix.org/conference/atc18/presentation/yan-francis)
motivates checking emulators against real paths and multiple controllers rather
than treating broad randomization as proof of realism. The
[ns-3 Wi-Fi model](https://www.nsnam.org/docs/models/html/wifi-design.html) includes
MAC access, aggregation, propagation, error and rate-control mechanisms absent
here; our shared serialized service is a coarse approximation.
[CellReplay](https://www.usenix.org/conference/nsdi25/presentation/sentosa) shows
that replay accuracy can depend on workload in cellular settings. This motivates
testing load sensitivity of any future device-derived service model; it does not
validate our Wi-Fi simulator.

A deployment coverage claim requires whole-session measurements on Herman Wi-Fi
and Caliban Ethernet: application goodput and timing under idle and loaded paths,
RTT distributions and tails, residual loss bursts, outages, direction asymmetry,
and changes across locations/sessions. ICMP alone is insufficient for capacity.
Keep entire locations/sessions/days held out; never split adjacent trace windows
across train and test. Record capture method and workload, not just timestamps.
Device measurements and credentials remain local unless sanitized for publication.
Only the small idle inventory check below has been performed; no loaded-path calibration has been performed.

Important uncovered settings include independent TCP/QUIC competitors, more than
four clients, gigabit service, ECN/AQM, multi-hop/per-host RTT diversity, true
bidirectional bulk, endpoint CPU/inference delays, and permanent disconnection.
Do not claim broad Internet coverage or calibrated robot performance.

## Audit and retraining gates

1. Generate 1,000 labeled configurations and verify range/stratum coverage and
   reproducibility. Labels and schedules remain unavailable to deployed actors.
2. Compare legal deterministic controls on two seeds per stratum: ACK choices,
   EDF/DRR scheduling, fixed/AIMD/BDP, old delay-target and wider-bandwidth delay
   target. These are diagnostics, not statistically powered winners.
3. Report control sensitivity by stratum. A per-case best action is hindsight
   information and not a deployable oracle. Separate infeasible deadlines and
   zero-completed-object cases from integrity violations.
4. Run a bounded joint v3 pilot with a receiver teacher prior of 0.5 (sender prior
   stays 0.9). Log receiver choice counts and mean normalized PPO advantages by
   action. Keep a teacher initial checkpoint and compare on new validation seeds.
5. Expand to multiple seeds, memoryless/shuffled-history ablations, transition
   recovery-time metrics and whole measured paths before acceptance claims.

The wider-bandwidth delay reference uses a 50-us queue-delay floor; v2 used 1 ms,
which implied an approximately 8-Mbps target ceiling with 512-B packets. This
change is part of the v3 reference, not a learned gain. All controllers share the
16-MB/s rate and 2-MiB window limits. The exact-byte reliability kernel and v2.1
reward formula are retained; v3 versions the domain/profile and records the
exploration change separately.

[Sampler](../../src/trace_transport/goodput_distribution.py),
[audit runner](../../scripts/goodput_distribution_audit.py),
[experiment register](register.md).

## Completed calibration audit, 2026-10-24 UTC

The audit generated 1,000 configurations covering all twelve strata, then ran
216 deterministic case/controller evaluations (two cases per stratum, nine legal
controls). This is a sensitivity/calibration bank, not held-out performance.

| Stratum | Old delay / ACK16 B/s | Highest-goodput control | B/s | Its miss rate |
| --- | ---: | --- | ---: | ---: |
| steady | 62,805 | bdp | 232,107 | 37.67% |
| wired-lan | 409,600 | aimd | 972,117 | 0.10% |
| wifi-contention | 32,768 | fixed | 147,456 | 54.44% |
| mobile-fade | 43,691 | aimd | 65,536 | 51.25% |
| burst-interference | 62,805 | fixed | 114,688 | 25.31% |
| capacity-opportunities | 229,376 | delay-ack1 | 1,103,189 | 0.42% |
| propagation-step | 24,576 | delay-ack1 | 319,488 | 2.92% |
| reverse-congestion | 21,845 | delay-ack4 | 54,613 | 87.41% |
| loss-cause-switch | 184,320 | delay-ack1 | 432,811 | 66.00% |
| outage-recovery | 333,141 | delay-ack4 | 744,107 | 20.68% |
| producer-churn | 294,912 | delay-ack4 | 502,443 | 0.53% |
| bufferbloat | 267,605 | bdp | 1,288,875 | 0.00% |

Highest-goodput controls in this table are chosen after observing each stratum and
can sacrifice deadlines; they are not deployable history policies or acceptance
winners. They show legal-control sensitivity. The pilot reference is selected
across the entire calibration bank using deadline eligibility first (within 0.5
percentage points of the lowest aggregate miss rate), then goodput. That selects
the wider-bandwidth `delay-fast` / ACK16 teacher: 186,254 B/s and 27.88% misses.
Immediate ACKs reach 409,259 B/s but 28.98% misses, failing that eligibility rule.
Different strata favor different controls, supporting an adaptation hypothesis
without proving that endpoint observations can select them reliably.

[Coverage summary](../../artifacts/distribution-v3-coverage.json),
[audit configuration](../../artifacts/distribution-v3-audit-plan.json),
[paired records](../../artifacts/distribution-v3-audit-results.json).

A 300,000-decision CPU joint retraining pilot started 2026-10-24 UTC with the
selected reference, a 0.5 receiver teacher prior, 32-step histories, and separate
24-case stratified validation. No reserved acceptance seeds are used.
[Pilot configuration](../../artifacts/goodput-v3-pilot-plan.json),
[runner](../../scripts/goodput_v3_pilot.py).

## Small device inventory check

Two ten-packet idle probes on Herman observed 0% loss in each tiny sample, with
RTT ranges 0.983–97.342 ms and 1.004–9.553 ms. Caliban's successful ten-packet
probe observed 0% loss and 0.705–0.772 ms; its interface reports 1,000 Mbps.
Herman's reported PHY rates varied between snapshots; these are not available
transport bandwidth measurements. The first Caliban ping used an SSH alias that
ping could not resolve; the second used the SSH-resolved hostname. SSH inventory
succeeded in both attempts. No saturation traffic or device configuration changes
were used.

These snapshots expose gaps rather than validating the distribution: the current
2-ms minimum RTT misses Caliban's sub-millisecond idle path, the 80-Mbps base
service range does not establish coverage of gigabit throughput, and rare Wi-Fi
latency spikes need repeated measurements under varied loads. Before a long
acceptance study, extend low-latency/high-rate strata and profile the 20-ms actor
control interval versus these RTTs. The bounded v3 pilot is diagnostic retraining,
not a claim that deployment calibration is complete.

[Sanitized snapshots](../../artifacts/device-link-snapshot-v1-results.json),
[read-only capture tool](../../scripts/device_link_snapshot.py).

## Completed v3 pilot, 2026-10-24 UTC

The joint CPU pilot reached 300,008 endpoint decisions in 403 s, followed by
120 case/controller evaluations (24 new stratified cases, five modes). These
are pilot validation findings from one seed, not reserved acceptance results.

| Mode | Verified bulk B/s | Interactive misses | Confirmed completion fraction |
| --- | ---: | ---: | ---: |
| adaptive | 162,588 | 11.95% | 98.02% |
| initial | 162,588 | 11.95% | 98.02% |
| sender-only | 165,604 | 11.96% | 98.06% |
| receiver-only | 162,588 | 11.95% | 98.02% |
| joint | 165,604 | 11.96% | 98.06% |

Joint and sender-only results match: goodput is 1.85% above adaptive, with paired
95% difference interval −2,788 to +9,842 B/s. Deadline difference interval is
−0.488 to +0.503 percentage points, exceeding the fixed +0.5-point guard; do not
round it into a pass. All modes have four host-cases with no fully verified bulk
object during measurement, so the host guard also fails. This is not evidence of
zero packet service or permanent starvation. No wrong delivery was observed.

The initial model reproduces the adaptive reference exactly. Receiver-only
records also match the reference. During training, 120,663 active receiver actions
were sampled: 58,403 (48.4%) followed the teacher and all 21 alternatives were
sampled. Yet all 14,504 valid-history receiver proposals in a separate four-case
validation-range audit chose the teacher greedily (including inactive rows).
More exploration alone did not produce useful greedy receiver control. Mean PPO
advantages grouped by sampled choice are observational diagnostics, not causal
counterfactual action values, because the sender and world also change.

Next learning work should compare receiver actions from cloned identical states
over a sufficiently long rollout, establish conditional control headroom and
observability, then test an auxiliary action-value/imitation objective or a
structured receiver action representation. Retain strong deterministic ACK
comparisons. Expand to multiple training seeds and history ablations after this
diagnosis; a larger model or harder independent noise is not the immediate fix.
Deployment coverage also needs the measured-path gaps described above before a
long acceptance study.

[Validation and paired intervals](../../artifacts/goodput-v3-pilot-validation.json),
[training action telemetry](../../artifacts/goodput-v3-pilot-receiver-telemetry.json),
[greedy proposal audit](../../artifacts/goodput-v3-pilot-action-audit.json),
[final actor export](../../artifacts/goodput-v3-pilot-final-actors.pt).

Review correction: same-case shadow-teacher returns do not guarantee common packet impairment draws or variance reduction after actions diverge. See the [v4 review note](caliban-v4.md#review-correction-teacher-return-is-not-a-crn-control-variate); original result values are unchanged.
