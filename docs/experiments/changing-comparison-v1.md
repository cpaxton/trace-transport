# Changing-network sender/receiver comparison

Actual start/end: 2026-10-24 UTC. Completed, negative exploratory result.

We froze the final 1,000,006-decision joint checkpoint before looking at two new
16-case paired banks (81000–81015 and 82000–82015). The first samples the v2.1
families and includes steady and wireless-loss cases as controls. The second uses
four hosts with coupled capacity steps, RTT changes, reverse bottlenecks and 2%
loss. Each case has six seconds of production and two of drain. This uses new
stress seeds, not reserved acceptance seeds. The final checkpoint was chosen as
the training endpoint, not the validation winner (which was the initial teacher).
One training seed, small synthetic banks, no real-device performance claim.

| Suite | Controller | Verified bulk B/s | Interactive misses | Host-cases with zero verified bulk |
| --- | --- | ---: | ---: | ---: |
| changing | adaptive | 127,829 | 8.77% | 0 |
| changing | learned-sender | 123,563 | 9.26% | 0 |
| changing | learned-receiver | 127,829 | 8.77% | 0 |
| changing | joint-learned | 123,563 | 9.26% | 0 |
| coupled-harsh | adaptive | 45,227 | 39.30% | 1 |
| coupled-harsh | learned-sender | 43,349 | 42.24% | 4 |
| coupled-harsh | learned-receiver | 45,227 | 39.30% | 1 |
| coupled-harsh | joint-learned | 43,349 | 42.24% | 4 |

No wrong delivery was observed in any of the 128 case/controller evaluations.
Zero verified bulk means no fully completed bulk object during the measurement
window; it is not proof of zero packet service or permanent starvation. Reported
completion fractions include the drain and do not prove eventual recovery of all
admitted objects.

On coupled-harsh cases, joint learning has 4.15% less goodput than the adaptive
teacher, with paired 95% goodput-difference interval −6,827 to +3,243 B/s.
Interactive misses increase by 2.94 percentage points (paired interval +1.81 to
+4.17 points). Thus neither a throughput win nor deadline noninferiority is
established. The teacher itself has 39.30% misses and one zero-bulk host-case;
this identifies a difficult regime, not evidence that all misses are avoidable.

Receiver-only records match adaptive records, and joint records match sender-only
records. A four-case inference audit found all 12,088 valid-history receiver
proposals chose action zero (follow teacher). These counts include inactive rows
and are not applied-action counts. Sender mean absolute residual proposals were
about 0.420 for rate and 0.129 for window. Both actors received gradients during
training, but nonzero gradients do not establish a useful learned receiver.
The next diagnosis should examine categorical exploration, advantages by receiver
choice, and teacher-prior retention before scaling training further.

## Specific adaptive-baseline limitations to isolate

Our delay-target heuristic uses `srtt - min_rtt` as queueing-delay evidence. A
propagation-delay increase can look like queue growth; reverse congestion can
also contaminate end-to-end observations. ACK batch 16 / 25 ms is fixed, rate
changes per control step are bounded, and scheduling does not jointly optimize
future mixed-workload utility. These are plausible weaknesses to test separately,
not causal explanations established by the aggregate stress results. Histories
cannot reveal an unpredictable future transition before it occurs.

Measure adaptation headroom with legal alternative actions on each isolated
regime, then compare memoryless, shuffled-history and true-history policies.
Include several training seeds and stronger utility-optimizing references before
claiming a general learning advantage.

[Plan and checkpoint hash](../../artifacts/changing-comparison-v1-plan.json),
[full paired records and intervals](../../artifacts/changing-comparison-v1-results.json),
[action audit](../../artifacts/changing-comparison-v1-action-audit.json),
[runner](../../scripts/goodput_changing_comparison.py).
