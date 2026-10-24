# Maximum useful throughput: next experiment

The next objective is to deliver verified useful data as fast as the path permits.
The existing joint-v1 experiment prioritizes 100-ms interactive deadlines and
requires bulk throughput to stay near a reference; that is a different objective.
Its training, selection and test criteria remain frozen. The machine-readable
extension is [joint-goodput-v2-plan.json](../experiments/joint-goodput-v2-plan.json).
The first [bounded prototype and CPU pilot](experiments/goodput-v2.md) are now
implemented and completed as negative validation studies. The full acceptance program remains a plan, not a claim of a
trained throughput win.

## Two workloads

**Bulk-only maximum speed:** keep a sender continuously backlogged with text,
image, map or arbitrary binary objects. Measure verified object bytes per second
and completion time for a finite file. Interactive deadlines do not apply when
there is no interactive traffic.

**Mixed traffic:** keep image/map traffic backlogged while interactive text/status
messages retain the 100-ms target. Maximize useful bulk delivery subject to a
noninferiority constraint on interactive misses and limits on wire overhead and
queue growth. A small control message should not wait behind an entire image.
Report the throughput/deadline tradeoff rather than assuming every operating
point can maximize both simultaneously.

A bounded credit-driven source refills as buffer space becomes available. A fixed
image production rate is not sufficient for a maximum-speed test: it may leave a
fast link idle even when the transport has more capacity. Bounded queues prevent
an infinite source from allocating unbounded memory. Sending more packets is not
itself progress: duplicates, retransmissions and dropped bytes do not count.

## Objective and control

Use a fixed measurement window and count unique bytes belonging to fully verified
completed objects. Report startup, steady-state service, and bounded drain
separately; do not improve the denominator by quietly dropping slow cases.
Receiver-verified and sender-confirmed goodput are both required, along with exact
finite-object recovery tests. Large objects may need independently verified
chunks plus a whole-object manifest so useful progress is credited promptly while
whole-object correctness remains explicit.

Rate, in-flight window, message/class scheduling and receiver feedback are the
initial action space. Histories must expose local demand, credit/backpressure,
ACK rate, RTT variation, loss/reordering and feedback age. Actors still receive no
router capacity or future impairment schedule. They should increase service
when there is demand and spare capacity, then react when congestion or service
changes make continued increases wasteful. A large fixed rate is not a useful
substitute for that behavior.

Start with zero residuals around a strong adaptive deterministic teacher, then
train the actors jointly. If the actor replaces teacher logic, first learn to
imitate it. Checkpoints must record the teacher controls and objective/workload
versions so evaluation and live inference compose the same policy. Score verified goodput directly. If incremental unique-byte progress is
used to make rewards denser, use potential-based shaping with a terminal
correction rather than rewarding endless partial objects. Keep deadline, queue,
wire-cost and active-host constraints explicit. Compare simple policies and
history ablations as well as the transformer; a bigger network alone is not a
solution to a misaligned objective.

## Acceptance and scaling

Before the next training run, freeze the proposed targets: at least 10% more
verified goodput with a paired 95% interval above zero; mixed-traffic deadline
miss noninferiority with the upper confidence bound within 0.5 percentage points; at most 5% more wire bytes per
useful byte; zero wrong deliveries; and no starvation of active hosts. Tune and
select the deterministic reference under the same constraints. The current slow
fixed-rate file adapter is a correctness reference, not sufficient evidence of a
throughput win. The 100-ms target and impossible-deadline counts remain visible;
network outages do not disappear from the denominator.

Start with one backlogged flow and static links, then add mixed producers,
bidirectional feedback competition, multiple hosts, host arrivals/departures,
and persistent capacity/RTT/loss changes. Include high-capacity paths so source
rate, pacing, window, packet encoding, CPU event handling and actor inference can
each become the bottleneck. Report queue/latency tails, retransmissions,
per-host goodput and fairness, and control/inference costs. Use a held-out set of
whole measured paths after calibration; ICMP observations alone are not capacity
measurements.

Reuse CPU event workers with batched GPU learning and asynchronous independent
worlds for evaluation. Profile packet processing and serialization before adding
more workers; oversubscription and simulated packet floods can dominate neural
compute. A GPU-native simulator is a later acceleration path, checked against the
CPU reference. Real-device tests must measure wall-clock completion and include
steady-state inference costs, using the same receiver, workload and randomized
controller order. These extensions need new training and held-out evaluation;
the completed device smoke transfers do not establish the new objective.

The [literature review and difficulty ladder](research/learning-opportunities.md)
explains the new loss-cause, feedback-bottleneck and churn cases, stronger
baselines, and exploration/credit-assignment changes.

## Adaptive-control comparisons and diagnosable headroom

Use the [background review](research/learning-opportunities.md) as design evidence,
not a claim that the proposed scenarios reproduce published results. Aurora and
Remy motivate history and assumption-mismatch tests; Orca motivates an adaptive
teacher; Copa and PCC Vivace motivate stronger references; MOCC motivates explicit
application preferences; Jury motivates per-host fairness; Pensieve motivates a
separate future application-quality/freshness task.

The immediate comparison freezes a trained final checkpoint before evaluating new
paired seeds, and includes adaptive deterministic control, learned sender only,
learned receiver only, and both learned. Use the same exact-byte kernel, traffic,
rate/window limits and teacher configuration. Evaluate coupled capacity changes,
RTT changes, reverse ACK bottlenecks, and loss. Publish negative findings and
per-suite confidence intervals; stress tests are exploratory and do not replace
the reserved acceptance bank. See [comparison runner](../scripts/goodput_changing_comparison.py).

Then isolate each suspected weakness: propagation-delay steps versus queue growth,
reverse-only congestion, short capacity opportunities, fixed ACK overhead versus
latency, and mixed-class scheduling. First measure whether different legal actions
produce meaningful improvements. Compare memoryless and shuffled-history actors
against real history, with multiple training seeds. Audit action distributions,
observation normalization, credit assignment, and training/evaluation parity before
increasing model size or training budget. A deterministic adaptive baseline also
uses historical estimates; changing conditions alone do not guarantee an ML win.

Future utility-optimizing and full transport baselines should be distinguished
from our simplified delay-target, AIMD and bandwidth-delay-window heuristics.
Application quality changes and stale-message cancellation remain separate opt-in
experiments and cannot silently relax exact byte recovery.

## Versioned distribution follow-up

The [v3 coverage contract](experiments/distribution-v3.md) specifies realistic
mechanisms, synthetic sampling weights, deployment gaps, a control-sensitivity
audit, and receiver exploration telemetry. It leaves v2's negative results intact.
Adaptation is the first target; scheduling and receiver controls are retained and
audited separately. Real-world coverage remains a measured-path validation task.
