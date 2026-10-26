# What to improve, and when to stop

Planning date: 2026-10-26. Status: proposed diagnostics and experiments, **not new
performance evidence**. This is a focused literature review, not a systematic
survey. [Experiment register](../experiments/register.md).

The current evidence supports neither abandoning learning nor scaling the current
joint policy blindly. V6's 200k-decision, one-seed pilot gives +0.226% verified
object goodput, with paired difference interval [-3,868, +7,625] B/s containing
zero. Quality guards pass. V5's receiver-only audit matches the reference exactly;
v6's receiver contribution is still unmeasured. These are validation results,
not held-out acceptance. [V6 results](../../artifacts/reactive-v6-pilot-results.json).

## Literature and implications

| Primary source | What it establishes or investigates | Implication for this project |
| --- | --- | --- |
| [Mazilu et al., revised April 2026](https://arxiv.org/abs/2510.25105v2), experimental study / preprint | Tests publicly available learned controllers against Cubic and BBRv3; identifies weaknesses under changing capacity/latency and unseen fairness conditions. | Changing networks do not guarantee an RL advantage. Measure transient recovery and fairness explicitly; our custom delay controller is not a BBR/Cubic comparison. |
| [Genet, SIGCOMM 2022](https://uchi-jcl.github.io/group-website/publication/genet/) | Uses conventional-baseline-relative failures to generate a networking curriculum; broad uniform training can be inefficient. | Train on realistic recoverable weaknesses with a replay mixture, rather than adding arbitrary noise or uniformly expanding ranges. |
| [Mutant, NSDI 2025](https://www.usenix.org/conference/nsdi25/presentation/pappone) | Learns online from a selected set of existing congestion controllers with policy switching. | Test learned selection among strong controllers before direct joint packet control. Switching state and transition costs must be modeled. |
| [AliCCS, NSDI 2025](https://www.usenix.org/conference/nsdi25/presentation/zeng) | Reports production use of learned selection among established congestion-control algorithms in a CDN. | There is a practical use for a network-condition classifier/controller selector, even if a clean-slate learned protocol loses. CDN results are not evidence for our robot or simulator. |
| [Nuwa, 2025 preprint](https://arxiv.org/abs/2502.16498) | Proposes receiver-side heuristic congestion control with RL adaptation of a key parameter. | Receiver-side learning can be narrower than ACK action selection. Our ACK-only receiver is not a receiver-driven rate/credit controller. |
| [SIRD, NSDI 2025](https://www.usenix.org/conference/nsdi25/presentation/prasopoulos) | Separates receiver scheduling of owned links from reactive control of shared links. | Coordination authority matters. Receiver grants or per-sender credit are a possible new protocol experiment; a datacenter design cannot simply be assumed suitable for shared Wi-Fi. |
| [Age of Information survey](https://arxiv.org/abs/2007.08564) | Defines and analyzes timeliness of timestamped updates in constrained communication systems. | For robot images, usefulness includes age of the newest complete image, not just total bytes transferred. |

These findings motivate the following proposals; none establishes that our chosen
architecture or task will improve.

## 1. Establish where control has value

Run the v6 final checkpoint on the existing 24-case validation bank in joint,
sender-only, receiver-only and deterministic modes. Preserve profiles and source
hashes. Measure actual applied actions, pending-ACK changes/flushes, wire bytes,
verified delivery, confirmation and deadlines. A proposal with no transport effect
must not count as useful receiver behavior.

Record receiver probabilities and entropy, greedy choice and sampled frequency.
Training samples actions, while current evaluation uses argmax. With a 0.5 teacher
prior and many explicit alternatives, the teacher can remain the largest single
category even when alternatives collectively have substantial probability. This
is a diagnostic hypothesis, not a proven explanation. Compare greedy and a
predeclared, seeded stochastic policy; report stochastic variance and retain the
original greedy result. Do not select whichever deployment rule happens to win.

Then sweep feasible deterministic ACK settings and sender controllers on separate
**development** cases, including capacity changes, reverse-link contention,
non-congestive loss, and overload. Test a small hand-coded adaptive ACK controller.
Report Pareto tradeoffs rather than ranking a controller that violates deadlines
as better. Best-per-case hindsight selection is an optimistic diagnostic comparator,
not an attainable online policy or rigorous optimal bound. It requires a separate
selection bank and uncertainty reporting to avoid selection bias.

**Decision:** if practically different ACK controls offer no guarded headroom,
freeze the receiver and focus compute elsewhere. If headroom exists but the actor
cannot fit a tiny set of repeated cases, fix credit assignment or observations
before broad training. A memory-dependent task must contain locally observable
predictive evidence; independent future shocks cannot be anticipated from history.

## 2. Test learning mechanics with small controlled studies

Use a few feasible, repeatable development cases. Verify policy gradients and
transport sensitivity, then compare receiver-only training with a frozen sender,
sender-only training, alternating updates, and joint training. These are ablations
of the intended joint system, not a claim that joint learning is inherently wrong.
If needed, use role-specific counterfactual advantages, with explicit stochastic
coupling assumptions; cloned packet RNG streams are not true common random numbers
after trajectories diverge.

Count distinct sampled cases, completed episodes and wall-clock cost alongside
endpoint decisions. The v5 matched pilot completed only 18 episodes; many correlated
actions are not broad experience. Measure v6 counts separately. Test reward/credit
horizons against object completion delays and train on mixed feasible object sizes.
Keep verified whole-object delivery as the score; partial-byte progress remains a
training signal and diagnostic only.

Compare the transformer against a current-observation MLP, stacked-history MLP and
GRU at similar parameter and runtime budgets. Compare real history with shuffled
or masked history using a declared reset procedure. Predicting near-future delivered
service or RTT may be an auxiliary task, with labels constructed only after the
prediction time; future values must never enter deployed actor observations.

**Decision:** do not increase model size unless history beats the simpler controls
on new development cases and inference meets the chosen control period. Caliban's
previous measured ~19 ms actor-pair latency cannot service a 2 ms loop.

## 3. Train a smaller, more defensible adaptation problem

First candidate: a history-based selector among existing delay-fast, AIMD and
bandwidth-delay controllers, or a bounded residual around the best adaptive one.
Maintain controller states in shadow, document whether their measurements depend
on executed actions, bound switching frequency, charge all traffic, and test
hysteresis. Prefer an explicit adjustment interval initially around 50–100 ms,
subject to profiling and RTT; keep packet reliability and pacing in the kernel.
No claim of implementing Mutant, AliCCS, BBR or Copa is implied.

Use a Genet-inspired curriculum: emphasize cases where learning loses to the
strong baseline, while retaining broad replay (initial proposal: half broad,
half mined failures). Search only development conditions. Randomize event times,
remove episode-clock shortcuts, and hold out whole trace paths / combinations.
Include mechanisms with temporal persistence: contention episodes, fades, recovery,
shared half-duplex airtime, reverse bottlenecks and competing independent traffic.
Measure effects separately before combining them. Do not tune the test set or
make the baseline artificially weak.

Pilot at 3 seeds x 500k decisions only after the diagnostic gates; advance to
3 seeds x 5M only if development results show consistent guarded improvement.
These are proposed budget caps, not jobs launched by this document. Retain a
frozen raw-reward control so shaping changes can be interpreted.

## 4. Prioritize a robot application with measurable value

A promising hypothesis is exact-byte, application-aware object delivery:

- Reliable text/commands retain explicit acknowledgement and bounded queues.
- Camera images carry capture time and deadline. An explicitly opted-in stream
  may replace an obsolete **unsent** image with a newer one. Report dropped objects;
  never call them exact completions. Accepted images still require exact recovery.
- Map deltas carry version/dependency identifiers; a delta cannot be applied unless
  its base is present. Canceling a dependent delta needs a defined resync path.

Start with admission and scheduling, not learned compression or safety-critical
robot control. Compare EDF, fair queuing, latest-only camera buffering,
size/remaining-work-aware scheduling and their tuned combinations. Report p95 age
of the newest verified image, deadline misses, map lag/resync cost, bytes and CPU.
A learning win over FIFO alone would be weak evidence. This is a new opt-in task,
not a retroactive replacement for the existing throughput benchmark.

Initially measure non-motion transfers on Herman Wi-Fi and Caliban Ethernet:
application send/receive timestamps, object sizes, RTT/loss, retransmission and
interface counters. Fix clock assumptions; RTT needs no synchronized clocks,
one-way age does. Passive throughput is offered-load-dependent and is not ground
truth capacity. Split entire measurement sessions into development and held-out
paths, then test on the real devices. Preserve raw logs locally and publish only
sanitized artifacts. Synthetic Wi-Fi is currently not an RF-calibrated model.

## 5. Decide with a fixed acceptance protocol

For the existing goodput task, keep the frozen >=10% gain target, positive paired
interval and all quality guards. Report each training seed, aggregate across seeds,
per-family regressions and p95 recovery after a condition change. Match measured
real TCP/QUIC baselines in a future emulator/live comparison before claiming an
Internet congestion-control win; simulator stand-ins cannot support that claim.

For the robot freshness task, choose a separate primary criterion before training:
proposed >=10% lower p95 verified-image age, uncertainty excluding zero, with
predeclared command-delivery, map-progress, byte and compute noninferiority bounds.
Set numeric guard bounds after measuring deterministic feasibility, then freeze
before model selection. Correctly account for dropped/superseded images and report
all objectives. Use untouched session/trace splits and at least 3 training seeds;
bootstrap whole sessions/cases rather than correlated packets. Report both paired
case uncertainty and across-training-seed variation, not just a pooled packet CI.

If neither ACK control nor controller selection shows reproducible headroom, keep
the exact-transfer kernel and test harness as useful outputs and focus on robot
object scheduling. If strong deterministic scheduling also closes the gap, deploy
that and retain learning as a negative research result. Stop extending a losing
study merely because more compute is available.

## Immediate order

1. Finish v6 role/action audit; publish measured control effects.
2. Run the separate-development control sweep and tiny-case fit test.
3. Choose between sender adaptation, a selector, and robot object scheduling using
   those results; register the new configuration and acceptance rules.
4. Run bounded multi-seed pilots; scale only after the stated gates.

All four are **planned** here. No new training or device experiment is claimed.
