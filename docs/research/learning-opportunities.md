# Where learning might help — literature review, 2026-10-23

A harder benchmark is useful only if it has controllable headroom and local
observations contain information for choosing better actions. More independent
noise or impossible deadlines can lower every score without helping learning.
The scenarios below are our design proposals, not reproduction of published wins.

| Primary work | Finding relevant here | Proposed change |
| --- | --- | --- |
| [Aurora, ICML 2019](https://proceedings.mlr.press/v97/jay19a/jay19a.pdf) | Local feedback history can support distinguishing congestion from non-congestion loss; fairness, safety and generalization remain challenges. | Include wireless loss without queue growth, congestion loss with queue growth, and switching causes. Keep causes hidden from actors. Randomize transition times; compare history-one and shuffled-history controls. |
| [Remy, SIGCOMM 2013](https://conferences.sigcomm.org/sigcomm/2013/papers/sigcomm/p123.pdf) | Automatically designed control depends on the objective, traffic model and network assumptions supplied at design time. | Keep a continuously backlogged source, source-flow churn, competing hosts, and held-out regimes. Publish the design range and distribution shifts. |
| [Orca, SIGCOMM 2020](https://github.com/Soheil-ab/Orca) | The authors release a hybrid conventional/learned congestion controller rather than requiring a clean-slate replacement. | Learn residuals around a strong adaptive teacher. Record the teacher in checkpoints and use the same composition in evaluation. This is inspiration, not an Orca implementation. |
| [MOCC, EuroSys 2022](https://www.cse.ust.hk/~kaichen/papers/mocc-eurosys22.pdf) | Objective preferences can be inputs to a learned controller rather than fixed invisible reward weights. | Treat throughput-only and deadline-constrained service as explicit modes. Preference-conditioned actors are a later extension; the first v2 training profile has a fixed mixed objective. |
| [PCC Vivace, NSDI 2018](https://www.usenix.org/conference/nsdi18/presentation/dong) | Online utility optimization is another learning approach to rate control, not necessarily deep RL. | Add an online utility optimizer as a future baseline; do not equate a transformer win over weak fixed pacing with a win over learning-based control generally. |
| [Copa, NSDI 2018](https://www.usenix.org/conference/nsdi18/presentation/arun) | Delay-based control connects target rate to queueing delay and addresses coexistence with buffer-filling traffic. | Add a delay-target baseline alongside fixed, AIMD and bandwidth-delay-window control. Our small baseline omits Copa's full coexistence logic and must not be labeled a Copa reproduction. |
| [Jury, EuroSys 2025](https://cse.hkust.edu.hk/~kaichen/papers/jury-eurosys25.pdf) | Learned congestion controllers can generalize poorly in fairness; the work separates fairness handling from the main learned controller. | Report per-host useful bytes and starvation in addition to aggregate throughput, including unseen host counts. A better sum is not a win if another host gets no service. |
| [Pensieve, SIGCOMM 2017](https://web.mit.edu/pensieve/) | Application-level RL chooses video chunk bitrates using past delivery and playback observations. | Application-aware scheduling/freshness may offer more value for robot images and map updates than pacing alone. Lossy encoding or dropping stale images would be a separate opt-in task, never a change to exact-byte recovery. |

## Diagnose the learning setup before adding difficulty

Joint-v1 had a gap between greedy initialization and exploration. With 21
non-baseline receiver choices and a baseline logit of 0.5, the initial sampled
baseline probability was approximately `exp(0.5)/(exp(0.5)+21)`, about 7.3%.
Greedy evaluation nevertheless always chose that baseline. Queue sampling had a
similar problem. This does not prove the cause of the negative result, but it
means sampled training started far from the behavior being protected.

Goodput-v2 starts with a 90% receiver teacher prior and at least 90% scheduling
teacher probability over the maximum candidate set, plus smaller continuous
exploration. It uses verified 16-KiB image and 64-KiB map objects for more frequent
completion rewards, a longer GAE horizon, and fixed-window useful throughput.
These changes require a fresh experiment; they do not reinterpret joint-v1.

## A staged difficulty ladder

1. Establish headroom on backlogged, feasible steady links against tuned controls.
2. Add non-congestion loss versus congestion, reverse ACK bottlenecks, randomly
   timed handoffs and producer-flow churn. These families can require different
   actions even when aggregate loss rates look similar.
3. Add unseen combinations, host counts and whole measured service paths. The
   current Wi-Fi/RTT observations are not calibrated available-capacity traces.
4. Later add explicit application preferences, asymmetric object values and
   interference from independently controlled protocols. Benchmark stronger real
   TCP/QUIC algorithms before claiming Internet congestion-control superiority.

Use a finite-candidate privileged lookahead only as a diagnostic for remaining
headroom, never as a deployable baseline or a proven optimum. If useful decisions
are indistinguishable from endpoint history, adding a transformer cannot supply
missing information. Keep hypothesis changes and failed runs in the register.

## October 26 follow-up

See the [staged improvement plan](improvement-plan-2026-10-26.md) for the 2025–2026
literature update and decision gates after the completed v5/v6 pilots.
