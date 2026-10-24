# Caliban v4: parallel training and deployment coverage

This follow-up runs on Caliban while local robustness work continues. It uses a
new `goodput-v4` profile, preserving earlier v2/v3 results and checkpoint semantics.

## Coverage and controller changes

V4 includes wired service up to 1 Gbps and 0.3–4-ms wired RTT, motivated by
Caliban's reported gigabit interface and roughly 0.7-ms idle probe. Wireless base
service spans 0.5–300 Mbps; PHY rates are not available capacity measurements.
The twelve v3 strata remain, with larger image/map objects and bounded buffers.
These are coverage hypotheses, not calibrated deployment distributions.

Actor control intervals vary among 2/5/10/20 ms. The interval is explicit in both
actors' local observations, replacing the episode-clock feature; sender RTTs use
millisecond-scale log features. Feature dimensions remain the same, but semantics
are versioned and old checkpoints must not be applied to v4 worlds. The adaptive
reference uses a 1-us queue-delay floor so it does not have the earlier implicit
rate ceiling on high-capacity links. All policies share the same rate/window caps.
More endpoint opportunities at short ticks do not imply more simulated experience;
report world steps, episodes and simulated duration alongside training decisions.

## Review correction: teacher return is not a CRN control variate

The v2.1/v3 shadow teacher starts from a cloned RNG state. Different actions change
packet counts and therefore consume impairment draws differently. The two worlds
share the case/schedule but not guaranteed matched packet loss, corruption,
duplication or jitter. Subtracted teacher returns are a same-case baseline,
without a demonstrated variance reduction guarantee; they could increase variance.
The zero-action equality test does not establish coupling after actions diverge.
The existing scientific results remain unchanged and retain this limitation.

V4 longer training disables shadow-teacher subtraction and uses raw task reward
with PPO's learned value baseline. A regression test demonstrates RNG divergence
under nonzero actions. Receiver action diagnostics retain their finite-horizon,
action-dependent-randomness caveat. Evaluation remains paired by case/seed, not
claimed to have identical packet impairments across policies.

The other PR #4 review fixes select evaluation profile explicitly from checkpoint
metadata, name the fixed maximum endpoint dimension, and look up masked observation
features by name rather than numerical positions.

## Remote runtime and resource bounds

Caliban has 12 CPU cores and about 61 GiB RAM. At inspection about 35 GiB was
available; an existing workload remains untouched. Its root filesystem had only
about 1.5 GiB free, so the Python environment, source and raw runs are in an
executable RAM-backed user runtime. No sudo, OS/JetPack changes, robot motion,
GPU contention or system Python installation is used.

The runtime uses uv 0.7.2, Python 3.12.3, NumPy 2.5.4 and PyTorch 2.14.1+cpu from
the official CPU wheel index. This is an explicit CPU-wheel override of the
project's default PyTorch build; source manifests and runtime versions accompany
results. CUDA is unavailable in this environment. GPU acceleration would require
a compatible Jetson build and separate verification; this study does not use it.

A named user service has affinity to six cores, CPU quota request 600%, RAM max
8 GiB, swap max 512 MiB, tasks max 128, low priority, file logs and no terminal.
A study has a 72-hour limit. User-cgroup CPU quota availability varies; affinity
is the enforceable CPU-count bound. Do not kill unrelated processes to stop it:

```bash
ssh caliban systemctl --user stop trace-caliban-v4-study
```

The runtime directory disappears on reboot. Every 30 seconds, source metadata,
progress, all selection snapshots and optimizer checkpoints are atomically mirrored
into a dedicated persistent study directory. Mirroring requires 1 GiB disk free
and caps its selected data at 256 MiB; failure interrupts training rather than
filling the disk. The source archive is also retained for exact-code recovery.
Raw logs and environments stay out of Git. No automatic cleanup of other data is
performed.

## Benchmark and longer study

The same 64-step-history, width-64, two-layer model was benchmarked with 1/2/4
workers. Caliban measured about 68.1/72.2/77.2 endpoint opportunities per second;
four workers were selected from this short full-distribution benchmark.
Eight-endpoint sender+receiver inference measured about 18.8-ms median and
20.1-ms p95 on CPU. **This model is not deployable at a 2-ms control cadence on
Caliban as measured.** Simulator control ticks do not include wall-clock inference
costs. Faster/distilled actors, slower/event-triggered policy updates and a live
inference-delay model remain necessary deployment work.

The long plan has three sequential training seeds (42/43/44), five million
endpoint opportunities per seed, checkpoints every 500,000, and separate
calibration/validation banks. It calibrates adaptive delay/ACK settings, BDP and
AIMD before training, keeps the initial teacher as a candidate, and selects by
validation guards then goodput. Reserved test/OOD ranges stay unopened. These are
planned/running training budgets, not achieved performance or speedup claims.
At benchmark throughput this is roughly a multi-day CPU job; extrapolation from
a short benchmark is uncertain. Caliban supplies extra capacity in parallel with
the workstation, not a demonstrated per-run acceleration.

## Operation and recovery

The deployment helper creates immutable source slots, verifies a source-only ZIP
manifest, installs the package with uv, and runs targeted remote tests. It requires
the isolated uv/PyTorch runtime prepared above; it does not silently install a GPU
stack or use the root disk for a full environment.

```bash
.venv/bin/python -m scripts.caliban_deploy status
.venv/bin/python -m scripts.caliban_deploy study --source-slot source-study
```

The persistent directory is `~/trace-transport-studies/caliban-v4-20261024` on
Caliban. After a reboot, rebuild the isolated runtime with the pinned CPU packages,
restore its exact `source.zip` and manifest, install that source, and copy the
persistent study files into `runs/study` under the runtime. Then use the same
source slot with `study --resume`. Resume rejects source/config mismatches and
restores optimizer/RNG state; in-flight simulated worlds and histories are reset
explicitly, so it is not bitwise continuation of a live trajectory.

[Coverage implementation](../../src/trace_transport/goodput_deployment.py),
[study and benchmark](../../scripts/caliban_study.py),
[deployment/status helper](../../scripts/caliban_deploy.py),
[receiver counterfactual diagnostic](../../scripts/receiver_counterfactual.py).

## Reviewable artifacts and diagnostic scope

The matched workstation benchmark reached 164.9 opportunities/s with four workers,
versus 77.2 on Caliban. This is one short trial per worker setting on each machine;
it estimates extra capacity, not a statistically established combined speedup.
The larger model's workstation CPU inference also exceeds the 2-ms tick.

Eight receiver diagnostic states were attempted; seven had pending ACKs. All 22
legal actions were compared from each identical cloned state over a one-second
continuation. Different actions maximize return in different states. These
selected maxima can exploit future RNG noise; they are not expected action values,
proof of observability, or a runtime oracle. One missing-state case is retained.
Replicated future draws and longer horizons are required before using such labels
as supervised learning targets.

[Caliban benchmark](../../artifacts/caliban-v4-benchmark.json),
[workstation benchmark](../../artifacts/v4-local-benchmark.json),
[frozen study plan](../../artifacts/caliban-v4-plan.json),
[frozen deployed source](../../artifacts/caliban-v4-source.zip),
[receiver interventions](../../artifacts/receiver-counterfactual-v4-results.json).

The service has started its calibration/training pipeline. The three five-million
decision budgets are planned work, not completed training; no performance win or
reserved-test result is reported. Use the status command for actual progress.

## Local follow-up: elapsed-time credit assignment

A fixed per-step `gamma=.999, lambda=.98` has about ten times less physical trace
horizon at 2-ms control ticks than at 20-ms ticks. This is a plausible learning
limitation, not an established explanation of the negative results. The local
opt-in `--time-aware-gae` variant uses `gamma_t = gamma**(dt/.02)` and
`lambda_t = lambda**(dt/.02)`, with actual transition duration (including a shorter
terminal step). Rewards remain event increments. Terminal states never bootstrap.
Metadata records `mappo-time-gae-v1`; the default stays `mappo-step-gae-v1`.

This implementation is tested for equal prefix trace decay across 2/20-ms ticks,
legacy equivalence at 20 ms, terminal handling and a tiny training integration.
It has no performance result yet and is not enabled in the frozen Caliban study.
Action scaling, exploration, rollout span and real inference delays also change
with control frequency; time-aware GAE alone does not establish discretization
invariance. [Tallec et al., ICML 2019](https://proceedings.mlr.press/v97/tallec19a.html)
study the broader problem for Q-learning; this PPO variant is not a reproduction
of their algorithm or results.

A controlled next comparison should keep the v4 distribution, initial teacher,
model, rollout span and decision budget fixed, vary only this flag, and use fresh
validation cases. Use longer rollouts where necessary so a 2-ms tick can cover
ACK timers up to 250 ms plus path delay. Keep both variants' source frozen while
continuing development elsewhere.

[Implementation](../../src/trace_transport/learning.py),
[timing tests](../../tests/test_time_gae.py).

[Exact remote package inventory](../../artifacts/caliban-v4-runtime.json) accompanies the frozen source and checkpoints.
