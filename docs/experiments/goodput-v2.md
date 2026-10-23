# Goodput-v2.0 — pilot completed 2026-10-23; no validation win

Hypothesis: continuously backlogged verified objects, an adaptive starting point,
strong teacher priors during sampling, and shorter reward delay will support useful
joint learning. Additional challenge comes from switching loss causes, ACK-path
bottlenecks, randomly timed capacity/RTT changes, and source-flow churn. These are
synthetic service models, not calibrated robot Wi-Fi or full TCP implementations.

The [literature review](../research/learning-opportunities.md) explains the design.
The previous joint-v1 result and criteria remain unchanged.

## Current execution

- Calibration completed: 24 controls = fixed/AIMD/window-aware/delay-target ×
  three initial pacing rates × two feedback profiles, all with EDF scheduling.
- Calibration seeds: 71000–71007. Deadline-eligible controls are within 0.5
  percentage points of the lowest calibration miss rate; maximize goodput among
  them. This selects delay-target, initial 512,000 B/s, budgeted feedback.
- The delay-target baseline is inspired by Copa's queue-delay/rate relation. It
  omits full Copa coexistence logic; this is not a reproduction of Copa or BBR.
- Full validation reference uses seeds 73000–73015: 157,184 verified bulk B/s,
  4.26% interactive misses, zero wrong deliveries and no starved hosts.
- A CPU-only pilot completed 300,011 decisions in 251 seconds around that teacher, seed 42,
  two environments, history 16, width 32, one transformer layer, rollout 64,
  three optimization epochs, GAE lambda 0.98. Validation and a seven-snapshot sweep are complete: the initial teacher wins ranking; no trained snapshot passes the deadline guard.
- Reserved final test seeds 74000–74039 and OOD seeds 75000–75039 remain unopened.
  All selection and any additional tuning precede frozen testing.

This experiment's local files are `runs/goodput-v2/plan.json`, `reference.json`,
`cases/`, and `pilot42/`. Compact public results and hashes will be attached when
available. Planned scaling to three larger training runs is conditional on
pilot validation and headroom, not automatic spending of more compute.

## Objective and implementation

Forward image/map sources continuously refill within explicit byte/slot credit.
Interactive status traffic runs in both directions. Accepted bytes are never
silently evicted. A separate environment mode supports bulk-only simulation;
the first training study uses mixed traffic. Image objects are 16 KiB and map
objects 64 KiB, making verified completion rewards more frequent. A file/object
manifest for larger application objects remains future work.

Primary goodput counts fully verified image/map object bytes delivered during the
fixed six-second production window. Two seconds of drain are reported separately
and cannot inflate measured speed. Training rewards verified delivery, charges
wire traffic and deadline misses, and adds a small separately counted drain
reliability reward. Initial link service normalizes training reward only; actors
never see link capacity, future changes, or family labels. Input schemas remain
local and histories persist across objects.

The initialized receiver samples its teacher action with probability 90%; sender
scheduling is at least 90% teacher over the maximum candidate count. Continuous
exploration is smaller than v1. Both actors and the centralized critic train
jointly. Checkpoints record the adaptive teacher, source hashes, objective and
workload versions. Existing v1 inference/scoring is retained, and the old live
file adapter rejects v2 checkpoints until it can compose their adaptive teacher.

Proposed acceptance is the frozen goodput-v2 plan: ≥10% useful-goodput improvement
with paired interval above zero; deadline-difference interval upper bound ≤0.5
percentage points; wire/useful-byte cost ≤105%; zero wrong delivery and no host
starvation. Per-class delivery, sender-confirmed service and confirmed-completion
guards prevent a high aggregate score from hiding a reliability regression.
The tuned reference bank is still much smaller than a complete real TCP/QUIC
comparison; passing this test would establish a scoped simulator win only.

## Terminal/tmux investigation and resource isolation

The user journal records `gnome-terminal-server.service` exiting with SIGSEGV at
18:09 on 2026-10-23. This is evidence of a terminal-server crash, not proof of its
cause or a link to our earlier Python crashes. We do not change or restart the
user's terminal, tmux, drivers, or desktop services.

Experiments now run in separate systemd user services, not the terminal scope.
The pilot's verified kernel affinity is logical CPUs 28–31, with 8-GiB memory,
512-MiB swap and 128-task limits, low scheduling priority and file-only logs.
CPU quota is requested too; the CPU quota controller is not exposed in this
user's service cgroup, so verified affinity provides the actual CPU bound.
The pilot allocates no GPU memory. Only one `trace-*` experiment service is
allowed at once. Services stop after at most two hours; killing a service affects
its own process group and workers, not terminal/tmux. Resource limits reduce
interference but are not a confirmed fix for a native desktop crash.

```bash
.venv/bin/python -m scripts.run_limited --name trace-example \
  --log runs/example/service.log -- \
  .venv/bin/python -m scripts.goodput_study calibrate
systemctl --user status trace-example
# Stop only the named experiment:
systemctl --user stop trace-example
```

Do not run the example concurrently with the pilot. Follow progress through
`runs/goodput-v2/pilot42/progress.json` and the service log. Resume requires
matching configuration and source hashes; a changed experiment gets a new
version/directory. Results caches key by frozen seeds, profile and checkpoint.

## Pilot outcome

The final policy gained 1.7% useful throughput in one validation pass and 2.7%
in the snapshot sweep; both failed the deadline guard. All seven candidates
(including initial) were compared on validation only, and the initial policy was
selected. Reserved tests remain unopened. Two passes with the same final
checkpoint hash and network seeds produced different records in 15/16 cases;
asynchronous variable batching and floating-point amplification are a hypothesis,
not a proven cause. A new version will require fixed-shape per-world evaluation
and repeatability checks before a test win can be claimed.

- [Calibration](../../artifacts/goodput-v2.0-calibration.json)
- [Plan/source hashes](../../artifacts/goodput-v2.0-plan.json)
- [Reference](../../artifacts/goodput-v2.0-reference.json)
- [All pilot validation snapshots](../../artifacts/goodput-v2.0-pilot-validation.json)
- [Selection](../../artifacts/goodput-v2.0-pilot-selection.json)
- [Experimental final actor pair, not a better default](../../artifacts/goodput-v2.0-pilot-actors.pt)
