# Joint-v1 results — 2026-10-23

**Complete; no learning win.** Six runs reached at least five million active
endpoint decision opportunities each. Screening evaluated 59 checkpoint paths.
Full validation compared three identical initial controllers and six trained
candidates; only the initial controllers passed all bulk/completion/wire guards.
Validation selected `seed42/initial.pt`, with zero training decisions. It is not
a successful learned checkpoint.

The reference is fixed-rate/EDF with batch-8 feedback and a 250-ms maximum wait.
The selected initial actors follow fixed-rate/EDF and a 25-ms feedback teacher.
Their difference is deterministic feedback configuration, not learning. Initial
validation miss rates were 39.45% versus reference 39.31%; none of the six trained
full-validation candidates improved the initial policy ranking.

Calibration: 30000–30003; screening: 40000–40003; validation: 41000–41011.
Test: 50000–50039; OOD: 60000–60039. Selection was frozen before test evaluation.
Each session offers traffic for 20 seconds and drains for up to 20 seconds.
All 560 case/mode evaluations completed: 7 modes × 40 cases × 2 suites.
No wrong delivery was observed. These are paired network seeds, not 560 independent
network conditions or confidence intervals over training seeds.

## Held-out test

| Mode | Interactive misses | Image MiB/s | Map MiB/s | Confirmed / admitted | Wire / useful byte |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected initial policy | 40.00% | 0.0795 | 0.0773 | 138,769 / 138,811 | 1.613 |
| Deterministic reference | 40.71% | 0.0758 | 0.0760 | 138,341 / 138,381 | 1.569 |
| Selected sender + reference receiver | 40.71% | 0.0758 | 0.0760 | 138,341 / 138,381 | 1.569 |
| Reference sender + selected receiver | 40.00% | 0.0795 | 0.0773 | 138,769 / 138,811 | 1.613 |
| History-one training | 46.33% | 0.1076 | 0.0854 | 132,424 / 132,993 | 4.548 |
| Sender-only training | 48.65% | 0.1299 | 0.1402 | 138,859 / 139,163 | 1.947 |
| Receiver-only training | 39.82% | 0.0781 | 0.0758 | 138,666 / 138,706 | 1.587 |

Selected-initial versus reference relative miss reduction: **1.74%**. Paired seed-block bootstrap
95% interval for the absolute miss-rate difference: [-1.729, 0.194] percentage points.
Predeclared win condition: **not met**. Failed resource guard: completion.

## OOD handoffs and asymmetry

| Mode | Interactive misses | Image MiB/s | Map MiB/s | Confirmed / admitted | Wire / useful byte |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected initial policy | 79.07% | 0.0486 | 0.0381 | 180,876 / 180,991 | 1.823 |
| Deterministic reference | 79.68% | 0.0421 | 0.0320 | 180,634 / 180,771 | 1.839 |
| Selected sender + reference receiver | 79.68% | 0.0421 | 0.0320 | 180,634 / 180,771 | 1.839 |
| Reference sender + selected receiver | 79.07% | 0.0486 | 0.0381 | 180,876 / 180,991 | 1.823 |
| History-one training | 77.02% | 0.0546 | 0.0278 | 172,809 / 173,745 | 7.714 |
| Sender-only training | 77.54% | 0.0922 | 0.0778 | 175,382 / 177,832 | 2.329 |
| Receiver-only training | 79.50% | 0.0459 | 0.0307 | 180,888 / 181,015 | 1.855 |

Selected-initial versus reference relative miss reduction: **0.76%**. Paired seed-block bootstrap
95% interval for the absolute miss-rate difference: [-0.947, -0.348] percentage points.
Predeclared win condition: **not met**. All resource guards pass, but the 20% relative-reduction target fails.

## Interpretation and limits

On the test set the interval includes zero. The strict confirmed-completion ratio
also decreases slightly (99.96974% versus 99.97109%). On OOD, the difference interval
excludes zero but relative improvement is only 0.76%, far below 20%. In both cases
it compares untrained deterministic behavior. Do not attribute these gains to RL.
The history and frozen-role arms were fixed ablations, not candidates for a new
post-test selection. Some increase bulk delivery while increasing misses or wire
cost: no single scalar training curve resolves that tradeoff.

Goodput counts completed receiver payload bytes per summed simulated session
time. It excludes neural execution time. Desired interactive offers include
backpressure; no physically impossible cases are removed. OOD deliberately
includes intervals that cannot satisfy a 100-ms one-way application deadline.
Training was interrupted and resumed with optimizer/RNG state and fresh collector
worlds. This is not uninterrupted bitwise replay. A native exit 139 occurred; a
later worker fault was captured in CPython 3.12.10 JSON encoding at network seed
50016. Canonical Python container encoding and per-case persistence mitigated that
path, and all remaining cases completed. The underlying native cause is unconfirmed.
Asynchronous evaluation and batching preserve per-world histories and control
order; arbitrary trained actors are not promised bit-identical floating-point
outputs across batch configurations.

## Devices and next experiment

Herman Wi-Fi and Caliban Ethernet recovered 36/36 synthetic transfers across
deterministic and trained-sender-only checks. These were sequential correctness
trials with deterministic remote feedback, not fair randomized speed comparisons.
The selected initial model was not the trained sender-only model used on hardware.
ICMP observations do not estimate available bandwidth.

The next [goodput-v2 plan](../goodput-v2-plan.md) changes the objective explicitly:
continuously backlogged bulk sources, verified useful throughput, and a mixed mode
constrained by interactive deadlines. It requires new training and held-out tests;
the present study is not retroactively scored under those criteria.

## Artifacts

- [Full results and raw paired records](../../artifacts/joint-v1-results.json)
- [Selection and validation summaries](../../artifacts/joint-v1-selection.json)
- [Calibration and reference](../../artifacts/joint-v1-baseline.json)
- [Plan and seed splits](../../artifacts/joint-v1-plan.json)
- [Manifest, configurations and source hashes](../../artifacts/joint-v1-manifest.json)
- [Compressed training logs](../../artifacts/joint-v1-training.json.gz)
- [Selected **untrained** actor pair](../../artifacts/joint-v1-actors.pt)
- [Device results](../../artifacts/device-binary-v1-results.json)
- [Runtime notes](../../artifacts/joint-v1-runtime-notes.json)
- [Methods and reproduction](../joint-network-brain.md)
- [Paper](../../paper/main.tex)

The manifest also lists final actor-only exports for all six training runs.
Critic weights and optimizer/RNG state are excluded from deployment exports.
