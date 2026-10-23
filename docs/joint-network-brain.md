# Joint network brain experiment

This implementation trains separate sender and receiver transformers together with
MAPPO and a shared reward. Each actor sees only its endpoint's history. A separate
centralized critic sees current simulator state during training; it is removed
from actor exports. Models are shared between sessions of the same role.

## Run it

Install learning dependencies with `uv sync --locked --extra learn`, then use the
installed `trace-transport` command (activate `.venv` or use `uv run`).

```bash
trace-transport joint-simulate --clients 2 --report runs/joint-demo.json
trace-transport joint-train --device cuda --envs 16 --seed 42 --output runs/joint-v1/seed42
trace-transport joint-status
trace-transport console
```

Open `/joint` in the console to see training progress, role gradients, and study
results. CPU workers handle packet events; CUDA batches transformer inference and
optimization. This is not a GPU-native packet simulator. Worker count is the main
CPU parallelism control. The GPU needs PyTorch with matching CUDA support.

The full study expects six five-million-active-endpoint-decision runs:
`seed42`, `seed43`, `seed44` with matching seeds; `history1` with seed 42 and
`--history 1`; `sender-only` with seed 42 and `--freeze-receiver`; and
`receiver-only` with seed 42 and `--freeze-sender`. Use the corresponding
`--output runs/joint-v1/NAME`. Then:

```bash
trace-transport joint-study --root runs/joint-v1 --device cuda --envs 16 --wait
trace-transport joint-evaluate --checkpoint runs/joint-v1/study/joint-v1-actors.pt --trials 10
```

Use the actual export path in the study's selection report. Training refuses to
overwrite an existing run. Explicit `--resume PATH` restores a matching checkpoint's
actors, critic, optimizer states, RNG, and decision counters. It resets collector
worlds and histories with fresh episode seeds and records this discontinuity in
checkpoint metadata; it is not exact trajectory replay. `scripts/joint_smoke.py` runs a tiny end-to-end CPU
acceptance study, which checks plumbing and is not scientific evidence.

## Sessions, observations, and actions

Each client has two persistent directed sessions. Producers offer interactive
UTF-8 messages, opaque image bytes, and opaque map-delta bytes. Defaults are
1-KiB interactive messages at 20 Hz with a 100-ms deadline, 128-KiB images at
2 Hz, and 512-KiB map deltas every five seconds. Producers run for 20 seconds,
followed by 20 seconds of draining. Buffers have byte and per-class slot limits.
Blocked offers are counted; admitted data is never silently evicted. Miss rates
include blocked interactive offers, so refusing traffic cannot improve the score.

Sender history includes local ACK timing, flight, throughput, retries, queue
pressure and deadline information. The sender chooses rate and flight-window
residuals plus a message pointer among up to 64 queued candidates. Receiver
history includes arrivals, reordering, local backlog and feedback timing; its
categorical action chooses ACK batching/delay or follows the deterministic
feedback controller. The model uses 32 history steps, width 64, two transformer
layers and four heads by default. Histories persist across messages.

CRC, sequence ranges, retransmission, deduplication, and whole-message digest
verification remain deterministic. TRC2 wraps the existing packet in a protected
64-bit session identifier. The policy cannot manufacture successful delivery.
Both payload and feedback wire bytes consume simulated service. Reverse feedback
competes with reverse payload traffic. Optional shared airtime serializes both
directions, approximating a shared Wi-Fi medium above the MAC layer.

## Conditions and training

`joint-domain-v2` samples persistent capacity changes, correlated contention and
fades, loss bursts, outages, delay and asymmetry. Changes fall, hold, and recover;
independent per-step noise is not the only signal. There are 1–8 clients, a broad
payload/producer mix, and a 20% easy warm-up. Offered wire load includes data
headers, feedback, and both directions. Feasible, near-saturation and overloaded
buckets have weights 40/40/20%; cases are resampled rather than relabeled after
capacity clipping. An earlier sampler that undercounted wire load was retired;
its checkpoints are excluded from the final study.

These are synthetic service models, not calibrated robot Wi-Fi measurements.
Held-out `service-trace-v1` recordings can be supplied with
`joint-evaluate --trace FILE`; a recording must cover the whole 40-second session.
No trace loops or train/test slices are silently manufactured.

The shared reward emphasizes on-time interactive messages and penalizes misses,
while rewarding unique image/map progress and charging wire overhead and waiting.
Completion, corruption, useful bytes, deadline misses, blocked offers and wire
cost remain separate metrics: reward is a training signal, not the acceptance
criterion. Impossible-deadline lower bounds are reported without removing those
messages from the denominator.

## Win conditions and limits

The study calibrates 24 deterministic rate/scheduler/feedback combinations,
selects a reference on validation data subject to bulk-goodput eligibility,
screens checkpoints, then freezes selection before held-out tests. Initial
policies are eligible, so unsuccessful learning is visible. Test and OOD seeds
are disjoint from training and selection seeds. Results include sender/receiver
cross-play, one-step history, and separately trained frozen-role ablations.

A win requires at least 20% relative interactive miss reduction with a paired
seed bootstrap interval below zero; at least 95% of reference image/map goodput
and delivered volume; no lower completion ratio; at most 105% wire bytes per
delivered byte; and zero corrupt delivery. A zero-miss reference is a ceiling,
not an invitation to change the metric. Privileged clone lookahead is a headroom
diagnostic, not a proven optimal bound. General superiority is not established
by training curves, a single seed, or the smoke study.

## Real bytes

```bash
# Receiver terminal
trace-transport receive-file --output received.png --port 9001
# Sender terminal (use a remote hostname when appropriate)
trace-transport send-file --input image.png --host 127.0.0.1 --port 9001 --class image
```

The same commands carry text files or map deltas without decoding. Messages are
limited to 16 MiB and successful receive writes exact verified bytes atomically.
Real loopback PNG and map-buffer transfers were checked. By default this file adapter uses
the deterministic reliability/controller path. An optional experimental sender
actor is described below; joint learned session deployment
on Herman or Caliban is a subsequent integration step. A production system also
needs authentication, resource limits, cancellation/stale-image policy, fairness,
and measured device traces before claims about robot performance.

## Device engineering checks

The source-only probe runs on Python 3.10+ without installing a package on the
receiver. It copies only seven Python modules into a unique `/tmp` ZIP and retains
synthetic received files there for inspection:

```bash
uv run --locked python -m scripts.device_binary_trials \
  --ssh jetson1@herman --address HOST_IP --condition wifi-stationary \
  --report runs/devices/herman-binary.json
```

The probe sends three trials each of Unicode text, valid 256×256 RGB PNG bytes,
and a 512-KiB synthetic map. It also saves 80 ICMP observations over approximately
40 seconds. These raw RTT/loss observations are not `service-trace-v1` capacity
recordings. They must not be relabeled as measured available bandwidth.

Both Herman (Wi-Fi) and Caliban (Ethernet) completed all nine deterministic trials
and all nine experimental learned-sender trials. See
[device results](../artifacts/device-binary-v1-results.json). These are bounded
correctness checks, not randomized paired performance comparisons.

`send-file --checkpoint CHECKPOINT` now optionally runs the joint sender actor on
the local host with a deterministic remote receiver. The same option is available
in the probe. It constructs endpoint-local sender history using the simulation's
normalizations and applies rate/window residuals every 20 ms; message selection
is trivial for a single pending file. Producer rates and opposite-role traffic
are zero because these are single-file tests. They differ from the persistent
mixed training workload. The receiver actor is not deployed by this adapter.
A completed sender-only final checkpoint was used for the hardware checks,
independently of joint-policy validation selection. No claim of learned transport
superiority follows from exact recovery alone.

After a full study finishes, prepare reviewable artifacts with:

```bash
uv run --locked --extra learn python -m scripts.package_joint_study
```

Packaging requires the completed full study and recomputes the paired comparisons
from raw records. It exports actor weights without the critic/optimizers, stores
compressed training logs and configurations, and records source hashes. It never
publishes or merges a PR automatically.

The first full study used optimizer continuation after interruptions, with fresh
collector worlds explicitly recorded at each available resume point. The one-step
history run required several continuations and returned native exit code 139 in
a tracked run. This is not a bit-for-bit uninterrupted training comparison, and
a later worker diagnostic located a crash inside CPython 3.12.10's native JSON
container encoder during receipt serialization. The underlying native cause
remains unconfirmed. Wire encoding now uses the Python iterator with byte-for-byte
canonical JSON parity checks as a mitigation. Worker diagnostics now
include the seed, control step, and exit code. Remaining evaluation was continued
with cached completed modes, eight workers, and one numerical-library thread per
worker. Initial actors with exactly zero control and constant follow-baseline
heads can be evaluated through the mathematically identical baseline path; tests
compare that path with explicit actor inference, and any nonzero control excludes
the shortcut.

Held-out progress is saved per completed seed under `study/partial`, keyed by the
frozen selection and seed list. Resume reuses finished whole sessions and runs
only missing seeds; it never reports an incomplete mode as a full comparison.

Evaluation now schedules ready worlds asynchronously, preserving each world's
local history and control-step order instead of waiting for the slowest world at
every tick. Synchronous evaluation remains available through the Python API for
comparison; tests match complete records under nonzero deterministic controls.
As with other batched floating-point inference, cross-batch bit-for-bit equality
is not promised for arbitrary trained actors. The simulation's seeded event/RNG
semantics and the frozen selection/acceptance criteria remain unchanged.
