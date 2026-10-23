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
Real loopback PNG and map-buffer transfers were checked. This file adapter uses
the deterministic reliability/controller path; joint learned session deployment
on Herman or Caliban is a subsequent integration step. A production system also
needs authentication, resource limits, cancellation/stale-image policy, fairness,
and measured device traces before claims about robot performance.
