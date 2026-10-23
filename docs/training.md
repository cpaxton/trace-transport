# Training and evaluation

[Documentation index](README.md) · [Project overview](../README.md)

## Train and use the transformer

The repository includes the selected CPU-loadable checkpoint at
`artifacts/robust-v2-best.pt`, its SHA-256, and a compact diagnostic report at
`artifacts/robust-v2-results.json`. After installing the `learn` extra, try it
without training:

```bash
uv run --locked --extra learn trace-transport benchmark --controllers fixed aimd learned --reward completion \
  --checkpoint artifacts/robust-v2-best.pt
```

```bash
uv sync --locked --extra learn

# A software smoke run, not a claim of a good congestion controller.
uv run --locked --extra learn trace-transport train --updates 5 --envs 4 --flows 2 --rollout-steps 32 \
  --epochs 2 --history 16 --output checkpoints/smoke.pt

uv run --locked --extra learn trace-transport simulate --text 'A learned sender 🌐' \
  --controller transformer --checkpoint checkpoints/smoke.pt --show-text

uv run --locked --extra learn trace-transport evaluate --input message.txt --flows 4 --seeds 5 \
  --controllers fixed aimd transformer --checkpoint checkpoints/smoke.pt \
  --report runs/learned-comparison.json
```

Training randomizes capacity, RTT, jitter, random loss, queue size, and message
size for each episode. It uses a shared sender policy across independent CPU
networks and produces a schema-checked checkpoint plus JSONL metrics.
`--actions 2` enables window control as well as pacing. `--device cuda` moves the
model and PPO updates to a usable CUDA device; it does **not** move this CPU
simulator to the GPU. Start with the CPU for these small runs. A useful trained
controller requires longer training and held-out evaluation.

## Robust training and history baselines

`--profile legacy` preserves the original fixed-flow, 0--5% loss experiment.
`--profile robust` samples clean, lossy, changing, and stress episodes with one
through `--flows` senders, up to 15% random loss, up to 2% corruption, randomized
feedback capacity, smaller queues, outages, and capacity changes. The sampler is
independent of the frozen evaluation manifests. Inactive batch slots are masked
out of PPO rather than treated as synthetic agents.

`--reward legacy` retains `transport-v1` and gamma=0.99. `--reward completion`
selects `transport-v2` and gamma=0.999. At 20 ms per step these have roughly
2-second and 20-second horizons: a completion bonus after 20 seconds retains
about 0.0043% or 37% of its original weight, respectively.

The v2 per-transfer reward is:

```text
newly_ACKed_unique_bytes / max(message_bytes, 1)
+ 2 * just_completed - 2 * timed_out
- 0.05 * elapsed_active_seconds
- 0.01 * (RTT_inflation / 0.1 seconds) * elapsed_active_seconds
- 0.1 * overhead_wire_bytes / max(message_bytes, 512)
```

This adds a cost for waiting, normalizes progress by message size, and strengthens
completion and failure signals. Correctness remains enforced by protocol code.
Reports declare the full scoring specification and reject mixed reward profiles.
Old defaults remain available for reproducibility: explicitly pass
`--profile robust --reward completion` for new training and `--reward completion`
for a comparison in which every controller receives the same v2 score.

Three policies consume the same masked history window. The MLP flattens it; the
GRU processes its valid suffix, recomputing state from that window on every call.
These settings approximately match parameter counts, not computational cost:

| Model | History | Width | Layers | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Transformer | 16 | 64 | 2 | 102,723 |
| MLP | 16 | 152 | 3 | 103,211 |
| GRU | 16 | 104 | 2 | 105,875 |

```bash
# Substitute architecture, width, and layers from the table for each baseline.
uv run --locked --extra learn trace-transport train --architecture transformer --width 64 --layers 2 \
  --history 16 --profile robust --reward completion --flows 4 --envs 8 \
  --min-message-bytes 16384 --max-message-bytes 98304 --episode-seconds 20 \
  --updates 400 --rollout-steps 64 --epochs 2 --seed 42 \
  --output checkpoints/robust-v2-transformer-s42.pt

# Checkpoint metadata selects transformer/MLP/GRU automatically.
uv run --locked --extra learn trace-transport simulate --controller learned \
  --checkpoint checkpoints/robust-v2-transformer-s42.pt \
  --reward completion --text 'A learned protocol 🌐'

uv run --locked --extra learn trace-transport suite --reward completion --trials 10 --seed 1000 \
  --policy candidate=checkpoints/robust-v2-transformer-s42.pt
```

The robust-v2 experiment uses training seeds 42 and 43 per architecture, plus a
robust transformer with the legacy reward as an ablation. Validation uses seeds
1000--1009; final evaluation reserves seeds 2000--2019. Selection first maximizes
completed validation transfers, then mean per-scenario v2 score. The plan is
recorded in the [declared experiment plan](../experiments/robust-v2.json), with a copy
at `runs/robust-v2/plan.json` from the original run. Report actual transition
counts and wall time: different policies encounter different episode trajectories
even with the same update budget and sampler seed.

The completed comparison trained seven policies for 400 updates each. Six used
v2 (two seeds per architecture); one transformer used the old reward. The
validation-only selection chose `transformer-s43`. On reserved seeds 2000--2019:

| Scenario | Fixed-rate | AIMD | Previous policy | Selected transformer |
| --- | ---: | ---: | ---: | ---: |
| Clean | 40/40 | 40/40 | 40/40 | 40/40 |
| Lossy/corrupt | 40/40 | 40/40 | 40/40 | 40/40 |
| Changing capacity/outage | 40/40 | 40/40 | 38/40 | 40/40 |
| Four-sender stress | 80/80 | 4/80 | 26/80 | 80/80 |

The selected checkpoint completed 200/200 transfers with no wrong payloads.
Fixed-rate also completed all of them; this is evidence of progress over the
previous learned policy, not general superiority over transport baselines.
Training was seed-sensitive: transformer-s42 completed 0/100 validation transfers,
whereas transformer-s43, both MLPs, both GRUs, and the old-reward ablation each
completed 100/100. The ablation therefore does not establish that v2 alone caused
the improvement. Broader training and validation selection are part of the result.

The MLP trained in about 64 seconds per seed, versus approximately 198 seconds
for the transformer and 210 for the GRU on this CPU experiment. A single-thread
model-only timing probe found median batch-one inference of roughly 0.052 ms,
0.155 ms, and 0.276 ms for MLP, transformer, and GRU respectively. These are local
measurements, not portable throughput guarantees. The cheaper models remain
useful baselines; more training seeds are needed to assess stability.

The chosen bytes are also copied to `checkpoints/robust-v2-best.pt`. Review
`runs/robust-v2/validation.json`, `selection.json`, `final-test.json`, and
`model-costs.json` for the individual results and provenance. To reproduce the
selection/final-evaluation procedure after the declared checkpoints are trained:

```bash
# Reuse completed checkpoints to reproduce validation and final evaluation.
uv run --locked --extra learn python scripts/compare_learning.py --plan experiments/robust-v2.json

# Train missing candidates first, then run the declared comparisons.
uv run --locked --extra learn python scripts/compare_learning.py --plan experiments/robust-v2.json --train

# Use the selected model for exact text transfer.
uv run --locked --extra learn trace-transport simulate --controller learned --reward completion \
  --checkpoint checkpoints/robust-v2-best.pt --text 'Reliable text 🌐' --show-text
```

The comparison script checks checkpoint configurations against the declared plan,
rejects overlapping validation/test seed ranges, and writes the selection before
running the final test. It copies the chosen checkpoint to `--best-checkpoint`
(default `checkpoints/robust-v2-best.pt`). On a fresh checkout the optional previous
checkpoint may be absent; new candidates and baselines are still compared.
Reserve new test seeds if using these results to tune the next experiment.

For PPO stability experiments, `train` also supports `--target-kl 0.01` to stop
further optimization within an update when its estimated KL exceeds 1.5 times
the target. `--checkpoint-every 100` retains intermediate `.u0100.pt` snapshots
rather than only the final weights. JSONL logs include estimated KL, clip
fraction, optimizer-step count, early-stop status, and latent exploration scale.
Both options default off, preserving old experiment plans. Old configuration
records are normalized before comparison so newly added optional defaults do
not invalidate existing checkpoints.

The first KL-limited follow-up did not resolve seed sensitivity. With a 0.01
target, seed 42 stopped optimization early in 108/400 updates, yet its final
checkpoint still completed 0/50 validation transfers. Its update-100/200
snapshots completed 20/50, exposing the later loss of useful behavior. Seed 43
and all of its snapshots completed 50/50. The earlier selected checkpoint had
the better mean score on this diagnostic set and remains the bundled default.
See `artifacts/stability-v1-results.json`; this is a diagnostic, not a successful
stability fix.

## Compare progress on a fixed scenario suite

`suite` runs the versioned [reference-v1 scenarios](../src/trace_transport/scenarios/reference-v1.json):
clean, lossy/corrupt, changing capacity with an outage, and a four-sender stress
case. Definitions are fixed; each report embeds their contents and SHA-256, along
with the configuration, seed, and checkpoint digest for each trial. Compare
checkpoints on identical scenarios instead of changing the benchmark between runs.

```bash
# A longer sender-only training run, using the same model size as the smoke run.
uv run --locked --extra learn trace-transport train --updates 300 --envs 8 --flows 2 --rollout-steps 64 \
  --epochs 2 --history 16 --seed 42 --output checkpoints/longer-seed42.pt

# Evaluate baselines and both checkpoints using a separate range of evaluation seeds.
uv run --locked --extra learn trace-transport suite --trials 10 --seed 1000 \
  --policy smoke=checkpoints/smoke.pt \
  --policy trained=checkpoints/longer-seed42.pt \
  --report runs/longer-seed42-suite.json

# A quicker check or a larger evaluation set.
uv run --locked trace-transport suite --cases clean lossy --trials 2
uv run --locked --extra learn trace-transport suite --trials 50 --seed 2000 \
  --policy candidate=checkpoints/longer-seed42.pt
```

Repeat `--policy LABEL=CHECKPOINT` for named models. The table shows completion,
score, successful completion time, retries, and wrong payloads for each scenario.
`--manifest PATH` accepts another version-1 scenario file; `--format json` prints
the detailed report to stdout. `--baselines` with no values selects checkpoint
policies alone. Reports keep scenario scores separate because their message
sizes and difficulty differ. Training completion rate is diagnostic; evaluate
on separate settings and seeds, and reserve a fresh final test set after tuning.

The first 300-update run (`longer-seed42.pt`) processed 264,198 active transitions
in about 115 seconds on CPU. On reference-v1 seeds 1000--1009, transfer completion
was:

| Scenario | Fixed-rate | AIMD | 5-update smoke | 300-update trained |
| --- | ---: | ---: | ---: | ---: |
| Clean | 100% | 100% | 100% | 100% |
| Lossy/corrupt | 100% | 100% | 100% | 100% |
| Changing capacity/outage | 100% | 100% | 100% | 95% |
| Four-sender stress | 100% | 7.5% | 0% | 40% |

These are diagnostic results from one longer training seed, not evidence of an
overall win. The two learned policies have the same small architecture; their
training seeds, batch settings, and budgets differ. Completion improved over the
smoke policy on stress but remained below fixed-rate. The current training
distribution has two senders, 0--5% random loss, and no outages or capacity
changes, whereas stress has four senders, 10% loss, and an outage. The trained
policy's recent training completion rate was 98%, illustrating why a separate
evaluation suite matters. Full results and logs are generated at
`runs/longer-seed42-suite.json` and `checkpoints/longer-seed42.jsonl`.

Next experiments should broaden the training distribution to cover these failure
modes, keep this suite fixed for validation, and use fresh seeds for final tests.
Then compare the transformer with cheaper history models before expanding the
action space or investing in GPU simulation.

The five-update smoke checkpoint has not learned robust behavior. In a stress
check with four 81 KB transfers, 10% data/ACK loss, an outage, and changing
capacity, fixed-rate completed while rate-AIMD and the smoke transformer timed
out at 20 seconds. This is a single diagnostic case, not a benchmark conclusion.
The simulator reports incomplete transfers rather than accepting partial text.
Reproduce it after training with:

```bash
uv run --locked python - <<'PY'
from pathlib import Path
Path("runs").mkdir(exist_ok=True)
Path("runs/stress.txt").write_bytes(
    ("A history-based transport test: こんにちは 🌐\r\n" * 1500).encode("utf-8")
)
PY
uv run --locked --extra learn trace-transport evaluate --input runs/stress.txt --flows 4 --seed 17 --seeds 1 \
  --loss 0.1 --ack-loss 0.1 --jitter-ms 20 --queue-bytes 8192 \
  --outage 0.2 0.5 --capacity-change 1 2 --deadline 20 \
  --controllers fixed aimd transformer --checkpoint checkpoints/smoke.pt \
  --report runs/stress-evaluation.json
```
