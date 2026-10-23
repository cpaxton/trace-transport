# Trace Transport

**A self-play testbed for history-based learned network protocols.**

Trace Transport trains small transformers to deliver exact UTF-8 text over
networks with unknown capacity, packet loss, changing latency, and competing
traffic. Both endpoints run our software, so packet formats, feedback, and control
rules can be designed together.

The working CPU reference includes a seeded packet simulator, fixed-rate and AIMD
controllers, learned sender policies, PPO training, paired evaluation, a local
browser test bench, and single-message UDP adapters. The default learned policy
controls the sender. Separate receiver-only and
[joint sender/receiver experiments](docs/joint-network-brain.md) explore learned
feedback and persistent sessions. Joint training uses CPU packet workers and
batched GPU learning; GPU-native packet simulation remains planned.

The bundled model completed 200/200 transfers in the reserved robust-v2 test;
fixed-rate also completed 200/200. These results do not establish general
superiority over transport baselines. See the [training and evaluation guide](docs/training.md)
for selection, seed sensitivity, and negative results.

## Quick start

On Linux or macOS, start with Git and curl installed:

```bash
curl -fsSL https://raw.githubusercontent.com/cpaxton/trace-transport/main/scripts/install.sh | bash
cd trace-transport
trace-transport console --open
```

The installer sets up uv, locked dependencies including PyTorch, and the
`trace-transport` command in `~/.local/bin`. If that directory is missing from
PATH, follow the shell setup instruction printed by the installer.
The browser runs at `http://127.0.0.1:8765`.
See the [usage guide](docs/usage.md#quick-start) for manual installation,
custom directories, and baseline-only setup. Python 3.10 or newer is supported.

```bash
# Deliver exact UTF-8 bytes through a lossy simulated network.
trace-transport simulate --text 'Hello, こんにちは 🌐' \
  --loss 0.1 --ack-loss 0.1 --seed 7 --show-text --output received.txt

# Compare fixed-rate and AIMD on ten paired trials.
trace-transport benchmark

# Compare the bundled learned sender with the baselines.
trace-transport benchmark --controllers fixed aimd learned --reward completion \
  --checkpoint artifacts/robust-v2-best.pt
```

Benchmark reports are saved to `runs/benchmark.json`; browser reports go to
`runs/console/`. Run any subcommand with `--help` for options.

Exact reconstruction is enforced by protocol code. Incomplete transfers are
reported explicitly. Messages must be valid UTF-8, are buffered in full, and are
limited to 16 MiB. The UDP listener handles one message from one peer; streaming
and durable recovery across crashes remain future work.

## Documentation

- [Documentation index](docs/README.md)
- [Usage](docs/usage.md): browser controls, CLI options, benchmark metrics, UDP, and Python API.
- [Training and evaluation](docs/training.md): PPO, reward profiles, history baselines, reproducible experiments, and results.
- [Protocol design and roadmap](docs/design.md): correctness, local observations, action spaces, and planned GPU/joint learning work.
- [Receiver experiments and device trials](docs/receiver-experiments.md): feedback timing, Wi-Fi models, results, and real-device setup.
- [Joint network experiment](docs/joint-network-brain.md): persistent sessions, training commands, and acceptance criteria.
- [Development](docs/development.md): implementation map, verification, source export, and paper build.
- [Paper working draft](paper/main.tex) and [bibliography](paper/references.bib).

## Development checks

```bash
uv run --locked --extra learn pytest -q
```

Learning tests require the `learn` extra. Real socket tests may skip in restricted
environments; in-memory adapter tests still exercise recovery behavior. Keep local
environments, credentials, and raw training runs out of Git. Publish changes on a
feature branch through a pull request targeting `main`. See [Contributing](CONTRIBUTING.md).
