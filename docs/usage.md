# Using Trace Transport

[Documentation index](README.md) · [Project overview](../README.md)

## Quick start

On Linux or macOS, install into `./trace-transport` with Git and curl available:

```bash
curl -fsSL https://raw.githubusercontent.com/cpaxton/trace-transport/main/scripts/install.sh | bash
```

The [installer](../scripts/install.sh) installs uv if missing, clones the repository,
and installs locked dependencies including PyTorch. It adds a `trace-transport`
command in `~/.local/bin` pointing to the checkout's environment, so normal use is
`trace-transport console --open`. If that directory
isn't on PATH, it prints the shell configuration line to add. Set
`TRACE_TRANSPORT_BIN_DIR` to choose another command directory. It refuses to
overwrite an unrelated command. It reuses an existing checkout without pulling or
resetting your work. To choose a directory or skip PyTorch:

```bash
curl -fsSL https://raw.githubusercontent.com/cpaxton/trace-transport/main/scripts/install.sh \
  | bash -s -- --dir "$HOME/trace-transport" --baseline-only
```

For manual installation:

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (0.7.2 or
newer), then run these commands from a terminal:

```bash
git clone https://github.com/cpaxton/trace-transport.git
cd trace-transport
uv sync --locked --extra learn
source .venv/bin/activate
trace-transport console --open
```

For an existing manual checkout, run `uv sync --locked --extra learn` and
`source .venv/bin/activate`. The browser opens at
`http://127.0.0.1:8765`; if it does not, open that address manually. Choose a
scenario and controllers, then click **Run comparison**. Ten trials per controller
are selected by default. The bundled checkpoint enables learned-policy tests
without training first.

To add the plain command to PATH for an existing manual checkout, run
`bash scripts/install.sh --dir "$PWD"` once. The installer keeps the environment
in this checkout; keep that directory in place and use `uv sync --locked --extra
learn` there after dependency updates.

`uv` creates `.venv`, selects Python 3.12 from `.python-version`, and installs the
versions in `uv.lock`. Manual setup requires activating `.venv`; the curl installer provides a command
that selects its environment automatically.
Python 3.10 and newer remain supported; use `uv sync --python 3.10` to override
the default. PyTorch is optional: for baseline controllers only, run
`uv sync --locked` and activate `.venv`. Development tools,
including pytest, are in the default `dev` dependency group.

## Useful first commands

```bash
# Baselines: 10 paired trials with readable scores and a JSON report.
trace-transport benchmark

# Compare the shipped model with the baselines.
trace-transport benchmark \
  --controllers fixed aimd learned --reward completion \
  --checkpoint artifacts/robust-v2-best.pt

# Verify the complete implementation, including learned policies.
.venv/bin/python -m pytest -q
```

## Startup troubleshooting

- **`uv` is not found:** install it using the link above, reopen your terminal,
  and check `uv --version`.
- **No `.venv` exists:** run `uv sync --locked --extra learn`; `uv` creates it.
- **Learned policy is disabled:** include `--extra learn` to install PyTorch.
- **Tests load unrelated ROS or system plugins:** run
  `env -u PYTHONPATH .venv/bin/python -m pytest -q` from a clean shell.
- **Port 8765 is busy:** add `--port 8766` and open the printed address.
- **Browser does not open:** leave the console command running and open its URL.
- **Running outside the checkout:** change to the repository directory first;
  `console --root /path/to/trace-transport` controls model/report discovery.
- **A stale lockfile is reported:** use `uv lock` after intentionally changing
  dependencies, then commit `pyproject.toml` and `uv.lock` together. Routine
  startup and CI use `--locked` so they do not silently change resolved versions.

## Browser test bench

```bash
trace-transport console --open
```

Open `http://127.0.0.1:8765` if the browser does not open automatically. The local
dashboard runs real simulation batches with:

- Clean, lossy, changing-capacity, and stress presets plus editable network controls.
- Ten paired trials by default, configurable payloads, arbitrary inline UTF-8 text, and concurrent senders.
- Fixed-rate, AIMD, and a selectable learned checkpoint from `artifacts/` or `checkpoints/`.
- Live delivery bars, RTT/rate telemetry, completion and integrity counters, and controller score tables.
- Per-trial delivery/rate charts, JSON/CSV downloads, and cancellation that retains completed trials.

The simulator runs faster than wall time. Plots explicitly use simulated seconds;
the batch header separately shows elapsed wall time. CRC failures and missing
feedback are real simulated events. Observer telemetry is for inspection only
and does not give the actor access to hidden network state. The common selected
reward is used for every controller. Reports are also saved to `runs/console/`.

Use `--port 8766` for another port, or `--root /path/to/project` to locate that
project's models and report directory. The server binds to loopback, uses bundled
HTML/CSS/JS, and has no frontend package installation or external service. NumPy
is sufficient for baseline trials; install the `learn` extra for checkpoints.

## Command-line setup

The simulator needs NumPy; PyTorch is optional unless training or using a learned
checkpoint. This lightweight setup runs only the baseline controllers:

```bash
uv sync --locked
source .venv/bin/activate

# Send arbitrary inline text and save the exact UTF-8 bytes.
trace-transport simulate --text 'Hello, こんにちは 🌐' \
  --loss 0.1 --ack-loss 0.1 --seed 7 --show-text --output received.txt

# Competing senders, an outage, and changing capacity.
trace-transport simulate --text 'Competing traffic' --flows 4 \
  --outage 0.1 0.4 --capacity-change 0.5 2.0 --report runs/scenario.json

# For meaningful congestion behavior, supply a larger UTF-8 file of your own.
trace-transport evaluate --input message.txt --flows 4 --seeds 5 \
  --controllers fixed aimd --report runs/baselines.json

.venv/bin/python -m pytest -q
```

## Ten-trial benchmark and RL score

```bash
# Ten paired trials for fixed-rate and AIMD: two 32 KiB transfers per trial.
trace-transport benchmark

# Include a trained checkpoint, with the same workload and seeds for every controller.
trace-transport benchmark --controllers fixed aimd transformer \
  --checkpoint checkpoints/smoke.pt --trials 10 --seed 100

# More severe impairments or your own exact UTF-8 input.
trace-transport benchmark --input message.txt --loss 0.1 --ack-loss 0.1 \
  --corruption 0.02 --ack-corruption 0.02 --flows 4 --trials 10
```

`benchmark` is an alias for `evaluate`. Both now default to ten seeds, a readable
summary table, and a detailed JSON report at `runs/benchmark.json`. Use
`--format json` for JSON on stdout and `--report PATH` for another destination.
With no `--text` or `--input`, the workload is exactly 32,768 bytes of valid UTF-8;
`--message-bytes` changes its size. Explicitly supplied text, including an empty
string, is preserved. Ten trials per controller with two flows means twenty
transfer attempts per controller, not ten independent samples per flow.

Default benchmarks apply 2% random data/ACK loss and 1% data/ACK bit corruption,
plus finite-queue drops. `simulate` keeps corruption disabled by default; both
commands accept `--corruption` and `--ack-corruption`. A corrupted packet has an
actual flipped wire bit and is rejected by the existing CRC check. The policy
observes missing feedback and eventual timeouts, not a simulator corruption flag.

The report separates:

- **Completion %:** exact reconstructed bytes with sender confirmation before the deadline.
- **Exact RX %:** exact receiver reconstruction, even if final feedback never reached the sender.
- **Drops:** random, queue-overflow, and outage losses across both directions, divided by attempted packets.
- **Corruption:** packets with injected bit flips and packets actually rejected by CRC. Injected packets can still be in flight or encounter an outage at the deadline.
- **Wrong payloads:** reconstructed messages that differ from the source. Missing messages are incomplete, not silently counted as correct.
- **Retry %:** retransmitted application bytes divided by all application bytes transmitted.
- **Latency/goodput:** successful completion-time percentiles, pooled sampled RTT percentiles, and mean network goodput over trials. Completion latency excludes failures; completion rate and RL score retain them.
- **RL score:** the mean discounted sum of actual `TransportEnv.step()` rewards per transfer, including unfinished transfers. The common selected profile declares the reward and gamma (0.99 for legacy, 0.999 for completion). JSON also records the undiscounted return and both scores for each transfer.

Completion percentage is a key quality metric, but it is not the whole RL
objective: two policies can both complete 100% while having different queueing
delay and wire overhead. Compare returns with identical message sizes, seeds,
network settings, deadlines, and reward definitions; a larger message earns more
progress reward. The JSON stores each trial's configuration, per-flow return,
integrity status, packet counters, and checkpoint SHA-256. These quick batches
support debugging; broader held-out evaluation is needed for performance claims.

`--input` and `--output` preserve exact file bytes, including CRLF and trailing
newlines. Inputs must be valid UTF-8. The current implementation accepts messages
up to 16 MiB and buffers a complete message; streaming larger messages is future
work. `simulate` and `send` exit with status 2 on an incomplete transfer and do
not write a partial output as a successful message. `evaluate` records timeouts
as outcomes and exits normally. Rates use bits/s in network settings and bytes/s
inside the sender. Run any subcommand with `--help` for options.

## Send text between processes or hosts

In one terminal:

```bash
trace-transport listen --host 127.0.0.1 --port 9000 --output received.txt \
  --metadata '{"experiment":"custom-feedback"}'
```

In another:

```bash
trace-transport send --host 127.0.0.1 --port 9000 --text 'Across UDP 🌐'
```

Use a reachable interface/address instead of loopback for separate hosts. The
same sender supports `--controller transformer --checkpoint ...` and `--input`.
The listener serves one message from one peer, saves it atomically, and lingers
to re-ACK duplicates if final feedback was lost. Keep `--linger` long enough for
the sender's retries. SHA-256 completion metadata verifies the full reconstructed
payload. The sender receives confirmation, not an echoed copy of the text; the
receiver writes the actual output. Policy inference runs in a worker thread
while the UDP sender continues under the last action.

## Python interface

```python
from trace_transport import NetworkConfig, transfer_text

result = transfer_text(
    "Hello, network!\nこんにちは 🌐",
    config=NetworkConfig(loss=0.1, ack_loss=0.1, deadline=10),
    controller="aimd",
    seed=7,
)
assert result.completed
assert result.received_text.encode("utf-8") == result.sent_text.encode("utf-8")
```
