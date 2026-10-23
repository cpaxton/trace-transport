# Development and source export

[Documentation index](README.md) · [Project overview](../README.md)

## Implementation map and checks

| Module | Responsibility |
| --- | --- |
| `wire.py` | Versioned datagrams, CRC checks, bounded JSON metadata |
| `protocol.py` | Socket-independent sender, receiver, retries, reassembly, completion digest |
| `simulator.py` | Shared finite FIFO links, event timing, impairment, staggered starts |
| `environment.py` | Local observations, per-flow rewards, reset/step interface |
| `controllers.py` | Fixed-rate and rate-AIMD references |
| `feedback.py`, `receiver_learning.py` | Bounded ACK scheduling and a separate receiver policy |
| `receiver_curriculum.py`, `network_models.py` | Persistent degradation/recovery training domains and named Wi-Fi service models |
| `learning.py` | Causal masked attention, bounded-action PPO, checkpoint inference |
| `udp.py` | Real-time datagram adapters with asynchronous policy inference |
| `cli.py` | Simulation, paired evaluation, training, sender/listener commands |

Tests cover exact recovery, data and feedback loss, duplicate/reordered packets,
late ACKs, capacity changes, outages, receive limits, metadata wire costs,
observation isolation, reset semantics, checkpoint inference, PPO updates, and
UDP adapter recovery from a lost final ACK. Real loopback tests skip when the
execution sandbox forbids sockets; an in-memory datagram test still exercises
both adapter loops. Learning tests skip if PyTorch is unavailable.

The CPU simulator is the correctness reference. Receiver-only and persistent
joint-session experiments extend the default sender-only path; see the
[receiver experiments](receiver-experiments.md) and
[joint experiment guide](joint-network-brain.md). Recorded network traces,
streaming receive buffers, and GPU-native packet processing remain future work.
The [design and roadmap](design.md) describes the broader target.

## Source export and GitHub publication

```bash
# Build a clean Git bundle and source archive without modifying this workspace's .git.
.venv/bin/python scripts/export_repository.py

# From a terminal with GitHub access, initialize the empty destination repository.
.venv/bin/python scripts/export_repository.py --push
```

The exporter includes source, tests, browser assets, CI, paper sources, experiment
plans, the selected synthetic-data checkpoint, and compact result summaries.
It excludes environments, raw runs, other checkpoints, and local tool state.
`dist/publication/manifest.json` records each exported file's SHA-256 and the Git
commit. `--push` uses configured Git credentials and refuses a nonempty destination.

## Paper

The [working draft](../paper/main.tex) and [bibliography](../paper/references.bib)
document the system, prior work, evaluation plan, and preliminary fixed-scenario
results. These do not establish broad congestion-control superiority. With a LaTeX
installation, build from the paper directory:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```
