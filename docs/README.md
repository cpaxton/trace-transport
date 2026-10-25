# Documentation

[Project overview and quick start](../README.md)

| Guide | Contents |
| --- | --- |
| [Experiment register](experiments/register.md) | Dated experiment list, status, results, limitations and artifact links |
| [Usage](usage.md) | Browser test bench, installation, simulation, benchmark metrics, UDP transfer, Python API |
| [Training and evaluation](training.md) | Learned sender setup, PPO, reward definitions, history baselines, experiment results, fixed scenario suite |
| [Protocol design and roadmap](design.md) | Research scope, related work, correctness, observations and actions, GPU plans, acceptance conditions |
| [Receiver experiments and device trials](receiver-experiments.md) | Feedback timing, receiver results, Wi-Fi models, device setup |
| [Protocol-v3 experiments](experiments/protocol-v3.md) | Feedback budgets and stronger generalization tests |
| [Joint-v1 results](experiments/joint-v1.md) | Completed held-out/OOD comparisons, negative learning result, checkpoints and limitations |
| [Joint network experiment](joint-network-brain.md) | Persistent sessions, mixed workloads, joint training and acceptance criteria |
| [Maximum throughput plan](goodput-v2-plan.md) | Backlogged bulk traffic, verified goodput, and deadline-constrained optimization |
| [Learning opportunities](research/learning-opportunities.md) | Literature, history-dependent challenges, stronger baselines and fairness |
| [Goodput-v2 pilots](experiments/goodput-v2.md) | Calibration, bounded CPU pilot, scoring and resource isolation |
| [V3 distribution and retraining](experiments/distribution-v3.md) | Coverage matrix, realistic mechanisms and gaps, control-sensitivity audit and receiver diagnosis |
| [Caliban v4 parallel training](experiments/caliban-v4.md) | Deployment coverage, review corrections, remote limits, scaling and recovery |
| [V5 learning-signal diagnosis](experiments/learning-signal-v5.md) | Per-family failures, infeasible completion guard, progress shaping and controlled pilot |
| [Development](development.md) | Implementation map, test coverage, source export, paper build |

Command examples assume the curl installer has added `trace-transport` to PATH.
Run them from the checkout so bundled checkpoints and scripts resolve. Python
scripts and test commands use `.venv/bin/python` to select the installed environment.
Manual installation is covered in the [usage guide](usage.md#quick-start).

Start with the quick start, then use the usage guide for command examples.
Training reward and held-out performance are separate measurements; the training
guide preserves unfinished work and negative results alongside successful runs.
The design guide includes proposed capabilities beyond the current implementation.
