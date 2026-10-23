# Documentation

[Project overview and quick start](../README.md)

| Guide | Contents |
| --- | --- |
| [Usage](usage.md) | Browser test bench, installation, simulation, benchmark metrics, UDP transfer, Python API |
| [Training and evaluation](training.md) | Learned sender setup, PPO, reward definitions, history baselines, experiment results, fixed scenario suite |
| [Protocol design and roadmap](design.md) | Research scope, related work, correctness, observations and actions, GPU plans, acceptance conditions |
| [Receiver experiments and device trials](receiver-experiments.md) | Feedback timing, receiver results, Wi-Fi models, device setup |
| [Protocol-v3 experiments](experiments/protocol-v3.md) | Feedback budgets and stronger generalization tests |
| [Joint network experiment](joint-network-brain.md) | Persistent sessions, mixed workloads, joint training and acceptance criteria |
| [Development](development.md) | Implementation map, test coverage, source export, paper build |

Start with the quick start, then use the usage guide for command examples.
Training reward and held-out performance are separate measurements; the training
guide preserves unfinished work and negative results alongside successful runs.
The design guide includes proposed capabilities beyond the current implementation.
