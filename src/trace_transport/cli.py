"""Command-line text transfer, paired evaluation, training, and UDP adapters."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from .benchmark import benchmark_text, format_benchmark, summarize_runs
from .controllers import make_controller
from .environment import TransportEnv
from .simulator import NetworkConfig
from .rewards import reward_spec
from .feedback import FEEDBACK_PROFILES,profile_config


def _network_arguments(parser):
    parser.add_argument("--feedback-profile", choices=list(FEEDBACK_PROFILES), default="legacy",
                        help="legacy ACKs, budgeted receipt summaries, or sparse receipt summaries")
    parser.add_argument("--bandwidth-mbps", type=float, default=1.0)
    parser.add_argument("--rtt-ms", type=float, default=50.0)
    parser.add_argument("--jitter-ms", type=float, default=5.0)
    parser.add_argument("--loss", type=float, default=.02)
    parser.add_argument("--ack-loss", type=float, default=.02)
    parser.add_argument("--corruption", type=float, default=0.0, help="probability of a wire bit flip in a data packet")
    parser.add_argument("--ack-corruption", type=float, default=0.0, help="probability of a wire bit flip in an ACK")
    parser.add_argument("--duplicate", type=float, default=0.0)
    parser.add_argument("--queue-bytes", type=int, default=16_384)
    parser.add_argument("--payload-size", type=int, default=512)
    parser.add_argument("--rate-kbps", type=float, default=1024.0,
                        help="initial pacing rate in kilobits/second (1000 bits/kbit)")
    parser.add_argument("--deadline", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outage", nargs=2, type=float, metavar=("START", "END"), action="append", default=[])
    parser.add_argument("--capacity-change", nargs=2, type=float, metavar=("SECONDS", "MBPS"), action="append", default=[])


def _config(args):
    return NetworkConfig(
        bandwidth_bps=args.bandwidth_mbps * 1e6, base_rtt=args.rtt_ms / 1000,
        jitter=args.jitter_ms / 1000, loss=args.loss, ack_loss=args.ack_loss,
        corruption=args.corruption, ack_corruption=args.ack_corruption,
        duplicate=args.duplicate, queue_bytes=args.queue_bytes, payload_size=args.payload_size,
        initial_rate=args.rate_kbps * 1000 / 8, deadline=args.deadline,
        outages=tuple(tuple(x) for x in args.outage),
        capacity_changes=tuple((t, rate * 1e6) for t, rate in args.capacity_change),
    )


def _text_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", help="inline Unicode text")
    group.add_argument("--input", type=Path, help="UTF-8 file (read as exact bytes, without newline conversion)")


def _read_text(args):
    if args.input is not None:
        return args.input.read_bytes().decode("utf-8")
    return args.text if args.text is not None else "Hello, network!\nこんにちは 🌐"


def run_simulation(texts, config, controller, *, seed=0, checkpoint=None, device="cpu", start_times=None,
                   include_rtt_samples=False, reward_profile="legacy", progress=None, feedback_profile="legacy"):
    config=profile_config(config,feedback_profile)
    policy = make_controller(controller, len(texts), checkpoint=checkpoint, device=device)
    env = TransportEnv(texts, config, action_dimensions=policy.action_dimensions, start_times=start_times,
                       reward_profile=reward_profile,receiver_feedback=feedback_profile!="legacy",
                       receipt_feedback=feedback_profile!="legacy")
    obs, _ = env.reset(seed=seed)
    if feedback_profile != 'legacy':
        for scheduler in env.simulator.feedback_schedulers:
            scheduler.batch_size,scheduler.delay=8,FEEDBACK_PROFILES[feedback_profile]
    policy.reset(obs)
    returns = [0.0] * len(texts)
    discounted_returns = [0.0] * len(texts)
    discount = 1.0
    while not env.simulator.finished:
        obs, rewards, _, _, _ = env.step(policy.act(obs))
        returns = [total + float(reward) for total, reward in zip(returns, rewards)]
        discounted_returns = [total + discount * float(reward)
                              for total, reward in zip(discounted_returns, rewards)]
        discount *= env.reward_spec.discount_gamma
        if progress is not None:
            sim = env.simulator
            progress({"time": sim.now, "deadline": config.deadline,
                      "flows": [{"payload_bytes": len(sender.payload),
                                 "acknowledged_bytes": sender.acknowledged_bytes,
                                 "received_bytes": receiver.received_bytes,
                                 "complete": sender.complete, "receiver_complete": receiver.complete,
                                 "rate_bytes_per_second": sender.rate, "rtt_seconds": sender.srtt,
                                 "retransmitted_bytes": sender.retransmitted_bytes,
                                 "rl_return": returns[index]} for index, (sender, receiver)
                                in enumerate(zip(sim.senders, sim.receivers))],
                      "network": {"forward": asdict(sim.forward.counters),
                                  "feedback": asdict(sim.feedback.counters)}})
    results = env.simulator.results()
    flows = []
    for result, total, discounted in zip(results, returns, discounted_returns):
        flows.append({
            **result.summary(), "rl_return": total, "discounted_rl_return": discounted,
            "integrity_ok": (result.received_text.encode("utf-8") == result.sent_text.encode("utf-8")
                             if result.received_text is not None else None),
        })
    report = {
        "controller": controller, "seed": seed, "config": asdict(config),
        "feedback_profile": feedback_profile,
        "checkpoint": str(checkpoint) if controller in ("transformer", "learned") else None,
        "checkpoint_sha256": (hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
                              if controller in ("transformer", "learned") else None),
        "flows": flows, "elapsed_seconds": env.simulator.now,
        "reward": env.reward_spec.metadata(),
        "architecture": policy.model.config.architecture if hasattr(policy, "model") else None,
        "network": {"forward": asdict(env.simulator.forward.counters),
                    "feedback": asdict(env.simulator.feedback.counters)},
        "processed_events": env.simulator.processed_events,
    }
    if include_rtt_samples:
        report["rtt_samples_seconds"] = [sample for result in results for sample in result.rtt_samples]
    return results, report


def build_parser():
    parser = argparse.ArgumentParser(prog="trace-transport", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("joint-train", add_help=False, help="joint sender/receiver MAPPO training")
    commands.add_parser("joint-study", add_help=False, help="calibrate, validate and test paired joint policies")
    commands.add_parser("joint-status", add_help=False, help="read live joint-training and study status")
    commands.add_parser("joint-evaluate", add_help=False, help="evaluate a paired checkpoint or deterministic session policy")
    commands.add_parser("joint-headroom", add_help=False, help="privileged snapshot look-ahead diagnostic")
    commands.add_parser("joint-simulate", add_help=False, help="persistent binary robot workload")
    commands.add_parser("send-file", add_help=False, help="send an opaque binary file over TRC2")
    commands.add_parser("receive-file", add_help=False, help="receive a binary image/map/file over TRC2")
    commands.add_parser("receiver-categorical", add_help=False, help="train a baseline-initialized categorical receiver")
    commands.add_parser("sender-experiment", add_help=False, help="train residual pacing/window control")
    commands.add_parser("receiver-experiment", add_help=False,
                        help="train and evaluate a receiver-only policy (requires learn extra)")
    console = commands.add_parser("console", help="open the local browser test bench with live charts and batch controls")
    console.add_argument("--port", type=int, default=8765)
    console.add_argument("--root", type=Path, default=Path.cwd(), help="project directory containing artifacts/checkpoints")
    console.add_argument("--open", action="store_true", help="open the local URL in your browser")
    simulate = commands.add_parser("simulate", help="transfer text through the seeded packet simulator")
    _text_arguments(simulate)
    _network_arguments(simulate)
    simulate.add_argument("--controller", choices=["fixed", "aimd", "transformer", "learned"], default="aimd")
    simulate.add_argument("--checkpoint")
    simulate.add_argument("--device", default="cpu")
    simulate.add_argument("--reward", choices=["legacy", "completion"], default="legacy")
    simulate.add_argument("--flows", type=int, default=1, help="concurrent copies sharing the link")
    simulate.add_argument("--stagger-seconds", type=float, default=0)
    simulate.add_argument("--output", type=Path, help="save exactly reconstructed text from flow 0")
    simulate.add_argument("--report", type=Path)
    simulate.add_argument("--show-text", action="store_true")

    evaluate = commands.add_parser("evaluate", aliases=["benchmark"],
                                   help="run a readable 10-trial benchmark with actual RL scores")
    _text_arguments(evaluate)
    _network_arguments(evaluate)
    evaluate.add_argument("--controllers", nargs="+", choices=["fixed", "aimd", "transformer", "learned"], default=["fixed", "aimd"])
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--reward", choices=["legacy", "completion"], default="legacy")
    evaluate.add_argument("--flows", type=int, default=2)
    evaluate.add_argument("--seeds", "--trials", dest="seeds", type=int, default=10,
                          help="paired trials per controller (default: 10)")
    evaluate.add_argument("--message-bytes", type=int, default=32_768,
                          help="generated UTF-8 payload bytes when no --text/--input is given")
    evaluate.add_argument("--format", choices=["table", "json"], default="table")
    evaluate.add_argument("--report", type=Path, default=Path("runs/benchmark.json"))
    evaluate.set_defaults(corruption=.01, ack_corruption=.01)

    suite = commands.add_parser("suite", help="compare checkpoints on a versioned set of ordinary and hard networks")
    suite.add_argument("--manifest", type=Path, help="custom scenario manifest; default: bundled reference-v1")
    suite.add_argument("--cases", nargs="+", help="case names to run; default: all")
    suite.add_argument("--trials", type=int, default=10)
    suite.add_argument("--seed", type=int, default=1000)
    suite.add_argument("--baselines", nargs="*", choices=["fixed", "aimd"], default=["fixed", "aimd"])
    suite.add_argument("--policy", action="append", default=[], metavar="LABEL=CHECKPOINT",
                       help="named checkpoint; repeat to compare smoke and trained policies")
    suite.add_argument("--device", default="cpu")
    suite.add_argument("--reward", choices=["legacy", "completion"], default="legacy")
    suite.add_argument("--format", choices=["table", "json"], default="table")
    suite.add_argument("--report", type=Path, default=Path("runs/suite.json"))

    training = commands.add_parser("train", help="train a transformer, history MLP, or GRU using PPO")
    training.add_argument("--output", default="checkpoints/transformer.pt")
    training.add_argument("--updates", type=int, default=20)
    training.add_argument("--envs", type=int, default=4)
    training.add_argument("--flows", type=int, default=2)
    training.add_argument("--rollout-steps", type=int, default=64)
    training.add_argument("--epochs", type=int, default=4)
    training.add_argument("--minibatch", type=int, default=128)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--target-kl", type=float, help="stop a PPO update when estimated KL exceeds 1.5 times this value")
    training.add_argument("--checkpoint-every", type=int, default=0, help="save intermediate checkpoints every N updates")
    training.add_argument("--history", type=int, default=64)
    training.add_argument("--width", type=int, default=64)
    training.add_argument("--heads", type=int, default=4)
    training.add_argument("--layers", type=int, default=2)
    training.add_argument("--actions", type=int, choices=[1, 2], default=1)
    training.add_argument("--architecture", choices=["transformer", "mlp", "gru"], default="transformer")
    training.add_argument("--profile", choices=["legacy", "robust"], default="legacy")
    training.add_argument("--reward", choices=["legacy", "completion"], default="legacy")
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--device", default="cpu")
    training.add_argument("--threads", type=int, default=1)
    training.add_argument("--min-message-bytes", type=int, default=16_384)
    training.add_argument("--max-message-bytes", type=int, default=65_536)
    training.add_argument("--episode-seconds", type=float, default=4.0)

    listen = commands.add_parser("listen", help="receive one text message over UDP and linger to repeat ACKs")
    listen.add_argument("--host", default="127.0.0.1")
    listen.add_argument("--port", type=int, default=9000)
    listen.add_argument("--output", type=Path, required=True)
    listen.add_argument("--timeout", type=float, default=30)
    listen.add_argument("--linger", type=float, default=10,
                        help="seconds to repeat feedback for duplicate packets after receiver completion")
    listen.add_argument("--metadata", default="{}", help="custom JSON feedback metadata within the wire budget")
    listen.add_argument("--receipt-feedback", action="store_true", help="use cumulative/selective receipt summaries (requires updated sender)")
    listen.add_argument("--ready-json", action="store_true", help="print a JSON readiness line after binding")

    send = commands.add_parser("send", help="send arbitrary text to a UDP listener")
    _text_arguments(send)
    send.add_argument("--host", default="127.0.0.1")
    send.add_argument("--port", type=int, default=9000)
    send.add_argument("--deadline", type=float, default=20)
    send.add_argument("--controller", choices=["fixed", "aimd", "transformer", "learned"], default="aimd")
    send.add_argument("--checkpoint")
    send.add_argument("--device", default="cpu")
    return parser


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments and arguments[0] == "receiver-experiment":
        from .receiver_experiment import main as receiver_main
        return receiver_main(arguments[1:])
    if arguments and arguments[0] in ("receiver-categorical", "sender-experiment"):
        if arguments[0] == "receiver-categorical":
            from .receiver_v3 import main as research_main
        else:
            from .sender_experiment import main as research_main
        return research_main(arguments[1:])
    if arguments and arguments[0] in ("joint-train","joint-simulate","send-file","receive-file","joint-study","joint-headroom","joint-status","joint-evaluate"):
        from .joint_cli import main as joint_main
        return joint_main(arguments)
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        if args.command == "console":
            from .console import serve_console
            serve_console(args.root, port=args.port, open_browser=args.open)
            return 0
        if args.command == "suite":
            from .suite import format_suite, run_suite
            report = run_suite(manifest_path=args.manifest, trials=args.trials, seed=args.seed,
                               baselines=args.baselines, policy_specs=args.policy, cases=args.cases,
                               device=args.device, reward_profile=args.reward,
                               progress=lambda message: print(message, file=sys.stderr, flush=True))
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(format_suite(report) if args.format == "table" else json.dumps(report, indent=2))
            if args.format == "table":
                print(f"\nDetailed report: {args.report}")
            return 0
        if args.command == "train":
            from .learning import ModelConfig, TrainConfig, train
            train(args.output, TrainConfig(
                updates=args.updates, environments=args.envs, flows=args.flows,
                rollout_steps=args.rollout_steps, epochs=args.epochs, minibatch=args.minibatch,
                seed=args.seed, device=args.device, threads=args.threads,
                min_message_bytes=args.min_message_bytes, max_message_bytes=args.max_message_bytes,
                episode_seconds=args.episode_seconds,
                profile=args.profile, reward_profile=args.reward,
                learning_rate=args.learning_rate, target_kl=args.target_kl, checkpoint_every=args.checkpoint_every,
            ), ModelConfig(history=args.history, width=args.width, heads=args.heads,
                           layers=args.layers, action_dimensions=args.actions, architecture=args.architecture))
            return 0
        if args.command == "listen":
            from .udp import receive_udp
            result = receive_udp(args.host, args.port, args.output, timeout=args.timeout,
                                 linger=args.linger, metadata=json.loads(args.metadata),
                                 receipt_feedback=args.receipt_feedback,
                                 ready=(lambda port: print(json.dumps({"ready": True, "port": port}), flush=True))
                                 if args.ready_json else None)
            print(json.dumps(result, indent=2))
            return 0 if result["completed"] else 2
        if args.command == "send":
            from .udp import send_udp
            result = send_udp(_read_text(args), args.host, args.port, deadline=args.deadline,
                              controller=args.controller, checkpoint=args.checkpoint, device=args.device)
            print(json.dumps(result.summary(), indent=2))
            return 0 if result.completed else 2
        if args.flows <= 0:
            raise ValueError("--flows must be positive")
        text, config = _read_text(args), _config(args)
        if args.command == "simulate":
            if args.stagger_seconds < 0:
                raise ValueError("--stagger-seconds must be nonnegative")
            results, report = run_simulation(
                [text] * args.flows, config, args.controller, seed=args.seed,
                checkpoint=args.checkpoint, device=args.device,
                start_times=[i * args.stagger_seconds for i in range(args.flows)],
                reward_profile=args.reward,feedback_profile=args.feedback_profile,
            )
            if args.output is not None and results[0].completed:
                args.output.write_bytes(results[0].received_text.encode("utf-8"))
            if args.show_text and results[0].completed:
                report["received_text"] = results[0].received_text
            status = 0 if all(result.completed for result in results) else 2
        else:
            if args.seeds <= 0:
                raise ValueError("--seeds must be positive")
            runs = []
            if args.text is None and args.input is None:
                text = benchmark_text(args.message_bytes)
            for seed in range(args.seed, args.seed + args.seeds):
                for controller in args.controllers:
                    _, run = run_simulation([text] * args.flows, config, controller, seed=seed,
                                            checkpoint=args.checkpoint, device=args.device, include_rtt_samples=True,
                                            reward_profile=args.reward,feedback_profile=args.feedback_profile)
                    runs.append(run)
            report = {
                "benchmark": {"trials": args.seeds, "flows": args.flows,
                              "payload_bytes": len(text.encode("utf-8")), "seed_start": args.seed,
                              "loss": config.loss, "ack_loss": config.ack_loss,
                              "corruption": config.corruption, "ack_corruption": config.ack_corruption},
                "summary": summarize_runs(runs), "runs": runs,
            }
            status = 0  # Timeouts are valid evaluation outcomes, reported per flow.
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if args.command in ("evaluate", "benchmark") and args.format == "table":
            print(format_benchmark(report))
            print(f"\nDetailed report: {args.report}")
        else:
            print(json.dumps(report, indent=2))
        return status
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
