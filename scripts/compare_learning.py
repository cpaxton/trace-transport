"""Evaluate a declared learning plan, select on validation, then test fresh seeds.

Run after training, or pass --wait while the independent training jobs finish.
The plan specifies checkpoint paths and expected final update counts. The final
test is evaluated once, after writing the validation-only selection decision.
"""

import argparse
import contextlib
import json
from pathlib import Path
import shutil
import time

import torch
from dataclasses import asdict

from trace_transport.learning import ModelConfig, TrainConfig

from trace_transport.suite import format_suite, run_suite, select_candidate


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def matching_configuration(checkpoint, entry):
    return (asdict(ModelConfig(**checkpoint["model_config"])) == asdict(ModelConfig(**entry["model"]))
            and asdict(TrainConfig(**checkpoint["training"]["config"])) == asdict(TrainConfig(**entry["training"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("experiments/robust-v2.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/robust-v2"))
    parser.add_argument("--best-checkpoint", type=Path, default=Path("checkpoints/robust-v2-best.pt"))
    parser.add_argument("--train", action="store_true", help="train missing candidates; reuse matching completed checkpoints")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validation_seeds = set(range(plan["validation"]["seed"],
                                 plan["validation"]["seed"] + plan["validation"]["trials"]))
    test_seeds = set(range(plan["final_test"]["seed"],
                           plan["final_test"]["seed"] + plan["final_test"]["trials"]))
    if validation_seeds & test_seeds:
        parser.error("validation and final-test seeds must be disjoint")
    entries = [*plan["runs"], plan["reward_ablation"]]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.train:
        from trace_transport.learning import train
        for entry in entries:
            path = Path(entry["checkpoint"])
            existing = torch.load(path, weights_only=True) if path.exists() else {}
            if existing and not matching_configuration(existing, entry):
                parser.error(f"checkpoint configuration differs from plan: {entry['label']}")
            if existing.get("training", {}).get("update", 0) >= entry["training"]["updates"]:
                print(f"Using completed checkpoint: {entry['label']}", flush=True)
                continue
            print(f"Training {entry['label']} (partial checkpoints restart from the declared seed)", flush=True)
            with (output / f"{entry['label']}.train.log").open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log):
                    train(entry["checkpoint"], TrainConfig(**entry["training"]), ModelConfig(**entry["model"]))
    torch.set_num_threads(1)
    start = time.monotonic()
    while True:
        pending = []
        for entry in entries:
            path = Path(entry["checkpoint"])
            checkpoint = torch.load(path, weights_only=True) if path.exists() else {}
            if checkpoint.get("training", {}).get("update", 0) < entry["training"]["updates"]:
                pending.append(entry["label"])
            elif not matching_configuration(checkpoint, entry):
                parser.error(f"checkpoint configuration differs from plan: {entry['label']}")
        if not pending:
            break
        if not args.wait or time.monotonic() - start > args.timeout:
            parser.error(f"training incomplete: {', '.join(pending)}")
        print(f"Waiting for final checkpoints: {', '.join(pending)}", flush=True)
        time.sleep(5)

    policies = [f"{entry['label']}={entry['checkpoint']}" for entry in entries]
    previous = plan.get("previous_checkpoint")
    if previous and not Path(previous).is_file():
        print(f"Optional previous checkpoint is unavailable; comparing new candidates and baselines: {previous}", flush=True)
        previous = None
    if previous:
        policies.append(f"previous={previous}")
    validation = run_suite(trials=plan["validation"]["trials"], seed=plan["validation"]["seed"],
                           policy_specs=policies, reward_profile="completion",
                           progress=lambda message: print(message, flush=True))
    if validation["manifest_sha256"] != plan["suite_sha256"]:
        raise ValueError("evaluation manifest changed since the experiment was declared")
    write_json(output / "validation.json", validation)
    (output / "validation.txt").write_text(format_suite(validation) + "\n", encoding="utf-8")
    winner, metrics = select_candidate(validation, [entry["label"] for entry in entries])
    candidate = next(entry for entry in entries if entry["label"] == winner)
    args.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.best_checkpoint.resolve() != Path(candidate["checkpoint"]).resolve():
        shutil.copyfile(candidate["checkpoint"], args.best_checkpoint)
    decision = {"selected": winner, "checkpoint": candidate["checkpoint"], "validation_metrics": metrics,
                "rule": plan["selection"], "reward_profile": "completion", "selected_copy": str(args.best_checkpoint)}
    write_json(output / "selection.json", decision)
    print(f"Selected on validation: {winner}", flush=True)

    # The reserved final range cannot influence the decision above.
    final_policies = [f"{winner}={candidate['checkpoint']}"]
    if previous:
        final_policies.append(f"previous={previous}")
    final = run_suite(trials=plan["final_test"]["trials"], seed=plan["final_test"]["seed"],
                      policy_specs=final_policies, reward_profile="completion",
                      progress=lambda message: print(message, flush=True))
    write_json(output / "final-test.json", final)
    final_text = format_suite(final)
    (output / "final-test.txt").write_text(final_text + "\n", encoding="utf-8")
    print(final_text)


if __name__ == "__main__":
    main()
