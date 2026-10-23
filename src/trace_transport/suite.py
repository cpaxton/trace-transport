"""Versioned scenario suites for comparing named checkpoints on fixed seeds."""

import hashlib
import json
from pathlib import Path
from statistics import fmean

from .benchmark import benchmark_text, summarize_runs
from .simulator import NetworkConfig
from .rewards import reward_spec

DEFAULT_MANIFEST = Path(__file__).with_name("scenarios") / "reference-v1.json"


def load_manifest(path=None):
    source = Path(path) if path is not None else DEFAULT_MANIFEST
    raw = source.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("version") != 1 or not manifest.get("cases"):
        raise ValueError("suite manifest requires version 1 and nonempty cases")
    names = set()
    for case in manifest["cases"]:
        if not case.get("name") or case["name"] in names:
            raise ValueError("suite case names must be nonempty and unique")
        names.add(case["name"])
        if type(case.get("flows")) is not int or case["flows"] < 1:
            raise ValueError("suite flow counts must be positive integers")
        # Validate size without allocating the full workload.
        size = case.get("message_bytes")
        if type(size) is not int or not 1 <= size <= 16 * 1024 * 1024:
            raise ValueError("suite message_bytes must be between 1 and 16 MiB")
        NetworkConfig(**case["network"])
    return manifest, hashlib.sha256(raw).hexdigest()


def parse_policies(specifications, baselines):
    labels = set(baselines)
    if len(labels) != len(baselines):
        raise ValueError("baseline names must be unique")
    policies = []
    for specification in specifications:
        label, separator, checkpoint = specification.partition("=")
        if not separator or not label.strip() or not checkpoint:
            raise ValueError("each --policy must be LABEL=CHECKPOINT")
        if label in labels or any(character.isspace() for character in label):
            raise ValueError("policy labels must be unique and contain no whitespace")
        if not Path(checkpoint).is_file():
            raise ValueError(f"checkpoint does not exist: {checkpoint}")
        labels.add(label)
        policies.append((label, checkpoint))
    if not labels:
        raise ValueError("select a baseline or a checkpoint policy")
    return policies


def run_suite(*, manifest_path=None, trials=10, seed=1000, baselines=("fixed", "aimd"),
              policy_specs=(), cases=None, device="cpu", progress=None, reward_profile="legacy"):
    # Import here so cli can expose this command without an import cycle.
    from .cli import run_simulation

    if type(trials) is not int or trials < 1:
        raise ValueError("--trials must be a positive integer")
    if any(baseline not in ("fixed", "aimd") for baseline in baselines):
        raise ValueError("suite baselines must be fixed or aimd")
    policies = parse_policies(policy_specs, baselines)
    scoring = reward_spec(reward_profile).metadata()
    manifest, manifest_hash = load_manifest(manifest_path)
    available = {case["name"] for case in manifest["cases"]}
    if cases is not None and (not cases or set(cases) - available):
        raise ValueError(f"unknown suite cases; available: {', '.join(sorted(available))}")
    selected = [case for case in manifest["cases"] if cases is None or case["name"] in cases]
    controllers = [(name, name, None) for name in baselines]
    controllers.extend((label, "transformer", checkpoint) for label, checkpoint in policies)
    reports = []
    for case in selected:
        if progress:
            progress(f"Evaluating {case['name']}: {trials} paired trials x {len(controllers)} controllers")
        text = benchmark_text(case["message_bytes"])
        config = NetworkConfig(**case["network"])
        runs = []
        for trial_seed in range(seed, seed + trials):
            for label, controller, checkpoint in controllers:
                _, run = run_simulation([text] * case["flows"], config, controller,
                                        seed=trial_seed, checkpoint=checkpoint, device=device,
                                        include_rtt_samples=True, reward_profile=reward_profile)
                run["policy_type"] = controller
                run["controller"] = label
                runs.append(run)
        reports.append({"name": case["name"], "description": case.get("description", ""),
                        "message_bytes": case["message_bytes"], "flows": case["flows"],
                        "summary": summarize_runs(runs), "runs": runs})
    return {"suite": manifest["name"], "manifest": manifest, "manifest_sha256": manifest_hash,
            "trials_per_case": trials, "seed_start": seed, "cases": reports, "scoring": scoring}


def format_suite(report):
    lines = [f"Suite {report['suite']} | {report['trials_per_case']} paired trials/scenario"
             f" | seeds {report['seed_start']}..{report['seed_start'] + report['trials_per_case'] - 1}",
             "", f"{'Scenario':<12} {'Controller':<16} {'Complete':>13} {'RL score':>10}"
             f" {'Done p50':>10} {'Retry':>8} {'Wrong':>7}", "-" * 84]
    for case in report["cases"]:
        for row in case["summary"]:
            complete = f"{row['completed_transfers']}/{row['transfers']} {row['completion_percent']:.0f}%"
            duration = row["completion_p50_seconds"]
            duration_text = f"{duration:.3f}s" if duration is not None else "--"
            lines.append(f"{case['name']:<12} {row['controller']:<16} {complete:>13}"
                         f" {row['mean_discounted_rl_return']:>10.3f} {duration_text:>10}"
                         f" {row['retransmission_percent']:>7.1f}% {row['wrong_payloads']:>7}")
    scoring = report.get("scoring", {"version": "transport-v1", "discount_gamma": .99})
    lines.extend(["", f"RL score: {scoring['version']}, gamma={scoring['discount_gamma']}; failures remain in the score.",
                  "Compare controllers within each scenario: message sizes and conditions differ between scenarios.",
                  "Done p50 includes successful transfers only; read it alongside completion rate."])
    return "\n".join(lines)


def select_candidate(report, labels):
    """Validation-only selection: completion first, then common per-case score."""
    metrics = {}
    for label in labels:
        rows = [next((row for row in case["summary"] if row["controller"] == label), None)
                for case in report["cases"]]
        if not rows or any(row is None for row in rows):
            raise ValueError(f"candidate missing from validation scenarios: {label}")
        if any(row["wrong_payloads"] for row in rows):
            continue
        metrics[label] = {
            "completed": sum(row["completed_transfers"] for row in rows),
            "transfers": sum(row["transfers"] for row in rows),
            "mean_scenario_score": fmean(row["mean_discounted_rl_return"] for row in rows),
        }
    if not metrics:
        raise ValueError("no byte-correct validation candidates")
    winner = max(metrics, key=lambda label: (metrics[label]["completed"], metrics[label]["mean_scenario_score"]))
    return winner, metrics
