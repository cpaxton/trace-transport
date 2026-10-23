import hashlib
import json

import pytest

from trace_transport.cli import main
from trace_transport.suite import DEFAULT_MANIFEST, load_manifest, parse_policies, run_suite
from trace_transport.suite import select_candidate


def test_bundled_suite_is_versioned_and_fingerprinted():
    manifest, digest = load_manifest()
    assert manifest["name"] == "reference-v1"
    assert [case["name"] for case in manifest["cases"]] == ["clean", "lossy", "changing", "stress"]
    assert digest == hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest()
    assert manifest["cases"][-1]["flows"] == 4


def test_suite_baselines_use_identical_seeds_and_configs():
    report = run_suite(cases=["clean"], trials=2, seed=7000)
    case = report["cases"][0]
    assert [(run["seed"], run["controller"]) for run in case["runs"]] == [
        (7000, "fixed"), (7000, "aimd"), (7001, "fixed"), (7001, "aimd"),
    ]
    assert case["runs"][0]["config"] == case["runs"][1]["config"]
    assert all(row["completion_percent"] == 100 for row in case["summary"])
    assert report == run_suite(cases=["clean"], trials=2, seed=7000)


def test_suite_json_stdout_is_parseable_and_progress_goes_to_stderr(tmp_path, capsys):
    output = tmp_path / "suite.json"
    assert main(["suite", "--cases", "clean", "--trials", "1", "--format", "json",
                 "--report", str(output)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == json.loads(output.read_text())
    assert "Evaluating clean" in captured.err


@pytest.mark.parametrize("options", [dict(trials=0), dict(cases=["unknown"]), dict(baselines=()),
                                   dict(baselines=("fixed", "fixed"))])
def test_invalid_suite_selections_fail_before_running(options):
    with pytest.raises(ValueError):
        run_suite(**options)


def test_invalid_policy_labels_and_missing_checkpoints(tmp_path):
    checkpoint = tmp_path / "placeholder.pt"
    checkpoint.write_bytes(b"placeholder")
    for specifications in [["missing-separator"], ["model=missing.pt"], [f"fixed={checkpoint}"],
                           [f"model={checkpoint}", f"model={checkpoint}"]]:
        with pytest.raises(ValueError):
            parse_policies(specifications, ["fixed", "aimd"])


def test_named_checkpoints_keep_their_identity_and_hash(tmp_path):
    pytest.importorskip("torch")
    from trace_transport.learning import HistoryTransformer, ModelConfig, save_checkpoint

    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, HistoryTransformer(ModelConfig(history=8, width=16, heads=2, layers=1)))
    report = run_suite(cases=["clean"], trials=1, baselines=(), policy_specs=[f"candidate={checkpoint}"])
    run = report["cases"][0]["runs"][0]
    assert run["controller"] == "candidate"
    assert run["policy_type"] == "transformer"
    assert run["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert report["cases"][0]["summary"][0]["completion_percent"] == 100


def test_candidate_selection_prioritizes_completion_over_reward():
    report = {"cases": [{"summary": [
        {"controller": "complete", "completed_transfers": 10, "transfers": 10,
         "mean_discounted_rl_return": 1, "wrong_payloads": 0},
        {"controller": "high_reward", "completed_transfers": 9, "transfers": 10,
         "mean_discounted_rl_return": 10, "wrong_payloads": 0},
    ]}]}
    winner, _ = select_candidate(report, ["complete", "high_reward"])
    assert winner == "complete"
    report["cases"][0]["summary"][1]["completed_transfers"] = 10
    assert select_candidate(report, ["complete", "high_reward"])[0] == "high_reward"
    report["cases"][0]["summary"][1]["wrong_payloads"] = 1
    assert select_candidate(report, ["complete", "high_reward"])[0] == "complete"
