import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_comparison_rejects_overlapping_validation_and_test_seeds(tmp_path):
    pytest.importorskip("torch")
    plan = json.loads(Path("experiments/robust-v2.json").read_text())
    plan["final_test"]["seed"] = plan["validation"]["seed"]
    source = tmp_path / "overlap.json"
    source.write_text(json.dumps(plan))
    result = subprocess.run([sys.executable, "scripts/compare_learning.py", "--plan", str(source),
                             "--output", str(tmp_path / "output")], capture_output=True, text=True)
    assert result.returncode == 2
    assert "must be disjoint" in result.stderr
    assert not (tmp_path / "output" / "final-test.json").exists()


def test_declared_training_and_model_budgets_are_comparable():
    plan = json.loads(Path("experiments/robust-v2.json").read_text())
    assert {entry["training"]["seed"] for entry in plan["runs"]} == {42, 43}
    assert {entry["model"]["architecture"] for entry in plan["runs"]} == {"transformer", "mlp", "gru"}
    assert {entry["training"]["updates"] for entry in plan["runs"]} == {400}
    assert {entry["model"]["history"] for entry in plan["runs"]} == {16}
    sizes = [entry["parameters"] for entry in plan["runs"]]
    assert max(sizes) / min(sizes) < 1.04
