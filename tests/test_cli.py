import json

from trace_transport.cli import main


def test_cli_preserves_utf8_and_crlf_files(tmp_path, capsys):
    payload = "hello\r\nこんにちは 🌐\x00".encode("utf-8")
    source, destination, report = tmp_path / "in.txt", tmp_path / "out.txt", tmp_path / "run.json"
    source.write_bytes(payload)
    assert main(["simulate", "--input", str(source), "--output", str(destination),
                 "--report", str(report), "--loss", ".1", "--ack-loss", ".2"]) == 0
    assert destination.read_bytes() == payload
    assert json.loads(report.read_text())["flows"][0]["completed"]


def test_cli_timeout_returns_nonzero_and_does_not_write_partial_text(tmp_path, capsys):
    destination = tmp_path / "out.txt"
    assert main(["simulate", "--loss", "1", "--deadline", ".5", "--output", str(destination)]) == 2
    assert not destination.exists()


def test_paired_evaluation_records_same_seeds(tmp_path, capsys):
    path = tmp_path / "evaluation.json"
    assert main(["evaluate", "--seeds", "2", "--report", str(path)]) == 0
    runs = json.loads(path.read_text())["runs"]
    assert [(r["seed"], r["controller"]) for r in runs] == [(0, "fixed"), (0, "aimd"), (1, "fixed"), (1, "aimd")]
