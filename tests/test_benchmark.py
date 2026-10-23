from dataclasses import replace
import json

import numpy as np
import pytest

from trace_transport import NetworkConfig
from trace_transport.benchmark import benchmark_text, summarize_runs
from trace_transport.cli import build_parser, main, run_simulation
from trace_transport.environment import TransportEnv


def test_benchmark_defaults_to_ten_paired_trials_and_real_traffic():
    args = build_parser().parse_args(["benchmark"])
    assert args.seeds == 10
    assert args.flows == 2
    assert args.format == "table"
    assert args.corruption == .01 and args.ack_corruption == .01
    assert len(benchmark_text(args.message_bytes).encode("utf-8")) == 32_768
    assert build_parser().parse_args(["evaluate", "--trials", "3"]).seeds == 3


@pytest.mark.parametrize("size", [1, 7, 512, 32_768])
def test_generated_workload_has_exact_valid_utf8_size(size):
    assert len(benchmark_text(size).encode("utf-8")) == size


def test_reported_rl_returns_are_actual_environment_rewards():
    cfg = NetworkConfig(loss=0, ack_loss=0, jitter=0)
    text = "Actual environment rewards 🌐" * 50
    _, report = run_simulation([text], cfg, "fixed", seed=4)
    env = TransportEnv([text], cfg)
    env.reset(seed=4)
    total = discounted = 0.0
    discount = 1.0
    while not env.simulator.finished:
        _, rewards, _, _, _ = env.step([[0]])
        total += float(rewards[0])
        discounted += discount * float(rewards[0])
        discount *= .99
    assert report["flows"][0]["rl_return"] == pytest.approx(total)
    assert report["flows"][0]["discounted_rl_return"] == pytest.approx(discounted)


def test_summary_distinguishes_confirmation_integrity_and_missing_output():
    clean = NetworkConfig(loss=0, ack_loss=0, jitter=0, deadline=.5)
    runs = [run_simulation(["hello"], cfg, "fixed", include_rtt_samples=True)[1] for cfg in [
        clean, replace(clean, ack_loss=1), replace(clean, loss=1),
    ]]
    summary = summarize_runs(runs)[0]
    assert summary["transfers"] == 3
    assert summary["completed_transfers"] == 1
    assert summary["completion_percent"] == pytest.approx(100 / 3)
    assert summary["exact_receiver_percent"] == pytest.approx(200 / 3)
    assert summary["wrong_payloads"] == 0
    assert summary["timeouts"] == 2
    assert runs[2]["flows"][0]["integrity_ok"] is None
    assert summary["mean_rl_return"] == pytest.approx(np.mean([r["flows"][0]["rl_return"] for r in runs]))
    assert summary["rtt_p95_seconds"] is not None


@pytest.mark.parametrize("corruption,ack_corruption,receiver_complete", [(1, 0, False), (0, 1, True)])
def test_corrupted_datagrams_are_rejected_without_false_success(corruption, ack_corruption, receiver_complete):
    cfg = NetworkConfig(loss=0, ack_loss=0, corruption=corruption, ack_corruption=ack_corruption, deadline=.6)
    results, run = run_simulation(["hello"], cfg, "fixed", seed=5)
    assert not results[0].completed
    assert results[0].receiver_complete is receiver_complete
    link = run["network"]["forward" if corruption else "feedback"]
    assert link["corrupted_packets"] > 0
    assert link["corruption_rejections"] > 0
    assert link["delivered_packets"] == 0
    assert summarize_runs([run])[0]["wrong_payloads"] == 0


def test_loss_and_corruption_recover_exact_payload():
    text = "recover every byte 🌐\r\n" * 300
    results, run = run_simulation([text], NetworkConfig(
        loss=.05, ack_loss=.05, corruption=.15, ack_corruption=.15, deadline=15,
    ), "fixed", seed=7)
    assert results[0].completed and results[0].received_text == text
    assert run["flows"][0]["integrity_ok"] is True
    assert sum(link["corruption_rejections"] for link in run["network"].values()) > 0
    assert results[0].retransmitted_payload_bytes > 0


def test_readable_benchmark_and_machine_readable_report(tmp_path, capsys):
    report = tmp_path / "benchmark.json"
    assert main(["benchmark", "--trials", "2", "--controllers", "fixed", "--message-bytes", "2048",
                 "--corruption", "0", "--ack-corruption", "0", "--report", str(report)]) == 0
    output = capsys.readouterr().out
    assert "2 paired trials/controller" in output
    assert "Complete" in output and "CRC reject" in output and "RL score" in output
    data = json.loads(report.read_text())
    assert data["summary"][0]["completed_transfers"] == 4
    assert len(data["runs"]) == 2
    assert data["benchmark"]["payload_bytes"] == 2048


def test_json_output_and_explicit_empty_text(tmp_path, capsys):
    report = tmp_path / "benchmark.json"
    assert main(["benchmark", "--text", "", "--trials", "1", "--controllers", "fixed", "--format", "json",
                 "--report", str(report)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["benchmark"]["payload_bytes"] == 0
    assert output["summary"][0]["completion_percent"] == 100
    assert output["summary"][0]["retransmission_percent"] == 0


def test_complete_failure_is_a_valid_scored_benchmark(tmp_path, capsys):
    report = tmp_path / "failed.json"
    assert main(["benchmark", "--trials", "1", "--flows", "1", "--controllers", "fixed",
                 "--loss", "1", "--deadline", ".2", "--report", str(report)]) == 0
    summary = json.loads(report.read_text())["summary"][0]
    assert summary["completion_percent"] == 0
    assert summary["completion_p50_seconds"] is None
    assert summary["mean_rl_return"] < 0
    assert summary["drop_percent"] == 100


@pytest.mark.parametrize("name,value", [("--trials", "0"), ("--message-bytes", "0"),
                                       ("--message-bytes", "16777217"),
                                       ("--corruption", "1.1"), ("--ack-corruption", "-1")])
def test_invalid_benchmark_parameters_are_rejected(name, value, tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["benchmark", name, value, "--report", str(tmp_path / "invalid.json")])
    assert error.value.code == 1
