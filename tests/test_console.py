import csv
import io
import json
from types import SimpleNamespace
import time

import pytest

from trace_transport.console import TestBench as Bench, make_handler, validated_plan
from trace_transport.cli import run_simulation
from trace_transport.simulator import NetworkConfig


def wait(bench, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = bench.snapshot(job_id)
        if state["status"] not in ("queued", "running", "stopping"):
            return state
        time.sleep(.005)
    pytest.fail("batch did not finish")


def test_console_runs_real_paired_trials_and_exports_csv(tmp_path):
    bench = Bench(tmp_path)
    try:
        state = bench.start({"scenario": "clean", "trials": 2, "flows": 2, "message_bytes": 2048,
                             "controllers": ["fixed", "aimd"], "seed": 7000})
        done = wait(bench, state["id"])
        assert done["status"] == "completed", done["error"]
        assert done["finished_runs"] == 4
        assert sum(row["completed_transfers"] for row in done["summary"]) == 8
        assert done["current"]["flows"][0]["complete"]
        assert bench.trace(state["id"], 0)["points"]
        report = bench.report(state["id"])
        assert [(run["seed"], run["controller"]) for run in report["runs"]] == [
            (7000, "fixed"), (7000, "aimd"), (7001, "fixed"), (7001, "aimd")]
        records = list(csv.DictReader(io.StringIO(bench.csv(state["id"]))))
        assert len(records) == 8
        assert all(row["completed"] == "True" and row["integrity_ok"] == "True" for row in records)
    finally:
        bench.close()


def test_progress_observer_does_not_change_simulation_or_scoring():
    settings = NetworkConfig(loss=.1, ack_loss=.1, corruption=.1)
    points = []
    original, a = run_simulation(["telemetry 🌐" * 100], settings, "fixed", seed=4, reward_profile="completion")
    observed, b = run_simulation(["telemetry 🌐" * 100], settings, "fixed", seed=4,
                                 reward_profile="completion", progress=points.append)
    assert original == observed and a == b
    assert points and points[-1]["flows"][0]["complete"]


def test_cancelling_does_not_score_unfinished_trial_as_a_result(tmp_path):
    bench = Bench(tmp_path)
    try:
        state = bench.start({"trials": 50, "flows": 4, "message_bytes": 100_000,
                             "controllers": ["fixed"], "network": {"loss": 1, "deadline": 60}})
        bench.cancel(state["id"])
        done = wait(bench, state["id"])
        assert done["status"] == "cancelled"
        assert not bench.report(state["id"])["runs"]
    finally:
        bench.close()


@pytest.mark.parametrize("text", ["", "こんにちは 🌐\r\n"])
def test_console_inline_text_preserves_exact_payload(text, tmp_path):
    bench = Bench(tmp_path)
    try:
        state = bench.start({"scenario": "clean", "trials": 1, "flows": 1,
                             "controllers": ["fixed"], "text": text})
        assert wait(bench, state["id"])["status"] == "completed"
        flow = bench.report(state["id"])["runs"][0]["flows"][0]
        assert flow["integrity_ok"] and flow["payload_bytes"] == len(text.encode())
    finally:
        bench.close()


@pytest.mark.parametrize("payload", [{"trials": 0}, {"controllers": []}, {"flows": 20},
                                     {"controllers": ["learned"], "checkpoint": "/etc/passwd"},
                                     {"network": {"loss": float("nan")}},
                                     {"text": "\ud800"}, {"network": {"deadline": 1000}}])
def test_console_rejects_invalid_plans(payload):
    with pytest.raises(ValueError):
        validated_plan(payload, [])


def request(bench, method, path, body=None, origin=None):
    encoded = json.dumps(body or {}).encode()
    headers = [f"{method} {path} HTTP/1.1", "Host: 127.0.0.1:8765", "Connection: close"]
    if origin:
        headers.append(f"Origin: {origin}")
    if method == "POST":
        headers.extend(["Content-Type: application/json", f"Content-Length: {len(encoded)}"])
    raw = ("\r\n".join(headers) + "\r\n\r\n").encode() + (encoded if method == "POST" else b"")

    class Connection:
        def __init__(self):
            self.output = bytearray()

        def makefile(self, *args):
            return io.BytesIO(raw)

        def sendall(self, data):
            self.output.extend(data)

    connection = Connection()
    make_handler(bench)(connection, ("127.0.0.1", 12345), SimpleNamespace(server_port=8765))
    header, data = bytes(connection.output).split(b"\r\n\r\n", 1)
    return int(header.split()[1]), data


def test_http_routes_serve_the_application_and_block_cross_origin_jobs(tmp_path):
    bench = Bench(tmp_path)
    try:
        status, content = request(bench, "GET", "/")
        assert status == 200 and b"Controller comparison" in content
        assert request(bench, "GET", "/app.js")[0] == 200
        status, content = request(bench, "GET", "/api/catalog")
        assert status == 200
        assert {"clean", "lossy", "changing", "stress", "wifi-mobile-fade"} <= {
            case["name"] for case in json.loads(content)["scenarios"]}
        assert request(bench, "GET", "/../../etc/passwd")[0] == 404
        assert request(bench, "POST", "/api/jobs", origin="https://example.com")[0] == 403
        status, content = request(bench, "POST", "/api/jobs", {"scenario": "clean", "trials": 1,
                                                                 "controllers": ["fixed"], "message_bytes": 512})
        assert status == 202
        job_id = json.loads(content)["id"]
        wait(bench, job_id)
        status, content = request(bench, "GET", f"/api/jobs/{job_id}/report.json")
        assert status == 200 and json.loads(content)["status"] == "completed"
    finally:
        bench.close()
