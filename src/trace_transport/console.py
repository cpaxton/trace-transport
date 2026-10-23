"""Local browser test bench, backed by the real packet simulator."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import asdict
import copy
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import importlib.util
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import urlparse
import uuid
import webbrowser

from .benchmark import benchmark_text, summarize_runs
from .cli import run_simulation
from .simulator import NetworkConfig
from .suite import load_manifest

STATIC = Path(__file__).with_name("web")


def model_catalog(root):
    models = []
    for directory in ("artifacts", "checkpoints"):
        for path in sorted((Path(root) / directory).glob("*.pt")):
            models.append({"id": path.relative_to(root).as_posix(), "label": path.stem,
                           "bytes": path.stat().st_size})
    return models


def validated_plan(request, catalog):
    manifest, _ = load_manifest()
    cases = {case["name"]: case for case in manifest["cases"]}
    case = cases.get(request.get("scenario", "lossy"))
    if case is None:
        raise ValueError("unknown scenario")

    def integer(name, default, minimum, maximum):
        value = request.get(name, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
        return value

    trials = integer("trials", 10, 1, 50)
    flows = integer("flows", case["flows"], 1, 8)
    seed = integer("seed", 3000, 0, 2**31 - 1)
    size = integer("message_bytes", case["message_bytes"], 1, 2 * 1024 * 1024)
    text = request.get("text")
    if text is not None:
        try:
            if not isinstance(text, str) or len(text.encode("utf-8")) > 1024 * 1024:
                raise ValueError("inline text must be UTF-8 text of at most 1 MiB")
        except UnicodeEncodeError as exc:
            raise ValueError("inline text must be valid UTF-8") from exc
    controllers = request.get("controllers", ["fixed", "aimd"])
    if (not isinstance(controllers, list) or any(name not in ("fixed", "aimd", "learned") for name in controllers)
            or len(set(controllers)) != len(controllers) or not controllers):
        raise ValueError("select unique controllers from fixed, aimd, learned")
    checkpoint = request.get("checkpoint") if "learned" in controllers else None
    if checkpoint is not None and checkpoint not in {model["id"] for model in catalog}:
        raise ValueError("choose a checkpoint from the model list")
    if "learned" in controllers and checkpoint is None:
        raise ValueError("a checkpoint is required for the learned controller")
    overrides = request.get("network", {})
    allowed = {"bandwidth_bps", "base_rtt", "jitter", "loss", "ack_loss", "corruption",
               "ack_corruption", "queue_bytes", "deadline"}
    if not isinstance(overrides, dict) or set(overrides) - allowed:
        raise ValueError("unsupported network control")
    for value in overrides.values():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("network controls must be finite numbers")
    config = NetworkConfig(**{**case["network"], **overrides})
    if not .1 <= config.deadline <= 60:
        raise ValueError("deadline must be between 0.1 and 60 simulated seconds")
    if not 100_000 <= config.bandwidth_bps <= 100_000_000:
        raise ValueError("bandwidth must be between 0.1 and 100 Mbit/s")
    reward = request.get("reward", "completion")
    if reward not in ("legacy", "completion"):
        raise ValueError("unknown reward profile")
    return {"scenario": case["name"], "trials": trials, "flows": flows, "seed": seed,
            "message_bytes": len(text.encode("utf-8")) if text is not None else size,
            "text": text, "controllers": controllers, "checkpoint": checkpoint,
            "network": asdict(config), "reward": reward}


class TestBench:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.models = model_catalog(self.root)
        self.jobs = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trace-test")

    def catalog(self):
        manifest, _ = load_manifest()
        with self.lock:
            active = next((job["id"] for job in self.jobs.values()
                           if job["status"] in ("queued", "running", "stopping")), None)
        return {"models": self.models, "learning_available": importlib.util.find_spec("torch") is not None,
                "active_job": active, "scenarios": [
                    {**case, "network": asdict(NetworkConfig(**case["network"]))} for case in manifest["cases"]]}

    def start(self, request):
        plan = validated_plan(request, self.models)
        with self.lock:
            if any(job["status"] in ("queued", "running", "stopping") for job in self.jobs.values()):
                raise ValueError("a batch is already running; stop it or wait for completion")
            if len(self.jobs) >= 12:
                del self.jobs[next(iter(self.jobs))]
            job_id = uuid.uuid4().hex[:12]
            self.jobs[job_id] = {"id": job_id, "status": "queued", "plan": plan,
                                 "reports": [], "traces": [], "current": None, "points": [],
                                 "error": None, "stop": threading.Event(), "started": time.monotonic(),
                                 "elapsed_wall_seconds": 0.0}
        self.executor.submit(self._run, job_id)
        return self.snapshot(job_id)

    def _run(self, job_id):
        job = self.jobs[job_id]
        plan = job["plan"]
        text = plan["text"] if plan["text"] is not None else benchmark_text(plan["message_bytes"])
        config = NetworkConfig(**plan["network"])
        with self.lock:
            job["status"] = "running"
        try:
            for seed in range(plan["seed"], plan["seed"] + plan["trials"]):
                for controller in plan["controllers"]:
                    if job["stop"].is_set():
                        raise CancelledError()
                    with self.lock:
                        job["points"] = []
                    last_sample = -math.inf

                    def progress(state):
                        nonlocal last_sample
                        if job["stop"].is_set():
                            raise CancelledError()
                        with self.lock:
                            job["current"] = {**state, "seed": seed, "controller": controller}
                            if state["time"] - last_sample >= .05 or all(flow["complete"] for flow in state["flows"]):
                                last_sample = state["time"]
                                job["points"].append(copy.deepcopy(job["current"]))
                                if len(job["points"]) > 300:
                                    job["points"] = job["points"][::2]
                    checkpoint = str(self.root / plan["checkpoint"]) if controller == "learned" else None
                    _, report = run_simulation([text] * plan["flows"], config, controller,
                                               seed=seed, checkpoint=checkpoint, reward_profile=plan["reward"],
                                               include_rtt_samples=True, progress=progress)
                    with self.lock:
                        job["reports"].append(report)
                        job["traces"].append({"seed": seed, "controller": controller,
                                              "points": copy.deepcopy(job["points"])})
            with self.lock:
                job["status"] = "completed"
        except CancelledError:
            with self.lock:
                job["status"] = "cancelled"
        except Exception as exc:
            with self.lock:
                job["status"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with self.lock:
                job["elapsed_wall_seconds"] = time.monotonic() - job["started"]
            directory = self.root / "runs" / "console"
            try:
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"{job_id}.json").write_text(json.dumps(self.report(job_id), indent=2), encoding="utf-8")
            except OSError as exc:
                with self.lock:
                    job["error"] = f"Report could not be saved: {exc}"

    def snapshot(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            reports = job["reports"]
            plan = {key: value for key, value in job["plan"].items() if key != "text"}
            return copy.deepcopy({
                "id": job_id, "status": job["status"], "plan": plan, "error": job["error"],
                "finished_runs": len(reports), "total_runs": plan["trials"] * len(plan["controllers"]),
                "summary": summarize_runs(reports) if reports else [],
                "current": job["current"], "points": job["points"],
                "traces": [{"index": i, "seed": trace["seed"], "controller": trace["controller"]}
                           for i, trace in enumerate(job["traces"])],
                "elapsed_wall_seconds": (time.monotonic() - job["started"] if job["status"] in
                                         ("queued", "running", "stopping") else job["elapsed_wall_seconds"]),
            })

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] in ("queued", "running"):
                job["stop"].set()
                job["status"] = "stopping"
        return self.snapshot(job_id)

    def trace(self, job_id, index):
        with self.lock:
            return copy.deepcopy(self.jobs[job_id]["traces"][index])

    def report(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            return copy.deepcopy({"status": job["status"], "plan": job["plan"], "error": job["error"],
                                  "summary": summarize_runs(job["reports"]) if job["reports"] else [],
                                  "runs": job["reports"], "traces": job["traces"],
                                  "elapsed_wall_seconds": job["elapsed_wall_seconds"]})

    def csv(self, job_id):
        report = self.report(job_id)
        output = io.StringIO(newline="")
        fields = ["controller", "seed", "flow", "completed", "integrity_ok", "payload_bytes",
                  "acknowledged_bytes", "elapsed_seconds", "retransmitted_payload_bytes", "data_wire_bytes",
                  "feedback_wire_bytes", "rl_return", "discounted_rl_return"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for run in report["runs"]:
            for index, flow in enumerate(run["flows"]):
                writer.writerow({field: ({"controller": run["controller"], "seed": run["seed"], "flow": index}
                                         | flow).get(field) for field in fields})
        return output.getvalue()

    def close(self):
        with self.lock:
            for job in self.jobs.values():
                job["stop"].set()
        self.executor.shutdown(wait=True, cancel_futures=True)


def make_handler(bench):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send_body(self, code, body, kind="application/json; charset=utf-8"):
            encoded = json.dumps(body, allow_nan=False).encode() if isinstance(body, (dict, list)) else body
            if isinstance(encoded, str):
                encoded = encoded.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(encoded)

        def allowed_request(self):
            host = self.headers.get("Host", "")
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            return host in allowed and (not origin or origin == f"http://{host}")

        def do_GET(self):
            if not self.allowed_request():
                self.send_body(403, {"error": "Use the local console origin"})
                return
            path = urlparse(self.path).path
            static = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "application/javascript; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8")}
            try:
                if path in static:
                    file, kind = static[path]
                    self.send_body(200, (STATIC / file).read_bytes(), kind)
                elif path == "/api/catalog":
                    self.send_body(200, bench.catalog())
                else:
                    parts = path.strip("/").split("/")
                    if len(parts) < 3 or parts[:2] != ["api", "jobs"]:
                        raise KeyError(path)
                    job_id = parts[2]
                    if len(parts) == 3:
                        self.send_body(200, bench.snapshot(job_id))
                    elif parts[3:] == ["report.json"]:
                        self.send_body(200, bench.report(job_id))
                    elif parts[3:] == ["report.csv"]:
                        self.send_body(200, bench.csv(job_id), "text/csv; charset=utf-8")
                    elif len(parts) == 5 and parts[3] == "trace":
                        self.send_body(200, bench.trace(job_id, int(parts[4])))
                    else:
                        raise KeyError(path)
            except (KeyError, IndexError, ValueError):
                self.send_body(404, {"error": "No such run or resource"})

        def do_POST(self):
            if not self.allowed_request():
                self.send_body(403, {"error": "Use the local console origin"})
                return
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("send application/json")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2 * 1024 * 1024:
                    raise ValueError("request is empty or too large")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("request must be an object")
                path = urlparse(self.path).path
                if path == "/api/jobs":
                    self.send_body(202, bench.start(data))
                elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                    self.send_body(200, bench.cancel(path.split("/")[3]))
                else:
                    self.send_body(404, {"error": "No such action"})
            except (ValueError, UnicodeDecodeError) as exc:
                self.send_body(400, {"error": str(exc)})
            except KeyError:
                self.send_body(404, {"error": "No such run"})
    return Handler


def serve_console(root, *, port=8765, open_browser=False):
    bench = TestBench(root)
    try:
        with ThreadingHTTPServer(("127.0.0.1", port), make_handler(bench)) as server:
            url = f"http://127.0.0.1:{server.server_port}"
            print(f"Trace Transport test bench: {url}\nPress Ctrl+C to stop.", flush=True)
            if open_browser:
                webbrowser.open(url)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        bench.close()
