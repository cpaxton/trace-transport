"""Readable batch evaluation with actual environment returns and wire counters."""

from collections import defaultdict
from statistics import fmean, stdev

from .protocol import MAX_MESSAGE_BYTES


def benchmark_text(size: int = 32_768) -> str:
    """A deterministic multi-packet workload of exactly size UTF-8 bytes."""
    if type(size) is not int or not 1 <= size <= MAX_MESSAGE_BYTES:
        raise ValueError("--message-bytes must be an integer between 1 byte and 16 MiB")
    unit = "Trace Transport: exact delivery. こんにちは 🌐\r\n".encode("utf-8")
    count, remainder = divmod(size, len(unit))
    return (unit * count + b"x" * remainder).decode("utf-8")


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(len(ordered) - 1, lower + 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize_runs(runs):
    scoring = [run.get("reward") for run in runs if run.get("reward") is not None]
    if scoring and any(spec != scoring[0] for spec in scoring):
        raise ValueError("cannot compare runs with different reward definitions")
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["controller"]].append(run)
    summaries = []
    for controller, trials in grouped.items():
        flows = [flow for run in trials for flow in run["flows"]]
        links = [link for run in trials for link in run["network"].values()]
        completed = sum(flow["completed"] and flow["integrity_ok"] is True for flow in flows)
        exact = sum(flow["integrity_ok"] is True for flow in flows)
        returns = [flow["rl_return"] for flow in flows]
        discounted_returns = [flow["discounted_rl_return"] for flow in flows]
        attempts = sum(link["attempted_packets"] for link in links)
        drops_by_cause = {name: sum(link[name] for link in links)
                          for name in ("random_drops", "queue_drops", "outage_drops")}
        drops = sum(drops_by_cause.values())
        payload_sent = sum(flow["payload_sent_bytes"] for flow in flows)
        retries = sum(flow["retransmitted_payload_bytes"] for flow in flows)
        rtts = [sample for run in trials for sample in run.get("rtt_samples_seconds", [])]
        durations = [flow["elapsed_seconds"] for flow in flows if flow["completed"]]
        network_goodputs = [sum(flow["acknowledged_bytes"] for flow in run["flows"]) * 8
                            / max(run["elapsed_seconds"], 1e-9) for run in trials]
        summaries.append({
            "controller": controller, "trials": len(trials), "transfers": len(flows),
            "completed_transfers": completed, "completion_percent": 100 * completed / len(flows),
            "fully_completed_trials": sum(all(flow["completed"] and flow["integrity_ok"] is True
                                              for flow in run["flows"]) for run in trials),
            "exact_receiver_transfers": exact, "exact_receiver_percent": 100 * exact / len(flows),
            "wrong_payloads": sum(flow["integrity_ok"] is False for flow in flows),
            "timeouts": sum(not flow["completed"] for flow in flows),
            "attempted_packets": attempts, "dropped_packets": drops,
            "drop_percent": 100 * drops / attempts if attempts else 0.0,
            **drops_by_cause,
            "corrupted_packets": sum(link["corrupted_packets"] for link in links),
            "corruption_rejections": sum(link["corruption_rejections"] for link in links),
            "generated_duplicates": sum(link["generated_duplicates"] for link in links),
            "retransmitted_payload_bytes": retries,
            "payload_sent_bytes": payload_sent,
            "retransmission_percent": 100 * retries / payload_sent if payload_sent else 0.0,
            "wire_bytes": sum(flow["data_wire_bytes"] + flow["feedback_wire_bytes"] for flow in flows),
            "mean_network_goodput_bps": fmean(network_goodputs),
            "completion_p50_seconds": percentile(durations, .5),
            "completion_p95_seconds": percentile(durations, .95),
            "rtt_p50_seconds": percentile(rtts, .5), "rtt_p95_seconds": percentile(rtts, .95),
            "mean_rl_return": fmean(returns),
            "rl_return_std": stdev(returns) if len(returns) > 1 else 0.0,
            "mean_discounted_rl_return": fmean(discounted_returns),
            "discounted_rl_return_std": stdev(discounted_returns) if len(discounted_returns) > 1 else 0.0,
        })
    return summaries


def format_benchmark(report):
    settings = report["benchmark"]
    scoring = report["runs"][0]["reward"]
    lines = [
        f"{settings['trials']} paired trials/controller | {settings['flows']} flows/trial"
        f" | {settings['payload_bytes']:,} UTF-8 bytes/flow",
        f"Random loss: data {100 * settings['loss']:.1f}%, ACK {100 * settings['ack_loss']:.1f}%"
        f" | Bit corruption: data {100 * settings['corruption']:.1f}%, ACK {100 * settings['ack_corruption']:.1f}%",
        "",
        f"{'Controller':<13} {'Complete':>12} {'Exact RX':>9} {'Drops':>8} {'CRC reject':>10}"
        f" {'Retry':>8} {'Done p50':>10} {'RL score':>11}",
        "-" * 101,
    ]
    for summary in report["summary"]:
        duration = summary["completion_p50_seconds"]
        duration_text = f"{duration:.3f}s" if duration is not None else "--"
        complete = f"{summary['completed_transfers']}/{summary['transfers']} {summary['completion_percent']:.0f}%"
        lines.append(
            f"{summary['controller']:<13} {complete:>12} {summary['exact_receiver_percent']:>8.1f}%"
            f" {summary['drop_percent']:>7.1f}% {summary['corruption_rejections']:>10}"
            f" {summary['retransmission_percent']:>7.1f}% {duration_text:>10} {summary['mean_discounted_rl_return']:>11.3f}"
        )
    lines.append("")
    for summary in report["summary"]:
        rtt = summary["rtt_p95_seconds"]
        rtt_text = f"{rtt * 1000:.1f} ms" if rtt is not None else "--"
        lines.append(
            f"{summary['controller']}: {summary['fully_completed_trials']}/{summary['trials']} trials fully complete;"
            f" {summary['timeouts']} timeouts; {summary['wrong_payloads']} wrong payloads;"
            f" RTT p95 {rtt_text}; goodput {summary['mean_network_goodput_bps'] / 1000:.1f} kbit/s"
        )
        lines.append(
            f"  Drops: random={summary['random_drops']}, queue={summary['queue_drops']},"
            f" outage={summary['outage_drops']}; corruption injected={summary['corrupted_packets']},"
            f" rejected={summary['corruption_rejections']}; RL score SD={summary['discounted_rl_return_std']:.3f}"
        )
    lines.extend([
        "",
        f"RL score = mean discounted episode return ({scoring['version']}, gamma={scoring['discount_gamma']}); higher is better.",
        "JSON also includes undiscounted RL return and per-transfer scores; this is not a completion percentage.",
        "Completion requires exact bytes + sender confirmation; Exact RX measures receiver reconstruction.",
        "Drops count both paths; Retry is retransmitted / all transmitted payload bytes.",
        "Done p50 includes successful transfers only; incomplete transfers remain in the RL score.",
    ])
    return "\n".join(lines)
