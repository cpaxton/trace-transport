"""Single-message UDP adapters using the same packet and transport state machines.

The listener keeps completed state for a finite linger interval so loss of final
feedback can be repaired. Session multiplexing and durable restart are future work.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import secrets
import select
import socket
import time

from .controllers import make_controller
from .environment import sender_observation, sender_snapshot
from .protocol import Receiver, Sender
from .simulator import TransferResult
from .wire import Packet


def send_udp(text: str, host: str, port: int, *, deadline=20.0, controller="aimd",
             checkpoint=None, device="cpu") -> TransferResult:
    if not math.isfinite(deadline) or deadline <= 0:
        raise ValueError("deadline must be finite and positive")
    policy = make_controller(controller, 1, checkpoint=checkpoint, device=device)
    sender = Sender(text.encode("utf-8"), secrets.randbits(64))
    previous = sender_snapshot(sender)
    observations = sender_observation(sender, 0, 0, previous, .02)[None]
    policy.reset(observations)
    feedback_bytes = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, ThreadPoolExecutor(max_workers=1) as inference:
        sock.connect((host, port))
        sock.setblocking(False)
        started = time.monotonic()
        next_control = 0.0
        last_control = 0.0
        pending_action = None
        while not sender.complete:
            now = time.monotonic() - started
            if now >= deadline:
                break
            # Send independently of the 20ms policy interval: pacing uses actual
            # monotonic time, not one burst per control decision.
            packet = sender.poll(now)
            if packet is not None:
                try:
                    sock.send(packet.encode())
                except (BlockingIOError, ConnectionRefusedError):
                    pass  # Treat an unavailable datagram as loss; timers retry.
            if now >= next_control:
                if pending_action is not None and pending_action.done():
                    sender.apply_action(pending_action.result()[0])
                    pending_action = None
                if pending_action is None:
                    observations = sender_observation(sender, now, 0, previous, max(.02, now - last_control))[None]
                    previous = sender_snapshot(sender, feedback_bytes)
                    pending_action = inference.submit(policy.act, observations)
                    last_control = now
                next_control = now + .02
            wakeup = min(sender.next_wakeup(now), next_control, deadline)
            readable, _, _ = select.select([sock], [], [], max(0.0, wakeup - (time.monotonic() - started)))
            if readable:
                while True:
                    try:
                        wire = sock.recv(4096)
                    except (BlockingIOError, ConnectionRefusedError):
                        break
                    feedback_bytes += len(wire)
                    try:
                        ack = Packet.decode(wire)
                    except ValueError:
                        continue
                    sender.on_ack(ack, time.monotonic() - started)
    elapsed = sender.completed_at if sender.complete else min(deadline, time.monotonic() - started)
    return TransferResult(
        sent_text=text, received_text=None, completed=sender.complete,
        receiver_complete=sender.verified_completion, elapsed_seconds=elapsed,
        receiver_completed_seconds=None, payload_bytes=len(sender.payload),
        acknowledged_bytes=sender.acknowledged_bytes, data_wire_bytes=sender.wire_sent,
        feedback_wire_bytes=feedback_bytes, retransmitted_payload_bytes=sender.retransmitted_bytes,
        rtt_samples=tuple(sender.rtt_samples),
        payload_sent_bytes=sender.payload_sent,
    )


def receive_udp(host: str, port: int, output: Path, *, timeout=30.0, linger=10.0,
                metadata=None, ready=None) -> dict:
    if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(linger) or linger < 0:
        raise ValueError("timeout must be positive and linger nonnegative, both finite")
    receiver = Receiver(metadata=metadata)
    feedback_bytes = 0
    peer = None
    saved = False
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((host, port))
        if ready is not None:
            ready(sock.getsockname()[1])
        started = time.monotonic()
        end = timeout
        while True:
            now = time.monotonic() - started
            if now >= end:
                break
            sock.settimeout(max(1e-6, end - now))
            try:
                wire, address = sock.recvfrom(4096)
            except socket.timeout:
                break
            if peer is not None and address != peer:
                continue
            try:
                packet = Packet.decode(wire)
                ack = receiver.on_data(packet, time.monotonic() - started)
            except ValueError:
                continue
            if ack is None:
                continue
            peer = address
            if receiver.complete and not saved:
                payload = receiver.reconstruct()
                payload.decode("utf-8")  # The application accepts UTF-8 text.
                output = Path(output)
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(output.name + ".partial")
                temporary.write_bytes(payload)
                temporary.replace(output)
                saved = True
                end = min(timeout, time.monotonic() - started + linger)
            encoded = ack.encode()
            sock.sendto(encoded, address)
            feedback_bytes += len(encoded)
    return {"completed": saved, "payload_bytes": receiver.received_bytes,
            "feedback_wire_bytes": feedback_bytes, "output": str(output) if saved else None}
