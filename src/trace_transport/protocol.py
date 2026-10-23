"""Transport state machines, independent of simulation and socket I/O.

Packets carry one independently acknowledged byte range. Timers use seconds from
a monotonic clock supplied by the caller. Late and duplicate ACKs are harmless.
"""

from dataclasses import dataclass, field
import hashlib
import math

from .wire import ACK, DATA, MAX_PAYLOAD, ACK_BATCH, Packet

MAX_MESSAGE_BYTES = 16 * 1024 * 1024
RATE_MIN = 2_000.0
RATE_MAX = 100_000_000.0
WINDOW_MIN = 2_048.0
WINDOW_MAX = 4 * 1024 * 1024.0


@dataclass
class Segment:
    offset: int
    payload: bytes
    acknowledged: bool = False
    attempts: int = 0
    transmission_id: int | None = None
    sent_at: float = 0.0
    timeout_at: float = math.inf
    wire_bytes: int = 0
    transmission_ids: set[int] = field(default_factory=set)


class Sender:
    def __init__(self, payload: bytes, message_id: int, *, payload_size: int = 512,
                 rate: float = 128_000, cwnd: float = 16_384,
                 max_message_bytes: int = MAX_MESSAGE_BYTES):
        if not 1 <= payload_size <= MAX_PAYLOAD:
            raise ValueError(f"payload_size must be between 1 and {MAX_PAYLOAD}")
        if len(payload) > max_message_bytes:
            raise ValueError("message exceeds configured receive capacity; chunk it first")
        if not math.isfinite(rate) or not math.isfinite(cwnd) or rate <= 0 or cwnd <= 0:
            raise ValueError("rate and cwnd must be finite and positive")
        self.payload = payload
        self.expected_digest = hashlib.sha256(payload).hexdigest()
        self.verified_completion = False
        self.message_id = message_id
        self.rate = float(min(RATE_MAX, max(RATE_MIN, rate)))
        self.cwnd = float(min(WINDOW_MAX, max(WINDOW_MIN, cwnd)))
        self.receiver_credit = max_message_bytes
        offsets = range(0, len(payload), payload_size) if payload else [0]
        self.segments = {i: Segment(i, payload[i:i + payload_size]) for i in offsets}
        self.next_send_at = 0.0
        self.next_transmission_id = 1
        self.in_flight: dict[int, Segment] = {}
        self.flight_bytes = 0
        self.srtt = 0.0
        self.rttvar = 0.0
        self.min_rtt = 0.0
        self.rtt_samples: list[float] = []
        self.last_ack_at: float | None = None
        self.completed_at: float | None = None
        self.previous_action = (0.0, 0.0)
        self.wire_sent = 0
        self.payload_sent = 0
        self.retransmitted_bytes = 0
        self.declared_lost_bytes = 0
        self.acknowledged_bytes = 0
        self.peer_metadata: dict = {}

    @property
    def complete(self) -> bool:
        return self.completed_at is not None

    @property
    def rto(self) -> float:
        return min(3.0, max(0.1, self.srtt + 4 * self.rttvar)) if self.srtt else 0.3

    @property
    def unsent_bytes(self) -> int:
        return sum(len(s.payload) for s in self.segments.values() if not s.attempts)

    def apply_action(self, action) -> None:
        values = tuple(float(x) for x in action)
        if len(values) not in (1, 2) or not all(math.isfinite(x) for x in values):
            raise ValueError("actions must contain one or two finite numbers")
        a = max(-1.0, min(1.0, values[0]))
        b = max(-1.0, min(1.0, values[1])) if len(values) == 2 else 0.0
        self.rate = min(RATE_MAX, max(RATE_MIN, self.rate * math.exp(0.2 * a)))
        self.cwnd = min(WINDOW_MAX, max(WINDOW_MIN, self.cwnd * math.exp(0.2 * b)))
        self.previous_action = (a, b)

    def _expire(self, now: float) -> None:
        for tx, segment in list(self.in_flight.items()):
            if segment.timeout_at <= now + 1e-10:
                del self.in_flight[tx]
                self.flight_bytes -= segment.wire_bytes
                segment.transmission_id = None
                self.declared_lost_bytes += len(segment.payload)

    def _eligible(self) -> Segment | None:
        # Retransmissions take priority over application data. One segment can
        # probe an otherwise empty reduced window, preventing window deadlock.
        retry = None
        fresh = None
        for segment in self.segments.values():
            if segment.acknowledged or segment.transmission_id is not None:
                continue
            if segment.attempts and retry is None:
                retry = segment
            elif not segment.attempts and fresh is None:
                fresh = segment
        segment = retry or fresh
        if segment is None:
            return None
        estimated_wire = len(segment.payload) + 64  # exact size checked in poll
        if self.flight_bytes and self.flight_bytes + estimated_wire > self.cwnd:
            return None
        outstanding_payload = sum(len(s.payload) for s in self.in_flight.values())
        if not segment.attempts and outstanding_payload + len(segment.payload) > self.receiver_credit:
            return None
        return segment

    def next_wakeup(self, now: float) -> float:
        if self.complete:
            return math.inf
        timeout = min((s.timeout_at for s in self.in_flight.values()), default=math.inf)
        ready = max(now, self.next_send_at) if self._eligible() is not None else math.inf
        return min(ready, timeout)

    def poll(self, now: float) -> Packet | None:
        self._expire(now)
        if self.complete or now + 1e-10 < self.next_send_at:
            return None
        segment = self._eligible()
        if segment is None:
            return None
        tx = self.next_transmission_id
        packet = Packet(DATA, self.message_id, tx, segment.offset, len(self.payload), segment.payload)
        wire_bytes = len(packet.encode())
        if self.flight_bytes and self.flight_bytes + wire_bytes > self.cwnd:
            return None
        self.next_transmission_id += 1
        if segment.attempts:
            self.retransmitted_bytes += len(segment.payload)
        segment.attempts += 1
        segment.transmission_id = tx
        segment.transmission_ids.add(tx)
        segment.sent_at = now
        segment.timeout_at = now + min(3.0, self.rto * 2 ** min(segment.attempts - 1, 4))
        segment.wire_bytes = wire_bytes
        self.in_flight[tx] = segment
        self.flight_bytes += wire_bytes
        self.next_send_at = now + wire_bytes / self.rate
        self.wire_sent += wire_bytes
        self.payload_sent += len(segment.payload)
        return packet

    def on_ack(self, packet: Packet, now: float) -> bool:
        if packet.kind == ACK_BATCH:
            entries = packet.metadata.get("acks")
            if not isinstance(entries, list) or not 2 <= len(entries) <= 8:
                return False
            # Validate the entire envelope before changing sender state.
            for entry in entries:
                if not isinstance(entry, list) or len(entry) != 4:
                    return False
                tx, offset, length, delay = entry
                if any(type(x) is not int or x < 0 for x in (tx, offset, length)):
                    return False
                if type(delay) not in (int, float) or not math.isfinite(delay) or not 0 <= delay <= .025:
                    return False
            accepted = False
            for tx, offset, length, delay in entries:
                metadata = {k: v for k, v in packet.metadata.items() if k != "acks"}
                metadata.update(length=length, ack_delay=delay)
                accepted |= self.on_ack(Packet(ACK, packet.message_id, tx, offset, packet.total,
                                               metadata=metadata), now)
            return accepted
        if packet.kind != ACK or packet.message_id != self.message_id or packet.total != len(self.payload):
            return False
        segment = self.segments.get(packet.offset)
        if segment is None or segment.acknowledged or not segment.attempts:
            return False
        if type(packet.metadata.get("length")) is not int or packet.metadata["length"] != len(segment.payload):
            return False
        # ACKs may refer to an earlier transmission of the same byte range.
        if packet.transmission_id not in segment.transmission_ids:
            return False
        if packet.metadata.get("complete") is True:
            if packet.metadata.get("sha256") != self.expected_digest:
                return False
            self.verified_completion = True
        if packet.transmission_id == segment.transmission_id:
            delay = packet.metadata.get("ack_delay", 0.)
            if type(delay) not in (int, float) or not math.isfinite(delay) or not 0 <= delay <= .025:
                delay = 0.
            sample = max(1e-6, now - segment.sent_at - delay)
            self.rtt_samples.append(sample)
            self.min_rtt = min(self.min_rtt, sample) if self.min_rtt else sample
            if self.srtt:
                self.rttvar = 0.75 * self.rttvar + 0.25 * abs(self.srtt - sample)
                self.srtt = 0.875 * self.srtt + 0.125 * sample
            else:
                self.srtt, self.rttvar = sample, sample / 2
        if segment.transmission_id in self.in_flight:
            del self.in_flight[segment.transmission_id]
            self.flight_bytes -= segment.wire_bytes
        segment.transmission_id = None
        segment.transmission_ids.clear()
        segment.acknowledged = True
        self.acknowledged_bytes += len(segment.payload)
        self.last_ack_at = now
        self.peer_metadata = packet.metadata.copy()
        credit = packet.metadata.get("credit")
        if type(credit) is int and credit >= 0:
            self.receiver_credit = credit
        if self.verified_completion and all(s.acknowledged for s in self.segments.values()):
            self.completed_at = now
        return True


class Receiver:
    def __init__(self, *, max_message_bytes: int = MAX_MESSAGE_BYTES, metadata: dict | None = None):
        self.max_message_bytes = max_message_bytes
        self.custom_metadata = metadata or {}
        if not isinstance(self.custom_metadata, dict):
            raise ValueError("custom metadata must be a JSON object")
        if {"length", "credit", "received", "complete", "sha256"} & self.custom_metadata.keys():
            raise ValueError("custom metadata cannot override reliability fields")
        Packet(ACK, 0, 0, 0, 0, metadata={
            **self.custom_metadata, "length": MAX_PAYLOAD, "credit": max_message_bytes,
            "received": max_message_bytes, "complete": True, "sha256": "0" * 64,
        }).encode()
        self.message_id: int | None = None
        self.total: int | None = None
        self.chunks: dict[int, bytes] = {}
        self.received_bytes = 0
        self.completed_at: float | None = None

    @property
    def complete(self) -> bool:
        return self.completed_at is not None

    def on_data(self, packet: Packet, now: float) -> Packet | None:
        if packet.kind != DATA or packet.total > self.max_message_bytes:
            return None
        if packet.offset + len(packet.payload) > packet.total:
            return None
        if packet.total and not packet.payload:
            return None
        if self.message_id is not None and (packet.message_id != self.message_id or packet.total != self.total):
            return None
        if packet.offset in self.chunks:
            if self.chunks[packet.offset] != packet.payload:
                return None
        else:
            end = packet.offset + len(packet.payload)
            if any(packet.offset < off + len(data) and off < end for off, data in self.chunks.items()):
                return None
            self.message_id, self.total = packet.message_id, packet.total
            self.chunks[packet.offset] = packet.payload
            self.received_bytes += len(packet.payload)
            if self.received_bytes == self.total:
                self.completed_at = now
        metadata = {
            **self.custom_metadata, "length": len(packet.payload),
            "credit": self.max_message_bytes, "received": self.received_bytes,
            "complete": self.complete,
        }
        if self.complete:
            metadata["sha256"] = hashlib.sha256(self.reconstruct()).hexdigest()
        ack = Packet(ACK, packet.message_id, packet.transmission_id, packet.offset, packet.total, metadata=metadata)
        ack.encode()  # Reject an oversized metadata extension at the source.
        return ack

    def reconstruct(self) -> bytes:
        if not self.complete:
            raise RuntimeError("message is incomplete")
        return b"".join(data for _, data in sorted(self.chunks.items()))
