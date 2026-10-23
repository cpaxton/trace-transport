"""Seeded discrete-event packet simulation with shared finite FIFO links."""

from collections import deque
from dataclasses import asdict, dataclass
import heapq
import math
import random
from typing import Callable

from .protocol import MAX_MESSAGE_BYTES, Receiver, Sender
from .wire import Packet


@dataclass(frozen=True)
class NetworkConfig:
    bandwidth_bps: float = 1_000_000
    feedback_bandwidth_bps: float = 1_000_000
    base_rtt: float = 0.05
    jitter: float = 0.005
    loss: float = 0.02
    ack_loss: float = 0.02
    corruption: float = 0.0
    ack_corruption: float = 0.0
    duplicate: float = 0.0
    queue_bytes: int = 16_384
    feedback_queue_bytes: int = 16_384
    payload_size: int = 512
    initial_rate: float = 128_000
    initial_cwnd: float = 16_384
    feedback_delay_budget: float = 0.0
    control_interval: float = 0.02
    deadline: float = 10.0
    max_message_bytes: int = MAX_MESSAGE_BYTES
    # (absolute simulation time, new forward-link bits/s).
    capacity_changes: tuple[tuple[float, float], ...] = ()
    feedback_capacity_changes: tuple[tuple[float, float], ...] = ()
    loss_changes: tuple[tuple[float, float, float], ...] = ()
    rtt_changes: tuple[tuple[float, float], ...] = ()
    # Bidirectional periods of no delivery; endpoint state remains intact.
    outages: tuple[tuple[float, float], ...] = ()

    def __post_init__(self):
        for name in ("bandwidth_bps", "feedback_bandwidth_bps", "initial_rate", "initial_cwnd",
                     "control_interval", "deadline"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.feedback_delay_budget) or not 0 <= self.feedback_delay_budget <= 1:
            raise ValueError("feedback delay budget must lie in [0,1] seconds")
        for name in ("base_rtt", "jitter"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("loss", "ack_loss", "corruption", "ack_corruption", "duplicate"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be a probability")
        if not 1 <= self.payload_size <= 1200:
            raise ValueError("payload_size must be in [1, 1200]")
        for name in ("queue_bytes", "feedback_queue_bytes", "max_message_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for changes in (self.capacity_changes, self.feedback_capacity_changes):
            last = -1.0
            for time, rate in changes:
                if not math.isfinite(time) or time < 0 or time <= last or not math.isfinite(rate) or rate <= 0:
                    raise ValueError("capacity changes require increasing times and positive rates")
                last = time
        last = -1.
        for time, forward_loss, reverse_loss in self.loss_changes:
            if not math.isfinite(time) or time < 0 or time <= last or not 0 <= forward_loss <= 1 or not 0 <= reverse_loss <= 1:
                raise ValueError("loss changes require increasing times and valid probabilities")
            last = time
        last = -1.
        for time, rtt in self.rtt_changes:
            if not math.isfinite(time) or time < 0 or time <= last or not math.isfinite(rtt) or rtt < 0:
                raise ValueError("RTT changes require increasing times and nonnegative delays")
            last = time
        for start, end in self.outages:
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError("outages require finite 0 <= start < end")


@dataclass
class LinkCounters:
    attempted_packets: int = 0
    admitted_packets: int = 0
    attempted_bytes: int = 0
    admitted_bytes: int = 0
    queue_drops: int = 0
    random_drops: int = 0
    outage_drops: int = 0
    delivered_packets: int = 0
    corrupted_packets: int = 0
    corruption_rejections: int = 0
    generated_duplicates: int = 0


class FifoLink:
    def __init__(self, rate: float, capacity: int, changes=()):
        self.rate = rate
        self.capacity = capacity
        self.changes = changes
        self.departures: deque[tuple[float, int]] = deque()
        self.occupancy = 0
        self.next_free = 0.0
        self.counters = LinkCounters()

    def _finish(self, start: float, bits: int) -> float:
        # Integrate serialization through capacity changes, including a change
        # while a packet is in service. Only the network sees this schedule.
        rate = self.rate
        time = start
        remaining = float(bits)
        for change_time, new_rate in self.changes:
            if change_time <= time:
                rate = new_rate
                continue
            possible = (change_time - time) * rate
            if remaining <= possible:
                return time + remaining / rate
            remaining -= possible
            time, rate = change_time, new_rate
        return time + remaining / rate

    def enqueue(self, now: float, size: int) -> float | None:
        self.counters.attempted_packets += 1
        self.counters.attempted_bytes += size
        while self.departures and self.departures[0][0] <= now + 1e-10:
            _, departed_size = self.departures.popleft()
            self.occupancy -= departed_size
        # Capacity explicitly includes the packet currently in service.
        if self.occupancy + size > self.capacity:
            self.counters.queue_drops += 1
            return None
        finish = self._finish(max(now, self.next_free), size * 8)
        self.next_free = finish
        self.occupancy += size
        self.departures.append((finish, size))
        self.counters.admitted_bytes += size
        self.counters.admitted_packets += 1
        return finish


@dataclass
class TransferResult:
    sent_text: str
    received_text: str | None
    completed: bool
    receiver_complete: bool
    elapsed_seconds: float
    receiver_completed_seconds: float | None
    payload_bytes: int
    acknowledged_bytes: int
    data_wire_bytes: int
    feedback_wire_bytes: int
    retransmitted_payload_bytes: int
    rtt_samples: tuple[float, ...]
    payload_sent_bytes: int = 0

    @property
    def goodput_bps(self) -> float:
        return self.acknowledged_bytes * 8 / max(self.elapsed_seconds, 1e-9)

    def summary(self) -> dict:
        result = asdict(self)
        result.pop("sent_text")
        result.pop("received_text")
        samples = sorted(result.pop("rtt_samples"))
        result["goodput_bps"] = self.goodput_bps
        result["rtt_p50_seconds"] = samples[len(samples) // 2] if samples else None
        result["rtt_p95_seconds"] = samples[min(len(samples) - 1, int(len(samples) * .95))] if samples else None
        return result


class PacketSimulator:
    def __init__(self, texts: list[str], config: NetworkConfig | None = None, *, seed: int = 0,
                 start_times: list[float] | None = None, receiver_metadata: dict | None = None,
                 receiver_feedback: bool = False, receipt_feedback: bool = False):
        if not texts:
            raise ValueError("at least one text transfer is required")
        self.config = config or NetworkConfig()
        self.texts = texts.copy()
        self.random = random.Random(seed)
        self.now = 0.0
        self._sequence = 0
        self._events: list[tuple[float, int, Callable, tuple]] = []
        self.starts = [0.0] * len(texts) if start_times is None else list(start_times)
        if len(self.starts) != len(texts) or any(not math.isfinite(t) or t < 0 for t in self.starts):
            raise ValueError("one finite, nonnegative start time is required per flow")
        cfg = self.config
        self.senders = [Sender(text.encode("utf-8"), self.random.getrandbits(64),
                               payload_size=cfg.payload_size, rate=cfg.initial_rate,
                               cwnd=cfg.initial_cwnd, max_message_bytes=cfg.max_message_bytes,
                               feedback_delay_budget=cfg.feedback_delay_budget)
                        for text in texts]
        self.receivers = [Receiver(max_message_bytes=cfg.max_message_bytes, metadata=receiver_metadata,
                                   receipt_feedback=receipt_feedback)
                          for _ in texts]
        # Separate streams keep impairment draws independent of message IDs.
        self.forward_random = random.Random(seed * 2 + 1)
        self.feedback_random = random.Random(seed * 2 + 2)
        self.forward = FifoLink(cfg.bandwidth_bps, cfg.queue_bytes, cfg.capacity_changes)
        self.feedback = FifoLink(cfg.feedback_bandwidth_bps, cfg.feedback_queue_bytes, cfg.feedback_capacity_changes)
        self.feedback_bytes = [0] * len(texts)
        self._wake_generation = [0] * len(texts)
        self.processed_events = 0
        self.feedback_schedulers = []
        if receiver_feedback:
            from .feedback import FeedbackScheduler
            for index in range(len(texts)):
                def schedule(time, callback, generation):
                    self._schedule(time, lambda: callback(generation, self.now))
                self.feedback_schedulers.append(FeedbackScheduler(
                    lambda ack, i=index: self._transmit(i, ack, feedback=True), schedule,
                    max_delay=max(.025,cfg.feedback_delay_budget)))
        for index in range(len(texts)):
            self._schedule_wake(index)

    @property
    def finished(self) -> bool:
        return all(sender.complete for sender in self.senders) or self.now >= self.config.deadline

    def _schedule(self, time: float, callback: Callable, *args) -> None:
        if time > self.config.deadline or not math.isfinite(time):
            return
        self._sequence += 1
        heapq.heappush(self._events, (max(self.now, time), self._sequence, callback, args))

    def _schedule_wake(self, index: int) -> None:
        self._wake_generation[index] += 1
        now = max(self.now, self.starts[index])
        time = self.senders[index].next_wakeup(now)
        self._schedule(time, self._wake, index, self._wake_generation[index])

    def _wake(self, index: int, generation: int) -> None:
        if generation != self._wake_generation[index]:
            return
        packet = self.senders[index].poll(self.now)
        if packet is not None:
            self._transmit(index, packet, feedback=False)
        self._schedule_wake(index)

    def _in_outage(self, now: float) -> bool:
        return any(start <= now < end for start, end in self.config.outages)

    def _transmit(self, index: int, packet: Packet, *, feedback: bool) -> None:
        wire = packet.encode()
        if feedback:
            self.feedback_bytes[index] += len(wire)
        link = self.feedback if feedback else self.forward
        rng = self.feedback_random if feedback else self.forward_random
        # A network-generated duplicate is independently serialized and charged.
        copies = 1 + int(rng.random() < self.config.duplicate)
        link.counters.generated_duplicates += copies - 1
        for _ in range(copies):
            finish = link.enqueue(self.now, len(wire))
            if finish is None:
                continue
            if self._in_outage(self.now):
                link.counters.outage_drops += 1
                continue
            loss = self.config.ack_loss if feedback else self.config.loss
            for time, data_loss, ack_loss in self.config.loss_changes:
                if time > self.now:
                    break
                loss = ack_loss if feedback else data_loss
            if rng.random() < loss:
                link.counters.random_drops += 1
                continue
            delivered_wire = wire
            corruption = self.config.ack_corruption if feedback else self.config.corruption
            if corruption and rng.random() < corruption:
                damaged = bytearray(wire)
                damaged[rng.randrange(len(wire))] ^= 1 << rng.randrange(8)
                delivered_wire = bytes(damaged)
                link.counters.corrupted_packets += 1
            base_rtt = self.config.base_rtt
            for time, rtt in self.config.rtt_changes:
                if time > self.now:
                    break
                base_rtt = rtt
            delay = max(0.0, base_rtt / 2 + rng.uniform(-self.config.jitter, self.config.jitter))
            self._schedule(finish + delay, self._deliver, index, delivered_wire, feedback)

    def _deliver(self, index: int, wire: bytes, feedback: bool) -> None:
        link = self.feedback if feedback else self.forward
        if self._in_outage(self.now):
            link.counters.outage_drops += 1
            return
        try:
            packet = Packet.decode(wire)
        except ValueError:
            link.counters.corruption_rejections += 1
            return
        link.counters.delivered_packets += 1
        if feedback:
            self.senders[index].on_ack(packet, self.now)
            self._schedule_wake(index)
        else:
            ack = self.receivers[index].on_data(packet, self.now)
            if ack is not None:
                if self.feedback_schedulers:
                    self.feedback_schedulers[index].on_ack(ack, self.now)
                else:
                    self._transmit(index, ack, feedback=True)

    def advance(self, actions, *, until: float | None = None) -> None:
        if self.finished:
            return
        if len(actions) != len(self.senders):
            raise ValueError("one action is required per sender")
        target = self.now + self.config.control_interval if until is None else until
        if not math.isfinite(target) or target <= self.now:
            raise ValueError("advance target must be finite and in the future")
        target = min(target, self.config.deadline)
        for index, (sender, action) in enumerate(zip(self.senders, actions)):
            if not sender.complete and self.now >= self.starts[index]:
                sender.apply_action(action)
                self._schedule_wake(index)
        while self._events and self._events[0][0] <= target:
            time, _, callback, args = heapq.heappop(self._events)
            self.now = time
            callback(*args)
            self.processed_events += 1
            if all(sender.complete for sender in self.senders):
                return
        self.now = target

    def results(self) -> list[TransferResult]:
        results = []
        for index, (text, sender, receiver) in enumerate(zip(self.texts, self.senders, self.receivers)):
            output = receiver.reconstruct() if receiver.complete else None
            if output is not None and output != text.encode("utf-8"):
                raise AssertionError("protocol produced incorrect application bytes")
            results.append(TransferResult(
                sent_text=text, received_text=output.decode("utf-8") if output is not None else None,
                completed=sender.complete and receiver.complete, receiver_complete=receiver.complete,
                elapsed_seconds=max(0.0, (sender.completed_at if sender.complete else self.now) - self.starts[index]),
                receiver_completed_seconds=(receiver.completed_at - self.starts[index]) if receiver.complete else None,
                payload_bytes=len(sender.payload), acknowledged_bytes=sender.acknowledged_bytes,
                data_wire_bytes=sender.wire_sent, feedback_wire_bytes=self.feedback_bytes[index],
                retransmitted_payload_bytes=sender.retransmitted_bytes,
                rtt_samples=tuple(sender.rtt_samples),
                payload_sent_bytes=sender.payload_sent,
            ))
        return results


def transfer_text(text: str, *, config: NetworkConfig | None = None, controller: str = "aimd",
                  seed: int = 0, checkpoint: str | None = None, device: str = "cpu") -> TransferResult:
    from .environment import TransportEnv
    from .controllers import make_controller

    policy = make_controller(controller, 1, checkpoint=checkpoint, device=device)
    env = TransportEnv([text], config=config, action_dimensions=policy.action_dimensions)
    observations, _ = env.reset(seed=seed)
    policy.reset(observations)
    while not env.simulator.finished:
        actions = policy.act(observations)
        observations, _, _, _, _ = env.step(actions)
    return env.simulator.results()[0]
