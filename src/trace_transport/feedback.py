"""Bounded receiver ACK batching; observations contain only receiver-local state.

ACK_BATCH is an experimental capability, currently supported by the simulator.
The deterministic reliability kernel still decides which bytes were received.
"""
from dataclasses import replace
import math

from .wire import ACK_BATCH, ACK_RECEIPT, Packet

MAX_ACK_DELAY = .025
MAX_ACK_BATCH = 8
FEEDBACK_PROFILES = {'legacy': None, 'budgeted': .025, 'sparse': .25}


def profile_config(config, name):
    if not isinstance(name,str) or name not in FEEDBACK_PROFILES:
        raise ValueError('unknown feedback profile')
    return replace(config, feedback_delay_budget=.25) if name != 'legacy' else config



class FeedbackScheduler:
    def __init__(self, emit, schedule, *, max_delay=MAX_ACK_DELAY):
        if not math.isfinite(max_delay) or not 0 <= max_delay <= 1:
            raise ValueError("invalid feedback delay limit")
        self.max_delay=max_delay
        self.emit, self.schedule = emit, schedule
        self.pending = {}
        self.batch_size, self.delay = 1, 0.0
        self.generation = 0
        self.completion_seen = False
        self.emitted = 0
        self.last_emit = 0.0
        self.arrivals = 0
        self.duplicates = 0
        self.last_arrival = 0.0
        self.interarrival = 0.0
        self.previous_offset = 0
        self.reordered = 0

    def set_action(self, action):
        if len(action) != 2 or not all(math.isfinite(float(x)) for x in action):
            raise ValueError('receiver action requires two finite values')
        a, b = (max(-1., min(1., float(x))) for x in action)
        self.batch_size = (1, 2, 4, 8)[min(3, int((a + 1) * 2))]
        self.delay = (b + 1) * self.max_delay / 2

    def on_ack(self, ack, now):
        self.arrivals += 1
        self.interarrival = now - self.last_arrival if self.arrivals > 1 else 0.
        self.last_arrival = now
        self.reordered += ack.offset < self.previous_offset
        self.previous_offset = ack.offset
        self.duplicates += ack.offset in self.pending
        # A new retransmission supersedes older pending feedback for this range.
        self.pending[ack.offset] = (ack, now)
        first_completion = ack.metadata['complete'] and (ack.kind != ACK_RECEIPT or not self.completion_seen)
        self.completion_seen |= ack.metadata['complete']
        if len(self.pending) >= self.batch_size or self.delay == 0 or first_completion:
            self.flush(now)
        elif len(self.pending) == 1:
            generation = self.generation
            self.schedule(now + self.delay, self._timer, generation)

    def _timer(self, generation, now=None):
        # The caller supplies the current simulated monotonic time.
        if generation == self.generation:
            self.flush(now)

    def flush(self, now):
        if not self.pending:
            return
        items = list(self.pending.values())
        self.pending.clear()
        self.generation += 1
        if items[-1][0].kind == ACK_RECEIPT:
            # The newest summary repeats the cumulative prefix and selective ranges.
            # Completion therefore confirms the entire message in one bounded packet.
            self._send([max(items, key=lambda item: item[1])], now)
            return
        # Split to respect the actual encoded metadata budget, including user fields.
        group = []
        for item in items:
            try:
                self._packet(group + [item], now).encode()
            except ValueError:
                if not group:
                    raise
                self._send(group, now)
                group = []
            group.append(item)
        if group:
            self._send(group, now)

    def _packet(self, items, now):
        ack = items[-1][0]
        metadata = dict(ack.metadata)
        metadata['ack_delay'] = min(self.max_delay, max(0., now - items[-1][1]))
        if len(items) == 1:
            return replace(ack, metadata=metadata)
        metadata['acks'] = [[a.transmission_id, a.offset, a.metadata['length'],
                             min(self.max_delay, max(0., now - time))] for a, time in items]
        return replace(ack, kind=ACK_BATCH, metadata=metadata)

    def _send(self, items, now):
        self.emit(self._packet(items, now))
        self.emitted += 1
        self.last_emit = now
