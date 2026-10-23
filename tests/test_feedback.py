import pytest

from trace_transport.feedback import FeedbackScheduler
from trace_transport.protocol import Receiver, Sender
from trace_transport.simulator import NetworkConfig, PacketSimulator
from trace_transport.wire import ACK_BATCH, Packet


def test_batch_ack_exact_ranges_and_delay_adjusted_rtt():
    sender, receiver = Sender(b'abcdef', 1, payload_size=3), Receiver()
    packets = [sender.poll(0), sender.poll(.01)]
    emitted, timers = [], []
    scheduler = FeedbackScheduler(emitted.append, lambda *args: timers.append(args))
    scheduler.batch_size, scheduler.delay = 4, .025
    scheduler.on_ack(receiver.on_data(packets[0], .02), .02)
    assert not emitted
    scheduler.on_ack(receiver.on_data(packets[1], .03), .03)
    assert len(emitted) == 1 and emitted[0].kind == ACK_BATCH
    ack = Packet.decode(emitted[0].encode())
    assert sender.on_ack(ack, .05)
    assert sender.complete
    assert sender.rtt_samples == pytest.approx([.04, .04])
    timers[0][1](timers[0][2], .045)
    assert len(emitted) == 1  # stale timer cannot send another ACK


def test_timer_flushes_partial_batch():
    emitted, timers = [], []
    scheduler = FeedbackScheduler(emitted.append, lambda *args: timers.append(args))
    scheduler.batch_size, scheduler.delay = 8, .025
    ack = Receiver().on_data(Sender(b'abcdef', 1, payload_size=3).poll(0), .02)
    scheduler.on_ack(ack, .02)
    time, callback, generation = timers[0]
    callback(generation, time)
    assert len(emitted) == 1
    assert emitted[0].metadata['ack_delay'] == pytest.approx(.025)


@pytest.mark.parametrize('batch', [1, 2, 4, 8])
def test_batched_feedback_exact_under_impairment(batch):
    text = 'arbitrary 🌐\r\n' * 512
    sim = PacketSimulator([text, text], NetworkConfig(loss=.1, ack_loss=.15, duplicate=.1,
                          corruption=.01, ack_corruption=.01, deadline=20),
                          seed=42, receiver_feedback=True)
    for feedback in sim.feedback_schedulers:
        feedback.batch_size, feedback.delay = batch, .025
    while not sim.finished:
        sim.advance([[0.], [0.]])
    assert all(r.completed and r.received_text == text for r in sim.results())


def test_malformed_batch_is_rejected_without_state_change():
    sender = Sender(b'abcdef', 1, payload_size=3)
    sender.poll(0)
    ack = Packet(ACK_BATCH, 1, 1, 0, 6, metadata={'acks': [[1, 0, 3, 0], [2, 3, 3, float('nan')]]})
    assert not sender.on_ack(ack, .1)
    assert sender.acknowledged_bytes == 0


def test_lost_completion_batch_recovers_on_duplicate():
    sender, receiver = Sender(b'abcdef', 1, payload_size=3), Receiver()
    emitted = []
    scheduler = FeedbackScheduler(emitted.append, lambda *args: None)
    scheduler.batch_size, scheduler.delay = 8, .025
    first, last = sender.poll(0), sender.poll(.01)
    scheduler.on_ack(receiver.on_data(first, .02), .02)
    scheduler.on_ack(receiver.on_data(last, .03), .03)
    assert len(emitted) == 1
    # Lose the entire final batch; both byte ranges must be recovered.
    for time in [.4, .5]:
        retry = sender.poll(time)
        assert retry is not None
        scheduler.on_ack(receiver.on_data(retry, time+.02), time+.02)
        sender.on_ack(emitted[-1], time+.04)
    assert sender.complete
    assert receiver.reconstruct() == b'abcdef'


def test_receiver_checkpoint_and_local_observation_schema(tmp_path):
    torch = pytest.importorskip('torch')
    from trace_transport.receiver_learning import RECEIVER_FEATURES, observations, make_model, save, load
    from trace_transport.learning import load_checkpoint
    sim = PacketSimulator(['hello'], receiver_feedback=True)
    before = observations(sim)
    assert before.shape == (1, len(RECEIVER_FEATURES)) == (1, 22)
    # Hidden network properties do not enter receiver observations.
    other = PacketSimulator(['hello'], NetworkConfig(bandwidth_bps=1234567, loss=.9), receiver_feedback=True)
    assert (before == observations(other)).all()
    path = tmp_path/'receiver.pt'
    model = make_model(42)
    save(path, model, {})
    assert load(path).config == model.config
    with pytest.raises(ValueError, match='incompatible'):
        load_checkpoint(path)
