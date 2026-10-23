from dataclasses import replace

import pytest

from trace_transport.feedback import FeedbackScheduler
from trace_transport.protocol import Receiver, Sender
from trace_transport.wire import ACK_RECEIPT, Packet


def test_one_completion_receipt_recovers_all_lost_feedback():
    sender, receiver = Sender(b'abcdefghi', 7, payload_size=3), Receiver(receipt_feedback=True)
    packets = [sender.poll(t) for t in [0, .01, .02]]
    for i, packet in enumerate(packets):
        ack = receiver.on_data(packet, .1+i*.01)
    assert ack.kind == ACK_RECEIPT and ack.metadata['prefix'] == 9
    assert sender.on_ack(Packet.decode(ack.encode()), .2)
    assert sender.complete and sender.acknowledged_bytes == 9
    assert sender.flight_bytes == 0 and not sender.in_flight
    assert len(sender.rtt_samples) == 1  # no invented timing for summarized bytes


def test_selective_receipt_repeats_prior_out_of_order_ranges():
    sender, receiver = Sender(b'abcdefghi', 7, payload_size=3), Receiver(receipt_feedback=True)
    first, second, last = [sender.poll(t) for t in [0, .01, .02]]
    receiver.on_data(last, .05)  # lose its feedback
    ack = receiver.on_data(second, .06)
    assert ack.metadata['prefix'] == 0 and ack.metadata['ranges'] == [[3, 9]]
    assert sender.on_ack(ack, .1)
    assert sender.acknowledged_bytes == 6
    assert not sender.complete
    assert sender.on_ack(receiver.on_data(first, .11), .15)
    assert sender.complete


def test_receipt_cannot_ack_unsent_data_or_fake_completion():
    sender, receiver = Sender(b'abcdef', 7, payload_size=3), Receiver(receipt_feedback=True)
    ack = receiver.on_data(sender.poll(0), .02)
    forged = replace(ack, metadata={**ack.metadata, 'prefix': 6})
    assert not sender.on_ack(forged, .05)
    assert sender.acknowledged_bytes == 0
    assert not sender.on_ack(replace(ack, metadata={**ack.metadata, 'complete': True,'sha256':'wrong'}), .05)
    assert sender.acknowledged_bytes == 0


def test_completion_retry_is_paced_and_confirms_all_bytes():
    sender, receiver = Sender(b'abcdef', 7, payload_size=3), Receiver(receipt_feedback=True)
    emitted, timers = [], []
    scheduler = FeedbackScheduler(emitted.append, lambda *args:timers.append(args))
    scheduler.batch_size, scheduler.delay = 8, .025
    for t in [0,.01]:
        scheduler.on_ack(receiver.on_data(sender.poll(t), t+.02), t+.02)
    assert len(emitted) == 1  # lose this completion receipt
    retry = sender.poll(.4)
    scheduler.on_ack(receiver.on_data(retry, .42), .42)
    assert len(emitted) == 1  # completed receiver does not ACK every duplicate immediately
    time, callback, generation = timers[-1]
    callback(generation, time)
    assert len(emitted) == 2
    assert sender.on_ack(emitted[-1], .5) and sender.complete


@pytest.mark.parametrize('bad_ranges', [[[2,1]], [[0,99]], [[True,3]], [[1,3],[2,4]], 'bad'])
def test_bad_summary_is_rejected_atomically(bad_ranges):
    sender, receiver = Sender(b'abcdef', 7, payload_size=3), Receiver(receipt_feedback=True)
    ack = receiver.on_data(sender.poll(0), .02)
    assert not sender.on_ack(replace(ack,metadata={**ack.metadata,'ranges':bad_ranges}),.04)
    assert sender.acknowledged_bytes == 0


def test_scheduler_uses_latest_summary_even_when_duplicate_keeps_dict_position():
    sender, receiver = Sender(b'abcdefghi',7,payload_size=3),Receiver(receipt_feedback=True)
    first,second,last=[sender.poll(t) for t in [0,.01,.02]]
    emitted=[]
    scheduler=FeedbackScheduler(emitted.append,lambda *args:None)
    scheduler.batch_size,scheduler.delay=8,.025
    scheduler.on_ack(receiver.on_data(first,.1),.1)
    scheduler.on_ack(receiver.on_data(second,.11),.11)
    # Update an existing pending entry with a newer summary.
    scheduler.on_ack(receiver.on_data(first,.12),.12)
    scheduler.flush(.13)
    assert emitted[-1].offset==first.offset
    assert emitted[-1].metadata['prefix']==6


def test_receipt_summary_reserves_scheduler_delay_budget():
    from trace_transport.wire import DATA
    receiver=Receiver(receipt_feedback=True,metadata={'note':'x'*270})
    packets=[Packet(DATA,7,i+1,1000001+i*2,16000000,b'x') for i in range(8)]
    emitted=[]
    scheduler=FeedbackScheduler(emitted.append,lambda *args:None)
    scheduler.batch_size,scheduler.delay=8,.025
    for i in range(8):
        scheduler.on_ack(receiver.on_data(packets[i],1+i*.00001),1+i*.00001)
    assert emitted
    emitted[-1].encode()  # sparse ranges plus metadata plus ACK delay all fit
    assert len(emitted[-1].metadata['ranges'])<8


def test_duplicate_receipt_cannot_move_first_confirmation_timestamp():
    sender=Sender(b'abc',7);receiver=Receiver(receipt_feedback=True)
    ack=receiver.on_data(sender.poll(0),.05)
    assert sender.on_ack(ack,.1)
    assert sender.completed_at==.1
    sender.on_ack(ack,.2)
    assert sender.completed_at==.1
