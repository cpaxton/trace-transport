import pytest

from trace_transport.protocol import Receiver, Sender
from trace_transport.wire import ACK, DATA, Packet


def test_wire_roundtrip_and_metadata_budget():
    packet = Packet(DATA, 7, 1, 0, 7, b"payload", {"custom": "🌐"})
    assert Packet.decode(packet.encode()) == packet
    with pytest.raises(ValueError, match="budget"):
        Packet(DATA, 7, 1, 0, 0, metadata={"custom": "x" * 512}).encode()


def test_corruption_and_truncation_are_rejected():
    wire = bytearray(Packet(DATA, 1, 1, 0, 4, b"test").encode())
    wire[-5] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        Packet.decode(bytes(wire))
    with pytest.raises(ValueError, match="truncated"):
        Packet.decode(b"short")


@pytest.mark.parametrize("payload", [b"", b"hello", "こんにちは\r\n🌐".encode(), b"x" * 2048])
def test_reversed_delivery_and_duplicate_feedback(payload):
    sender, receiver = Sender(payload, 1, payload_size=256), Receiver()
    packets = []
    now = 0.0
    while sender.unsent_bytes or not packets:
        packet = sender.poll(now)
        if packet:
            packets.append(packet)
        now += .01
    for packet in reversed(packets):
        ack = receiver.on_data(packet, now)
        assert ack is not None
        sender.on_ack(ack, now + .01)
        assert not sender.on_ack(ack, now + .02)
        repeated = receiver.on_data(packet, now + .03)
        assert repeated is not None
    assert sender.complete
    assert receiver.reconstruct() == payload
    assert sender.acknowledged_bytes == len(payload)
    assert sender.flight_bytes == 0


def test_lost_ack_retransmission_and_late_original_ack():
    sender, receiver = Sender(b"hello", 1), Receiver()
    original = sender.poll(0)
    original_ack = receiver.on_data(original, .02)
    resent = sender.poll(.31)
    assert resent.transmission_id != original.transmission_id
    assert sender.on_ack(original_ack, .32)
    assert sender.complete
    assert sender.flight_bytes == 0
    assert receiver.reconstruct() == b"hello"
    assert not sender.on_ack(receiver.on_data(resent, .33), .34)


def test_overlapping_or_conflicting_ranges_are_rejected():
    receiver = Receiver()
    assert receiver.on_data(Packet(DATA, 1, 1, 0, 6, b"abc"), 0)
    assert receiver.on_data(Packet(DATA, 1, 2, 1, 6, b"bad"), .01) is None
    assert receiver.on_data(Packet(DATA, 1, 3, 0, 6, b"xyz"), .02) is None
    assert receiver.received_bytes == 3
    assert receiver.on_data(Packet(DATA, 1, 4, 3, 6, b"def"), .03)
    assert receiver.reconstruct() == b"abcdef"


def test_custom_feedback_cannot_override_reliability_or_exceed_budget():
    with pytest.raises(ValueError, match="override"):
        Receiver(metadata={"complete": True})
    with pytest.raises(ValueError, match="budget"):
        Receiver(metadata={"custom": "x" * 500})
    receiver = Receiver(metadata={"custom": [1, 2, 3]})
    ack = receiver.on_data(Packet(DATA, 1, 1, 0, 0), 0)
    assert ack.metadata["custom"] == [1, 2, 3]
    assert ack.metadata["complete"] is True


def test_wrong_completion_digest_is_not_accepted():
    sender, receiver = Sender(b"hello", 1), Receiver()
    packet = sender.poll(0)
    ack = receiver.on_data(packet, .01)
    broken = Packet(ACK, 1, packet.transmission_id, 0, 5,
                    metadata={**ack.metadata, "sha256": "0" * 64})
    assert not sender.on_ack(broken, .02)
    assert not sender.complete
    assert sender.on_ack(ack, .03)


def test_ack_cannot_reuse_another_ranges_transmission_id():
    sender, receiver = Sender(b"abcdef", 1, payload_size=3), Receiver()
    first = sender.poll(0)
    second = sender.poll(.01)
    ack = receiver.on_data(first, .02)
    wrong_range = Packet(ACK, 1, first.transmission_id, second.offset, 6,
                         metadata={**ack.metadata, "length": len(second.payload)})
    assert not sender.on_ack(wrong_range, .03)
    assert sender.acknowledged_bytes == 0
    assert sender.on_ack(ack, .04)
    assert not sender.complete
    assert sender.on_ack(receiver.on_data(second, .05), .06)
    assert sender.complete and not any(s.transmission_ids for s in sender.segments.values())


def test_message_and_action_limits():
    with pytest.raises(ValueError, match="capacity"):
        Sender(b"too long", 1, max_message_bytes=2)
    sender = Sender(b"x", 1)
    with pytest.raises(ValueError, match="finite"):
        sender.apply_action([float("nan")])
    for _ in range(1000):
        sender.apply_action([-1, -1])
    assert sender.rate > 0 and sender.cwnd > 0
