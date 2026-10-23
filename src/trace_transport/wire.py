"""Versioned experimental datagrams with budgeted, customizable JSON metadata."""

from dataclasses import dataclass, field
import json
import struct
import zlib

DATA = 1
ACK = 2
ACK_BATCH = 3
MAX_METADATA = 512
MAX_PAYLOAD = 1200
_HEADER = struct.Struct("!4sBQQQQHI")
_CRC = struct.Struct("!I")
MAGIC = b"TRC1"


@dataclass(frozen=True)
class Packet:
    kind: int
    message_id: int
    transmission_id: int
    offset: int
    total: int
    payload: bytes = b""
    metadata: dict = field(default_factory=dict)

    def encode(self) -> bytes:
        if self.kind not in (DATA, ACK, ACK_BATCH):
            raise ValueError("unknown packet kind")
        if not isinstance(self.payload, bytes) or len(self.payload) > MAX_PAYLOAD:
            raise ValueError("payload must be bytes within the datagram budget")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON object")
        metadata = json.dumps(
            self.metadata, separators=(",", ":"), sort_keys=True, allow_nan=False
        ).encode("utf-8")
        if len(metadata) > MAX_METADATA:
            raise ValueError("metadata exceeds its 512-byte wire budget")
        for value in (self.message_id, self.transmission_id, self.offset, self.total):
            if type(value) is not int or not 0 <= value < 2**64:
                raise ValueError("packet integers must be unsigned 64-bit values")
        if self.offset > self.total or self.offset + len(self.payload) > self.total:
            raise ValueError("payload exceeds the declared message")
        if self.kind in (ACK, ACK_BATCH) and self.payload:
            raise ValueError("ACK packets cannot contain application payload")
        body = _HEADER.pack(
            MAGIC, self.kind, self.message_id, self.transmission_id,
            self.offset, self.total, len(metadata), len(self.payload),
        ) + metadata + self.payload
        return body + _CRC.pack(zlib.crc32(body))

    @classmethod
    def decode(cls, wire: bytes) -> "Packet":
        if len(wire) < _HEADER.size + _CRC.size:
            raise ValueError("truncated packet")
        body, checksum = wire[:-4], wire[-4:]
        if zlib.crc32(body) != _CRC.unpack(checksum)[0]:
            raise ValueError("packet checksum mismatch")
        magic, kind, message, tx, offset, total, meta_len, data_len = _HEADER.unpack_from(body)
        if magic != MAGIC or kind not in (DATA, ACK, ACK_BATCH):
            raise ValueError("unsupported protocol version or packet kind")
        if meta_len > MAX_METADATA or data_len > MAX_PAYLOAD:
            raise ValueError("packet exceeds a wire budget")
        if len(body) != _HEADER.size + meta_len + data_len:
            raise ValueError("packet length mismatch")
        try:
            metadata = json.loads(body[_HEADER.size:_HEADER.size + meta_len])
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("invalid metadata JSON") from exc
        packet = cls(kind, message, tx, offset, total, body[_HEADER.size + meta_len:], metadata)
        # Apply the same structural validation to received and locally constructed packets.
        packet.encode()
        return packet
