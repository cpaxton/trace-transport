"""Opaque application messages and a versioned session envelope."""
from dataclasses import dataclass
import math
import struct
import zlib

from .wire import Packet

CLASSES=('interactive','image','map','bulk')
MAX_MESSAGE_BYTES=16*1024*1024
_HEADER=struct.Struct('!4sQ')
_CRC=struct.Struct('!I')


@dataclass(frozen=True)
class Message:
    payload:bytes
    traffic_class:str='bulk'
    deadline_ms:float|None=None

    def __post_init__(self):
        if not isinstance(self.payload,bytes) or len(self.payload)>MAX_MESSAGE_BYTES:
            raise ValueError('message must contain bytes within the 16-MiB limit')
        if self.traffic_class not in CLASSES:
            raise ValueError('unknown traffic class')
        if self.deadline_ms is not None and (not math.isfinite(self.deadline_ms) or self.deadline_ms<=0):
            raise ValueError('deadline must be finite and positive')

    @classmethod
    def text(cls,text,deadline_ms=100):
        return cls(text.encode('utf-8'),'interactive',deadline_ms)


@dataclass(frozen=True)
class BinaryResult:
    message_id:int
    traffic_class:str
    payload:bytes|None
    completed:bool
    receiver_complete:bool
    offered_at:float
    admitted_at:float
    delivered_at:float|None
    confirmed_at:float|None
    deadline_ms:float|None

    def summary(self):
        return dict(message_id=self.message_id,traffic_class=self.traffic_class,
            completed=self.completed,receiver_complete=self.receiver_complete,
            delivered_bytes=len(self.payload) if self.payload is not None else 0,
            offered_at=self.offered_at,admitted_at=self.admitted_at,
            delivered_at=self.delivered_at,confirmed_at=self.confirmed_at,
            deadline_ms=self.deadline_ms,
            on_time=self.delivered_at is not None and (self.deadline_ms is None or
                self.delivered_at-self.offered_at<=self.deadline_ms/1000+1e-9))


def encode_frame(session_id,packet):
    body=_HEADER.pack(b'TRC2',session_id)+packet.encode()
    return body+_CRC.pack(zlib.crc32(body))


def decode_frame(wire):
    if len(wire)<_HEADER.size+4:
        raise ValueError('truncated session frame')
    body=wire[:-4]
    if zlib.crc32(body)!=_CRC.unpack(wire[-4:])[0]:
        raise ValueError('session frame checksum mismatch')
    magic,session=_HEADER.unpack_from(body)
    if magic!=b'TRC2':
        raise ValueError('unsupported session protocol')
    return session,Packet.decode(body[_HEADER.size:])
