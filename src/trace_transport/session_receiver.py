"""Indexed equivalent of the receipt kernel for sustained binary workloads.

Preserves the reference Receiver's wire receipts and validation, while avoiding
full overlap scans, repeated sorting, and repeated whole-message hashes.
"""
from bisect import bisect_left
import hashlib

from .protocol import Receiver
from .wire import DATA,ACK_RECEIPT,Packet


class SessionReceiver(Receiver):
    def __init__(self,**kwargs):
        super().__init__(receipt_feedback=True,**kwargs)
        self.offsets=[];self.spans=[];self.output=None;self.digest=None

    def on_data(self,packet,now):
        if packet.kind!=DATA or packet.total>self.max_message_bytes:return None
        if packet.offset+len(packet.payload)>packet.total or (packet.total and not packet.payload):return None
        if self.message_id is not None and (packet.message_id!=self.message_id or packet.total!=self.total):return None
        if packet.offset in self.chunks:
            if self.chunks[packet.offset]!=packet.payload:return None
        else:
            off=packet.offset;end=off+len(packet.payload);position=bisect_left(self.offsets,off)
            if position and self.offsets[position-1]+len(self.chunks[self.offsets[position-1]])>off:return None
            if position<len(self.offsets) and self.offsets[position]<end:return None
            self.offsets.insert(position,off)
            self.message_id,self.total=packet.message_id,packet.total
            self.chunks[off]=packet.payload;self.received_bytes+=len(packet.payload)
            j=bisect_left(self.spans,(off,-1));low=off;high=end
            if j and self.spans[j-1][1]==off:
                j-=1;low=self.spans[j][0];self.spans.pop(j)
            if j<len(self.spans) and self.spans[j][0]==end:
                high=self.spans[j][1];self.spans.pop(j)
            self.spans.insert(j,(low,high))
            if self.received_bytes==self.total:
                self.completed_at=now
                self.output=b''.join(self.chunks[off] for off in self.offsets)
                self.digest=hashlib.sha256(self.output).hexdigest()
        metadata={**self.custom_metadata,'length':len(packet.payload),'credit':self.max_message_bytes,
                  'received':self.received_bytes,'complete':self.complete}
        if self.complete:metadata['sha256']=self.digest
        prefix=self.spans[0][1] if self.spans and self.spans[0][0]==0 else 0
        start=1 if self.spans and self.spans[0][0]==0 else 0
        metadata.update(prefix=prefix,ranges=[list(pair) for pair in self.spans[start:start+8]])
        ack=Packet(ACK_RECEIPT,packet.message_id,packet.transmission_id,packet.offset,packet.total,metadata=metadata)
        while True:
            try:
                Packet(ACK_RECEIPT,packet.message_id,packet.transmission_id,packet.offset,packet.total,
                       metadata={**metadata,'ack_delay':.024999999999999998}).encode()
                break
            except ValueError:
                if not metadata['ranges']:raise
                metadata['ranges'].pop()
        ack.encode();return ack

    def reconstruct(self):
        if not self.complete:raise RuntimeError('message is incomplete')
        return self.output
