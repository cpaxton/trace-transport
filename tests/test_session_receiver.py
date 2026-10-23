import random
import pytest
from trace_transport.protocol import Receiver
from trace_transport.session_receiver import SessionReceiver
from trace_transport.wire import Packet,DATA


@pytest.mark.parametrize('seed',range(10))
def test_indexed_receiver_is_wire_equivalent_under_permutation_duplicates_and_conflicts(seed):
    rng=random.Random(seed);data=rng.randbytes(10000)
    packets=[Packet(DATA,7,i+1,off,len(data),data[off:off+64]) for i,off in enumerate(range(0,len(data),64))]
    rng.shuffle(packets);packets+=packets[:20]
    reference=Receiver(receipt_feedback=True);indexed=SessionReceiver()
    for i,p in enumerate(packets):
        a=reference.on_data(p,i*.01);b=indexed.on_data(p,i*.01)
        assert a==b
    assert indexed.reconstruct()==reference.reconstruct()==data
    bad=Packet(DATA,7,999,1,len(data),b'bad')
    assert indexed.on_data(bad,3) is reference.on_data(bad,3) is None


def test_empty_indexed_receiver():
    p=Packet(DATA,7,1,0,0,b'')
    a=Receiver(receipt_feedback=True);b=SessionReceiver()
    assert a.on_data(p,0)==b.on_data(p,0)
    assert b.reconstruct()==b''
