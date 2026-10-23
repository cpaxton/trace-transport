

def test_python_metadata_encoder_preserves_canonical_wire_bytes():
    import json,random
    from trace_transport.wire import Packet,ACK_RECEIPT,_HEADER
    rng=random.Random(61)
    for _ in range(200):
        metadata={'prefix':rng.randrange(2**63),'ranges':[[rng.randrange(1000),rng.randrange(1000)] for _ in range(8)],
            'complete':False,'ack_delay':rng.uniform(0,.25),'label':'telemetry 🌐','nullable':None,'length':512}
        packet=Packet(ACK_RECEIPT,1,2,0,1024,metadata=metadata)
        encoded=packet.encode();length=_HEADER.unpack_from(encoded)[-2]
        assert encoded[_HEADER.size:_HEADER.size+length]==json.dumps(metadata,separators=(',',':'),sort_keys=True,allow_nan=False).encode()
        assert Packet.decode(encoded)==packet
