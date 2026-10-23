from dataclasses import replace
import numpy as np
import pytest

from trace_transport.messages import Message,encode_frame,decode_frame
from trace_transport.joint_session import JointTransportEnv,WorkloadConfig,ACK_ACTIONS
from trace_transport.simulator import NetworkConfig
from trace_transport.wire import Packet,DATA


def run(env):
    while not env.done():env.step(np.zeros((env.count,3)),np.zeros(env.count,int))
    return env.metrics()


def small(**kw):
    return WorkloadConfig(production_seconds=.25,drain_seconds=3,interactive_hz=10,
        image_hz=5,image_bytes=4096,map_bytes=2048,map_interval=.2,**kw)


def test_persistent_binary_sessions_deliver_mixed_bidirectional_messages():
    env=JointTransportEnv(NetworkConfig(loss=.05,ack_loss=.1,feedback_delay_budget=.25),small())
    env.reset(seed=42);result=run(env)
    assert result['messages']>5
    assert result['completed']==result['messages']
    assert set(result['admitted'])=={'interactive','image','map'}
    assert result['wrong']==0
    assert env.sessions[0].messages and env.sessions[1].messages
    assert all(r.payload is not None for r in env.results())


def test_snapshot_replays_events_and_all_rng_streams_exactly():
    env=JointTransportEnv(NetworkConfig(loss=.1,ack_loss=.1,feedback_delay_budget=.25),small())
    env.reset(seed=42);env.step(np.zeros((2,3)),np.zeros(2,int))
    other=env.clone();snapshot=env.snapshot()
    a=run(env);b=run(other)
    assert a==b
    env.restore(snapshot)
    assert run(env)==a


def test_receiver_observation_does_not_read_sender_or_hidden_network():
    env=JointTransportEnv(NetworkConfig(feedback_delay_budget=.25),small());env.reset(seed=42)
    obs=env.observe()
    env.network=replace(env.network,bandwidth_bps=12345,loss=.99)
    env.sessions[0].rate=99999999
    after=env.observe()
    assert np.array_equal(obs['receiver'],after['receiver'])


def test_backpressure_is_counted_and_never_evicts_admitted_payloads():
    env=JointTransportEnv(NetworkConfig(loss=1,ack_loss=1,feedback_delay_budget=.25),small(image_slots=1))
    env.reset(seed=1);result=run(env)
    assert result['blocked']['image']>0
    assert result['admitted']['image']==1
    assert result['completed']==0
    assert result['interactive_miss_rate']==1


def test_shared_airtime_never_overlaps_directions():
    env=JointTransportEnv(NetworkConfig(loss=0,ack_loss=0,feedback_delay_budget=.25),small(shared_airtime=True))
    env.reset(seed=1);run(env)
    intervals=env.airtime_intervals
    assert intervals and {x[2] for x in intervals}=={0,1}
    assert all(a[1]<=b[0]+1e-10 for a,b in zip(intervals,intervals[1:]))


def test_new_envelope_detects_header_corruption():
    packet=Packet(DATA,1,1,0,3,b'\x00\xff\x01')
    frame=encode_frame(42,packet)
    assert decode_frame(frame)==(42,packet)
    damaged=bytearray(frame);damaged[8]^=1
    with pytest.raises(ValueError):decode_frame(bytes(damaged))


def test_follow_baseline_receiver_action_resets_previous_override():
    env=JointTransportEnv(workload=small());env.reset(seed=42)
    env.step(np.zeros((2,3)),np.full(2,len(ACK_ACTIONS),int))
    assert env.sessions[0].delay==.25
    env.step(np.zeros((2,3)),np.zeros(2,int))
    assert env.sessions[0].delay==.025 and env.sessions[0].batch==8


def test_reordering_is_per_message_not_between_message_offsets():
    env=JointTransportEnv(NetworkConfig(loss=0,ack_loss=0,jitter=0,feedback_delay_budget=.25),small())
    env.reset(seed=1);run(env)
    assert all(s.reordered==0 for s in env.sessions)


def test_impossible_deadlines_are_reported_without_changing_denominator():
    env=JointTransportEnv(NetworkConfig(base_rtt=.4,jitter=0,loss=0,ack_loss=0,feedback_delay_budget=.25),small())
    env.reset(seed=42);result=run(env)
    assert result['interactive_unavoidable_bound']==result['desired']['interactive']
    assert result['interactive_miss_rate']==1


def test_indexed_sender_matches_reference_under_loss_and_late_acks():
    from trace_transport.protocol import Sender,Receiver
    from trace_transport.joint_session import SessionSender
    reference=Sender(bytes(range(256))*8,7,payload_size=128)
    indexed=SessionSender(bytes(range(256))*8,7,payload_size=128)
    receiver=Receiver(receipt_feedback=True)
    waiting=[]
    for tick in range(1000):
        now=tick*.01
        a=reference.poll(now);b=indexed.poll(now)
        assert a==b
        if a is not None:
            ack=receiver.on_data(a,now+.01)
            if a.transmission_id%3:waiting.append((now+.08,ack))
        for time,ack in list(waiting):
            if time<=now:
                assert reference.on_ack(ack,now)==indexed.on_ack(ack,now)
                waiting.remove((time,ack))
        assert reference.acknowledged_bytes==indexed.acknowledged_bytes
        if reference.complete:break
    assert reference.complete and indexed.complete


def test_interactive_producers_preserve_valid_utf8_at_every_size():
    for size in [64,256,1024,2048]:
        w=small();w=replace(w,interactive_bytes=size)
        e=JointTransportEnv(NetworkConfig(loss=0,ack_loss=0,feedback_delay_budget=.25),w);e.reset(seed=42);run(e)
        for result in e.results():
            if result.traffic_class=='interactive' and result.payload is not None:
                assert len(result.payload)==size
                assert result.payload.decode('utf-8').encode('utf-8')==result.payload


def test_indexed_global_flight_and_timers_match_reference_world():
    import math
    from trace_transport.protocol import Sender
    from trace_transport.joint_session import TransmissionMap
    class SlowSender(Sender):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.in_flight=TransmissionMap()
        def next_timeout(self):return min((s.timeout_at for s in self.in_flight.values()),default=math.inf)
    cfg=NetworkConfig(loss=.1,ack_loss=.1,jitter=.01,duplicate=.1,feedback_delay_budget=.25)
    a=JointTransportEnv(cfg,small());b=JointTransportEnv(cfg,small(),sender_factory=SlowSender)
    a.reset(seed=42);b.reset(seed=42)
    rng=np.random.default_rng(1)
    while not a.done():
        actions=np.zeros((2,3));actions[:,:2]=rng.uniform(-.5,.5,(2,2));receiver=rng.integers(0,len(ACK_ACTIONS)+1,2)
        x=a.step(actions,receiver);y=b.step(actions,receiver)
        assert x[1:]==y[1:]
        for key in x[0]:assert np.array_equal(x[0][key],y[0][key])
        for session in a.sessions:assert session.flight_bytes==sum(session.flight.values())
    assert a.metrics()==b.metrics()


def test_indexed_receipts_match_reference_for_large_reordered_ranges_and_forgery():
    import random
    from trace_transport.protocol import Sender
    from trace_transport.joint_session import SessionSender
    from trace_transport.session_receiver import SessionReceiver
    from trace_transport.wire import Packet,ACK_RECEIPT
    payload=bytes(range(256))*2048
    a=Sender(payload,7,rate=100000000,cwnd=4000000,feedback_delay_budget=.25)
    b=SessionSender(payload,7,rate=100000000,cwnd=4000000,feedback_delay_budget=.25)
    receiver=SessionReceiver();packets=[]
    for _ in range(len(a.segments)):
        x=a.poll(a.next_send_at);y=b.poll(b.next_send_at)
        assert x==y;packets.append(x)
    random.Random(13).shuffle(packets)
    for i,packet in enumerate(packets):
        ack=receiver.on_data(packet,.1+i*.001)
        assert a.on_ack(ack,.2+i*.001)==b.on_ack(ack,.2+i*.001)
        assert a.acknowledged_bytes==b.acknowledged_bytes
        assert a.flight_bytes==b.flight_bytes
        if i%31==0:
            invalid=replace(ack,metadata={**ack.metadata,'ranges':[[1,2],[0,3]]})
            assert not a.on_ack(invalid,.3+i*.001)
            assert not b.on_ack(invalid,.3+i*.001)
    assert a.complete and b.complete and not b.unacknowledged
    for cls in [Sender,SessionSender]:
        s=cls(payload,9);packet=s.poll(0)
        forged=Packet(ACK_RECEIPT,9,packet.transmission_id,0,len(payload),metadata={
            'length':len(packet.payload),'prefix':len(payload),'ranges':[],'complete':False})
        assert not s.on_ack(forged,.1)  # Claims unsent bytes.
        assert s.acknowledged_bytes==0
