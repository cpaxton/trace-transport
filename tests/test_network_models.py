import io
import json

import pytest

from trace_transport.console import TestBench as Bench,validated_plan
from trace_transport.network_models import wifi_cases
from trace_transport.simulator import NetworkConfig,PacketSimulator,FifoLink
from trace_transport.wire import Packet,DATA


def test_wifi_models_are_named_and_available_in_test_bench(tmp_path):
    bench=Bench(tmp_path)
    try:
        names={case['name'] for case in bench.catalog()['scenarios']}
        assert {'wifi-steady','wifi-contention','wifi-mobile-fade','wifi-burst-loss'} <= names
        for case in wifi_cases():
            plan=validated_plan({'scenario':case['name'],'controllers':['fixed']},[])
            assert plan['network']==case['network']
        (tmp_path/'artifacts').mkdir()
        (tmp_path/'artifacts/receiver-v2-candidate.pt').write_bytes(b'not a sender')
        from trace_transport.console import model_catalog
        assert model_catalog(tmp_path)==[]
    finally:
        bench.close()


def test_loss_burst_is_persistent_then_recovers():
    sim=PacketSimulator(['x'],NetworkConfig(loss=0,ack_loss=0,duplicate=0,
        loss_changes=((.1,1.,1.),(.3,0.,0.))))
    packet=Packet(DATA,1,1,0,1,b'x')
    for time in [0.,.11,.15,.2,.29,.31]:
        sim.now=time
        sim._transmit(0,packet,feedback=False)
    assert sim.forward.counters.random_drops==4
    assert sim.forward.counters.attempted_packets==6


def test_reverse_capacity_change_affects_in_service_packet():
    link=FifoLink(8000,2000,changes=((.5,16000),))
    assert link.enqueue(0,1000)==pytest.approx(.75)


def test_device_harness_readiness_and_quoted_remote_command():
    from scripts.device_trials import wait_ready,remote_listener_command
    assert wait_ready(io.StringIO(json.dumps({'ready':True,'port':9001})+'\n'),1)==9001
    with pytest.raises(ValueError):
        wait_ready(io.StringIO('{"ready":false,"port":9001}\n'),1)
    command=remote_listener_command('/tmp/output with spaces.txt',20)
    assert "'/tmp/output with spaces.txt'" in command
    assert '--receipt-feedback --ready-json' in command
