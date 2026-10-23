import pytest

from trace_transport.feedback import FeedbackScheduler
from trace_transport.protocol import Sender,Receiver
from trace_transport.simulator import NetworkConfig,PacketSimulator


def test_configured_ack_wait_is_included_in_retransmission_timeout():
    old=Sender(b'abc',7)
    sender=Sender(b'abc',7,feedback_delay_budget=.25)
    assert old.rto==pytest.approx(.3)
    assert sender.rto==pytest.approx(.55)
    sender.poll(0)
    assert sender.poll(.4) is None
    assert sender.segments[0].attempts==1
    sender.srtt=.05;sender.rttvar=0
    assert sender.rto==pytest.approx(.35)


def test_wide_feedback_delay_is_reported_and_removed_from_rtt():
    sender=Sender(b'abcdef',7,payload_size=3,feedback_delay_budget=.25)
    receiver=Receiver(receipt_feedback=True)
    packet=sender.poll(0)
    emitted=[];timers=[]
    scheduler=FeedbackScheduler(emitted.append,lambda *x:timers.append(x),max_delay=.25)
    scheduler.batch_size,scheduler.delay=32,.2
    scheduler.on_ack(receiver.on_data(packet,.05),.05)
    when,callback,generation=timers[0];callback(generation,when)
    assert emitted[0].metadata['ack_delay']==pytest.approx(.2)
    assert sender.on_ack(emitted[0],.3)
    assert sender.rtt_samples==pytest.approx([.1])


@pytest.mark.parametrize('budget',[-1,1.01,float('nan')])
def test_invalid_feedback_budgets_rejected(budget):
    with pytest.raises(ValueError):NetworkConfig(feedback_delay_budget=budget)
    with pytest.raises(ValueError):Sender(b'x',1,feedback_delay_budget=budget)


def test_cli_profiles_are_explicit_and_report_actual_timing_budget():
    from trace_transport.cli import run_simulation
    from trace_transport.console import validated_plan
    cfg=NetworkConfig(loss=.02,ack_loss=.02)
    for profile in ['budgeted','sparse']:
        result,report=run_simulation(['hello 🌐'*100],cfg,'fixed',seed=42,feedback_profile=profile)
        assert result[0].completed
        assert report['feedback_profile']==profile
        assert report['config']['feedback_delay_budget']==.25
    plan=validated_plan({'feedback_profile':'sparse'},[])
    assert plan['feedback_profile']=='sparse' and plan['network']['feedback_delay_budget']==.25
    with pytest.raises(ValueError):validated_plan({'feedback_profile':[]},[])
