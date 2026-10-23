"""Named, stylized Wi-Fi service models above the MAC layer.

These model available service capacity and residual loss/access delay, not RF,
802.11 contention/backoff, PHY retransmissions, or a calibrated robot deployment.
Persistent schedules are hidden from endpoint policies. FIFO queueing remains
traffic-dependent; loss draws are random conditional on persistent impairment.
"""
from dataclasses import asdict
from .simulator import NetworkConfig


def wifi_cases():
    definitions = [
        ('wifi-steady','Wi-Fi conditions: stable service with small residual loss.',NetworkConfig(
            bandwidth_bps=6000000,feedback_bandwidth_bps=1000000,base_rtt=.012,jitter=.002,
            loss=.005,ack_loss=.005,deadline=20)),
        ('wifi-contention','Wi-Fi conditions: persistent airtime contention reduces both service rates, then recovers.',NetworkConfig(
            bandwidth_bps=6000000,feedback_bandwidth_bps=1000000,base_rtt=.012,jitter=.004,
            loss=.005,ack_loss=.005,deadline=20,
            capacity_changes=((.1,3000000),(.3,800000),(.9,800000),(1.5,3000000),(2.,6000000)),
            feedback_capacity_changes=((.1,512000),(.3,64000),(.9,64000),(1.5,256000),(2.,1000000)))),
        ('wifi-mobile-fade','Wi-Fi conditions: gradual fade, persistent poor service, loss/access-delay rise, and recovery.',NetworkConfig(
            bandwidth_bps=6000000,feedback_bandwidth_bps=1000000,base_rtt=.012,jitter=.005,
            loss=.005,ack_loss=.005,deadline=20,
            capacity_changes=((.1,3000000),(.3,1000000),(.5,400000),(1.,400000),(1.5,2000000),(2.,6000000)),
            feedback_capacity_changes=((.1,512000),(.3,128000),(.5,48000),(1.,48000),(1.5,256000),(2.,1000000)),
            loss_changes=((.1,.02,.02),(.3,.06,.08),(.5,.12,.15),(1.,.12,.15),(1.5,.03,.03),(2.,.005,.005)),
            rtt_changes=((.1,.02),(.3,.04),(.5,.07),(1.,.07),(1.5,.03),(2.,.012)))),
        ('wifi-burst-loss','Wi-Fi conditions: clustered interference bursts separated by clean intervals.',NetworkConfig(
            bandwidth_bps=3000000,feedback_bandwidth_bps=512000,base_rtt=.015,jitter=.004,
            loss=.005,ack_loss=.005,deadline=20,
            loss_changes=((.1,.6,.5),(.25,.005,.005),(.5,.5,.7),(.7,.005,.005),(1.2,.7,.6),(1.4,.005,.005)))),
    ]
    return [dict(name=name,description=description+' Stylized; not calibrated to hardware.',
                 flows=2,message_bytes=131072,network=asdict(config)) for name,description,config in definitions]


def generalization_cases():
    cases = {
        'loss-bursts':(NetworkConfig(loss=.01,ack_loss=.01,feedback_bandwidth_bps=64000,
            loss_changes=((.2,.4,.5),(.55,.01,.01),(1.2,.5,.4),(1.6,.01,.01)),deadline=25),2,32768),
        'delay-drift':(NetworkConfig(feedback_bandwidth_bps=64000,rtt_changes=((.2,.1),(.4,.2),(.8,.35),(1.5,.1),(2.,.05)),deadline=25),2,32768),
        'long-rtt':(NetworkConfig(base_rtt=.3,feedback_bandwidth_bps=64000,deadline=25),2,32768),
        'reverse-12k':(NetworkConfig(feedback_bandwidth_bps=12000,deadline=25),2,32768),
        'six-flows':(NetworkConfig(feedback_bandwidth_bps=192000,deadline=25),6,32768),
        'large-message':(NetworkConfig(bandwidth_bps=2000000,feedback_bandwidth_bps=128000,deadline=25),2,131072),
        'small-queue':(NetworkConfig(feedback_bandwidth_bps=64000,feedback_queue_bytes=2048,
                                    queue_bytes=2048,deadline=25),2,32768),
        'reverse-changes':(NetworkConfig(feedback_bandwidth_bps=32000,
             feedback_capacity_changes=((.3,512000),(.9,16000),(2.,64000)),deadline=25),2,32768),
    }
    cases.update({case['name']:(NetworkConfig(**case['network']),case['flows'],case['message_bytes'])
                  for case in wifi_cases()})
    return cases
