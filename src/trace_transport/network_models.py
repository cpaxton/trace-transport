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
