"""Stylized persistent degradation and recovery, not a calibrated Wi-Fi model.

Between episodes we randomize topology/workload parameters. Within an episode a
latent impairment persists: capacity gradually falls, loss/access delay rise,
and the link recovers. Individual packet drops are conditional random draws;
the underlying probabilities are correlated in time. Queueing remains causal.
"""
import math
from .simulator import NetworkConfig

DOMAIN = dict(forward_bps=[250000,4000000], reverse_bps=[24000,1000000],
              rtt_seconds=[.01,.18], data_loss=[0,.12], ack_loss=[0,.15],
              jitter_seconds=[0,.02], corruption=[0,.01], flows=[1,4],
              queues_bytes=[4096,8192,16384,32768], message_bytes=[16384,32768,49152],
              degradation_probability=.8, outage_probability=.3,
              onset_seconds=[.1,.7], decline_seconds=[.2,.8], hold_seconds=[.4,2.],
              recovery_seconds=[.3,1.2], deadline_seconds=20,
              model='persistent latent degradation; correlated capacity/loss/access-delay; causal FIFO queues')


def sample_receiver_episode(rng):
    def loguniform(low,high):
        return math.exp(rng.uniform(math.log(low),math.log(high)))
    forward = loguniform(*DOMAIN['forward_bps'])
    reverse = loguniform(*DOMAIN['reverse_bps'])
    base_rtt = rng.uniform(*DOMAIN['rtt_seconds'])
    loss, ack_loss = rng.uniform(0,.04),rng.uniform(0,.05)
    changes, reverse_changes, loss_changes, rtt_changes = (),(),(),()
    if rng.random() < DOMAIN['degradation_probability']:
        onset=rng.uniform(*DOMAIN['onset_seconds'])
        decline=rng.uniform(*DOMAIN['decline_seconds'])
        hold=rng.uniform(*DOMAIN['hold_seconds'])
        recovery=rng.uniform(*DOMAIN['recovery_seconds'])
        depth=rng.uniform(.25,.8)
        reverse_depth=rng.uniform(.25,.8)
        # Onset/decline/hold/recovery are coherent and randomly phased, not
        # independent new capacities every control interval.
        phases=[(onset,0.),(onset+decline/2,.5),(onset+decline,1.),
                (onset+decline+hold,1.),(onset+decline+hold+recovery/2,.5),
                (onset+decline+hold+recovery,0.)]
        changes=tuple((t,max(DOMAIN['forward_bps'][0],forward*(1-depth*s))) for t,s in phases)
        reverse_changes=tuple((t,max(DOMAIN['reverse_bps'][0],reverse*(1-reverse_depth*s))) for t,s in phases)
        loss_changes=tuple((t,min(.12,loss+.08*s),min(.15,ack_loss+.1*s)) for t,s in phases)
        # Access/propagation delay is distinct from serialization/queue delay.
        extra_delay=rng.uniform(.01,.08)
        rtt_changes=tuple((t,base_rtt+extra_delay*s) for t,s in phases)
    outages=()
    if rng.random() < DOMAIN['outage_probability']:
        start=rng.uniform(.1,.8)
        outages=((start,start+rng.uniform(.05,.4)),)
    cfg=NetworkConfig(bandwidth_bps=forward,feedback_bandwidth_bps=reverse,
        base_rtt=base_rtt,jitter=rng.uniform(*DOMAIN['jitter_seconds']),
        loss=loss,ack_loss=ack_loss,loss_changes=loss_changes,rtt_changes=rtt_changes,
        corruption=rng.uniform(*DOMAIN['corruption']),ack_corruption=rng.uniform(*DOMAIN['corruption']),
        duplicate=rng.uniform(0,.05),queue_bytes=rng.choice(DOMAIN['queues_bytes']),
        feedback_queue_bytes=rng.choice(DOMAIN['queues_bytes']),
        capacity_changes=changes,feedback_capacity_changes=reverse_changes,outages=outages,
        deadline=DOMAIN['deadline_seconds'])
    return cfg,rng.randint(*DOMAIN['flows']),rng.choice(DOMAIN['message_bytes'])
