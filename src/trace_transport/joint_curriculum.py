"""Versioned persistent mixed-traffic training domains and disjoint splits."""
from dataclasses import replace
import math
import random

from .joint_session import WorkloadConfig
from .simulator import NetworkConfig

DOMAIN=dict(version='joint-domain-v2',clients=[1,8],forward_bps=[100000,50000000],
 reverse_bps=[16000,10000000],rtt_seconds=[.002,.3],queues_bytes=[2048,4096,8192,16384,65536,262144],
 interactive_bytes=[64,2048],interactive_hz=[5,50],image_bytes=[32768,524288],image_hz=[.5,10],
 map_bytes=[65536,4194304],map_interval=[.5,10],families=['steady','contention','fade','burst'],
 family_weights=[.2,.3,.3,.2],load_weights=[.4,.4,.2],feedback_delay_budget=.25)


def sample_case(seed,*,production=20.,drain=20.,stage='full',ood=False,_attempt=0):
    rng=random.Random(seed)
    family=rng.choices(DOMAIN['families'],DOMAIN['family_weights'])[0]
    clients=rng.randint(1,8) if stage=='full' else rng.randint(1,2)
    if ood:clients=rng.choice([2,6,12])
    interactive=rng.choice([64,256,1024,2048]);hz=rng.uniform(5,50)
    image_size=rng.choice([32768,65536,131072,262144,524288])
    image_hz=math.exp(rng.uniform(math.log(.5),math.log(10)))
    map_size=rng.choice([65536,262144,524288,1048576,4194304]);map_interval=rng.uniform(.5,10)
    if stage=='easy':
        family='steady';interactive=256;hz=10;image_size=32768;image_hz=1.;map_size=65536;map_interval=5.
    workload=WorkloadConfig(clients=clients,production_seconds=production,drain_seconds=drain,
        interactive_bytes=interactive,interactive_hz=hz,image_bytes=image_size,image_hz=image_hz,
        map_bytes=map_size,map_interval=map_interval,shared_airtime=family!='steady',producer_phase_jitter=True)
    bucket=rng.choices(['feasible','near','overloaded'],[.4,.4,.2])[0]
    ratio=rng.uniform(*{'feasible':(.2,.7),'near':(.7,1.1),'overloaded':(1.1,3.)}[bucket])
    # Include DATA envelope/metadata and opposite-direction receipt traffic.
    control_wire=clients*hz*(interactive+math.ceil(interactive/512)*128+220)
    bulk_payload=clients*(image_size*image_hz+map_size/map_interval)
    bulk_wire=clients*(image_hz*(image_size+math.ceil(image_size/512)*128)+
                       (map_size+math.ceil(map_size/512)*128)/map_interval)
    up=8*(control_wire+bulk_wire)
    down=8*(control_wire+bulk_payload*220/4096)
    if family!='steady':
        share=rng.uniform(.4,.8)
        rate=up/(ratio*share);reverse=down/(ratio*(1-share))
    else:
        rate=up/ratio;reverse=down/ratio
    if stage=='easy':
        ratio=rng.uniform(.2,.4);rate=up/ratio;reverse=down/ratio
    if not (100000<=rate<=50000000 and 16000<=reverse<=10000000):
        if _attempt>=128:raise RuntimeError('unable to sample bounded load regime')
        return sample_case(rng.getrandbits(64),production=production,drain=drain,stage=stage,ood=ood,_attempt=_attempt+1)
    rtt=(rng.uniform(.002,.1) if rng.random()<.8 else rng.uniform(.1,.3)) if stage=='full' else rng.uniform(.005,.03)
    # A feasible label must also permit an isolated interactive payload to arrive.
    if bucket=='feasible' and (min(rate,reverse)<=0 or interactive*8/min(rate,reverse)+rtt/2>.08):
        if _attempt>=128:raise RuntimeError('unable to sample deadline-feasible regime')
        return sample_case(rng.getrandbits(64),production=production,drain=drain,stage=stage,ood=ood,_attempt=_attempt+1)
    loss=rng.uniform(0,.05) if family!='steady' else 0.
    changes=reverse_changes=loss_changes=rtt_changes=outages=()
    if family in ('fade','contention'):
        onset=rng.uniform(.1,max(.2,production*.25));fall=rng.uniform(.2,2);hold=rng.uniform(.5,5);recover=rng.uniform(.3,2)
        phases=[(onset,0.),(onset+fall/2,.5),(onset+fall,1.),(onset+fall+hold,1.),
                (onset+fall+hold+recover/2,.5),(onset+fall+hold+recover,0.)]
        depth=rng.uniform(.3,.9)
        changes=tuple((t,max(10000,rate*(1-depth*s))) for t,s in phases)
        reverse_changes=tuple((t,max(8000,reverse*(1-depth*s))) for t,s in phases)
        if family=='fade':
            loss_changes=tuple((t,min(.2,loss+.12*s),min(.2,loss+.12*s)) for t,s in phases)
            rtt_changes=tuple((t,rtt+.08*s) for t,s in phases)
    if family=='burst':
        rows=[];t=rng.uniform(.05,.2)
        while t<production:
            rows.extend([(t,rng.uniform(.3,.8),rng.uniform(.3,.8)),(t+rng.uniform(.05,.3),loss,loss)])
            t=rows[-1][0]+rng.uniform(.2,2.)
        loss_changes=tuple(rows)
    if stage=='full' and rng.random()<.3:
        start=rng.uniform(.1,max(.2,production*.3));outages=((start,start+rng.uniform(.1,5.)),)
    if ood:
        family='heldout-handoff'
        changes=((production*.2,rate*.15),(production*.4,rate*1.8),(production*.6,rate*.35))
        reverse_changes=((production*.25,max(8000,reverse*.1)),(production*.5,reverse*1.5))
        rtt_changes=((production*.2,.45),(production*.4,.05),(production*.6,.35))
    cfg=NetworkConfig(bandwidth_bps=rate,feedback_bandwidth_bps=reverse,base_rtt=rtt,
        jitter=rng.uniform(0,.03) if family!='steady' else 0.,loss=loss,ack_loss=loss,
        corruption=rng.uniform(0,.005) if family!='steady' else 0.,ack_corruption=rng.uniform(0,.005) if family!='steady' else 0.,
        duplicate=rng.uniform(0,.03) if stage=='full' else 0.,queue_bytes=rng.choice(DOMAIN['queues_bytes']) if stage=='full' else 16384,feedback_queue_bytes=rng.choice(DOMAIN['queues_bytes']) if stage=='full' else 16384,
        capacity_changes=changes,feedback_capacity_changes=reverse_changes,loss_changes=loss_changes,
        rtt_changes=rtt_changes,outages=outages,feedback_delay_budget=.25,deadline=production+drain)
    return cfg,workload,dict(family=family,target_load_bucket=bucket if stage=='full' else 'feasible',
        actual_offered_ratio=up/rate+down/reverse if workload.shared_airtime else max(up/rate,down/reverse),
        directional_wire_load=[up/rate,down/reverse],generation_seed=seed,version=DOMAIN['version'])
