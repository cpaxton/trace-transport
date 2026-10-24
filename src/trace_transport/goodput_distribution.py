"""Versioned synthetic service/workload strata for adaptation diagnostics.

These are transport-level stress models, not calibrated 802.11 or empirical
prevalence estimates. Whole measured paths remain a separate validation gate.
"""
import math
import random
import numpy as np

from .goodput import GoodputEnv
from .joint_session import WorkloadConfig
from .simulator import NetworkConfig

FAMILIES = ('steady', 'wired-lan', 'wifi-contention', 'mobile-fade',
            'burst-interference', 'capacity-opportunities', 'propagation-step',
            'reverse-congestion', 'loss-cause-switch', 'outage-recovery',
            'producer-churn', 'bufferbloat')
DOMAIN = dict(version='goodput-domain-v3.0', reward='verified-goodput-v3.0',
              families=list(FAMILIES), family_weights='uniform synthetic strata, not deployment prevalence',
              clients=[1,4], forward_bps=[500000,80000000], rtt_seconds=[.002,.12],
              rate_limit_bytes_per_second=16000000, window_limit_bytes=2097152,
              history_scope='local endpoint observations only',
              limitations=['not calibrated to devices', 'FIFO/drop-tail only',
                           'no independent TCP competitors', 'no RF/MAC or OS inference costs'])


def loguniform(rng, low, high):
    return math.exp(rng.uniform(math.log(low),math.log(high)))


def sample_case(seed, *, production=12., drain=3., stage='full', ood=False, family=None):
    rng=random.Random(seed)
    family=family or (stage[7:] if stage.startswith('family:') else rng.choice(FAMILIES))
    if family not in FAMILIES:raise ValueError('unknown v3 family')
    if stage=='easy':family='steady'
    wireless=family in ('wifi-contention','mobile-fade','burst-interference','outage-recovery')
    clients=1 if stage=='easy' else rng.randint(1,4)
    rate=loguniform(rng,500000,80000000)
    rtt=loguniform(rng,.002,.12)
    if family=='wired-lan':rate=loguniform(rng,10000000,80000000);rtt=loguniform(rng,.002,.012)
    reverse=max(128000,rate*rng.uniform(.15,1.))
    queue=max(4096,min(2097152,int(rate*rtt/8*loguniform(rng,.25,4.))))
    if family=='bufferbloat':queue=max(65536,min(2097152,int(rate*rng.uniform(.3,1.5)/8)))
    forward=[];backward=[];losses=[];delays=[];outages=[]
    loss=rng.uniform(0,.008) if wireless else 0.
    # Semi-Markov dwell times: episode-relative boundaries are not fixed phase cues.
    t=rng.uniform(.08,min(.8,max(.09,production*.15)))
    times=[]
    while t<production*.9:
        times.append(t);t+=loguniform(rng,.08,2.5)
    level=1.
    for i,t in enumerate(times):
        if family=='wifi-contention':
            level=rng.choice([.15,.35,.65,1.]);forward.append((t,rate*level));backward.append((t,reverse*level))
        elif family=='mobile-fade':
            level=max(.08,min(1.3,level*rng.uniform(.45,1.6)))
            forward.append((t,rate*level));backward.append((t,reverse*max(.08,level)))
            p=min(.15,.004+.015/level);losses.append((t,p,p));delays.append((t,rtt+.004/level))
        elif family=='burst-interference':
            bad=i%2==0;losses.append((t,rng.uniform(.2,.6) if bad else .002,rng.uniform(.1,.5) if bad else .002))
        elif family=='capacity-opportunities':
            forward.append((t,rate*(rng.uniform(1.5,3.) if i%2==0 else rng.uniform(.15,.4))))
        elif family=='propagation-step':delays.append((t,rtt+rng.uniform(.02,.1) if i%2==0 else rtt))
        elif family=='reverse-congestion':backward.append((t,rng.uniform(16000,96000) if i%2==0 else reverse))
        elif family=='loss-cause-switch':
            congested=i%2==0;forward.append((t,rate*.25 if congested else rate))
            losses.append((t,0. if congested else .03,0. if congested else .03))
    if family=='outage-recovery':
        start=rng.uniform(.15,.5)*production
        outages=[(start,min(production*.85,start+rng.uniform(.1,.8)))]
    if ood:
        # New parameter combinations, not a claim of an unseen model family.
        clients=4;wireless=True
        forward=[(t,rate*rng.choice([.1,.4,1.5])) for t in times]
        backward=[(t,rng.choice([32000.,128000.,reverse])) for t in times]
        delays=[(t,rng.choice([rtt,.15,.03])) for t in times]
        loss=.02
    cfg=NetworkConfig(bandwidth_bps=rate,feedback_bandwidth_bps=reverse,base_rtt=rtt,
        jitter=min(.005,rtt*.1),loss=loss,ack_loss=loss,queue_bytes=queue,
        feedback_queue_bytes=rng.choice([2048,8192,32768]),capacity_changes=tuple(forward),
        feedback_capacity_changes=tuple(backward),loss_changes=tuple(losses),rtt_changes=tuple(delays),
        outages=tuple(outages),deadline=production+drain,feedback_delay_budget=.25)
    workload=WorkloadConfig(clients=clients,production_seconds=production,drain_seconds=drain,
        interactive_bytes=rng.choice([64,128,512]),interactive_hz=rng.choice([5.,10.,20.]),
        image_bytes=rng.choice([16384,65536,131072]),map_bytes=rng.choice([65536,262144,524288]),
        byte_limit=4194304,interactive_slots=64,image_slots=rng.choice([1,2,4]),map_slots=1,
        image_hz=1.,map_interval=5.,shared_airtime=wireless,producer_phase_jitter=True)
    churn_times=[production*rng.uniform(.25,.4),production*rng.uniform(.6,.8)]
    label=dict(version=DOMAIN['version'],family=family,churn=family=='producer-churn',
               churn_times=churn_times,seed=seed,synthetic=True,shared_airtime=wireless,
               distribution_shift=ood,change_count=len(set(t for changes in [forward,backward,losses,delays] for t,*_ in changes)))
    return cfg,workload,label


class GoodputDistributionEnv(GoodputEnv):
    rate_limit=16000000.
    window_limit=2097152.
    def baseline(self,index,rate_control='fixed',scheduler='edf',feedback='budgeted'):
        if rate_control!='delay-fast':return super().baseline(index,rate_control,scheduler,feedback)
        # Wider-bandwidth reference: avoid the v2 1-ms floor's ~8-Mbps ceiling.
        s=self.sessions[index];a=np.zeros(2,np.float32)
        if s.last_feedback is not None:
            qdelay=max(.00005,s.srtt-s.min_rtt)
            target=min(self.rate_limit,max(2000.,self.network.payload_size/(.5*qdelay)))
            a[0]=np.clip(math.log(target/s.rate)/.2,-1,1)
            window=min(self.window_limit,max(2048.,1.5*target*s.min_rtt+2*self.network.payload_size))
            a[1]=np.clip(math.log(window/s.window)/.2,-1,1)
        return a
    def metrics(self):
        out=super().metrics();out['profile']='goodput-v3';return out
