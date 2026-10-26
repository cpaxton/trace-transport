"""V4 service coverage: sub-ms LAN, gigabit service, and explicit control timing.

Extends the frozen v3 strata. These ranges are deployment-motivated hypotheses,
not calibrated service distributions. Reported PHY rates are not capacity data.
"""
from dataclasses import replace
import math
import random
import numpy as np

from .goodput_distribution import sample_case as v3_case,GoodputDistributionEnv,FAMILIES,loguniform
from .joint_session import SENDER_FEATURES,RECEIVER_FEATURES

DOMAIN=dict(version='goodput-domain-v4.0',reward='verified-goodput-v4.0',families=list(FAMILIES),
    clients=[1,4],wired_bps=[10000000,1000000000],wireless_bps=[500000,300000000],
    wired_rtt_seconds=[.0003,.004],other_rtt_seconds=[.001,.12],control_intervals=[.002,.005,.01,.02],
    rate_limit_bytes_per_second=256000000,window_limit_bytes=16777216,
    observation_overrides={'sender.age':'log1p(control_interval/.002)',
                           'receiver.age':'log1p(control_interval/.002)',
                           'sender.rtt':'log1p(srtt/.001)','sender.min_rtt':'log1p(min_rtt/.001)'},
    limitations=['synthetic service, not measured prevalence','FIFO/drop-tail only','no independent TCP competitors',
                 'no RF/MAC or OS/inference delays','more opportunities at shorter ticks are not directly comparable training budgets'])


def sample_case(seed,*,production=12.,drain=3.,stage='full',ood=False,family=None):
    cfg,w,label=v3_case(seed,production=production,drain=drain,stage=stage,ood=ood,family=family)
    rng=random.Random(seed ^ 0x43414C4942414E)
    wired=label['family']=='wired-lan'
    rate=loguniform(rng,10000000,1000000000) if wired else loguniform(rng,500000,300000000 if w.shared_airtime else 80000000)
    rtt=loguniform(rng,.0003,.004) if wired else loguniform(rng,.001,.12)
    if stage=='easy':rate=min(rate,8000000.)
    factor=rate/cfg.bandwidth_bps
    ratio=cfg.queue_bytes/max(1.,cfg.bandwidth_bps*cfg.base_rtt/8)
    queue=max(4096,min(16777216,int(rate*rtt/8*ratio)))
    tick=rng.choice([.002,.005,.02] if wired else [.005,.01,.02])
    cfg=replace(cfg,bandwidth_bps=rate,feedback_bandwidth_bps=max(16000.,cfg.feedback_bandwidth_bps*factor),
        base_rtt=rtt,jitter=min(.005,rtt*.1),queue_bytes=queue,control_interval=tick,
        capacity_changes=tuple((t,v*factor) for t,v in cfg.capacity_changes),
        feedback_capacity_changes=tuple((t,v*factor) for t,v in cfg.feedback_capacity_changes),
        rtt_changes=tuple((t,max(.0001,v-cfg.base_rtt+rtt)) for t,v in cfg.rtt_changes))
    w=replace(w,byte_limit=16777216,image_bytes=rng.choice([16384,65536,131072,524288]),
              map_bytes=rng.choice([65536,262144,524288,2097152]))
    label.update(version=DOMAIN['version'],control_interval=tick)
    return cfg,w,label


class GoodputDeploymentEnv(GoodputDistributionEnv):
    rate_limit=256000000.
    window_limit=16777216.
    def observe(self):
        obs=super().observe()
        # Replace the episode clock, rather than changing legacy model shapes.
        # A v3 checkpoint must never be silently deployed with v4 semantics.
        value=math.log1p(self.network.control_interval/.002)
        obs['sender'][:,SENDER_FEATURES.index('age')]=value
        obs['receiver'][:,RECEIVER_FEATURES.index('age')]=value
        for i,s in enumerate(self.sessions):
            obs['sender'][i,SENDER_FEATURES.index('rtt')]=math.log1p(s.srtt/.001)
            obs['sender'][i,SENDER_FEATURES.index('min_rtt')]=math.log1p(s.min_rtt/.001)
        return obs
    def baseline(self,index,rate_control='fixed',scheduler='edf',feedback='budgeted'):
        if rate_control!='delay-fast':return super().baseline(index,rate_control,scheduler,feedback)
        s=self.sessions[index];a=np.zeros(2,np.float32)
        if s.last_feedback is not None:
            target=min(self.rate_limit,max(2000.,self.network.payload_size/(.5*max(.000001,s.srtt-s.min_rtt))))
            a[0]=np.clip(math.log(target/s.rate)/.2,-1,1)
            target_window=min(self.window_limit,max(2048.,1.5*target*s.min_rtt+2*self.network.payload_size))
            a[1]=np.clip(math.log(target_window/s.window)/.2,-1,1)
        return a
    def step(self,sender_actions,receiver_actions):
        before=self.now
        result=super().step(sender_actions,receiver_actions)
        result[0]['transition_seconds']=self.now-before
        return result
    def metrics(self):
        out=super().metrics();out['profile']='goodput-v4';out['control_interval']=self.network.control_interval
        return out
