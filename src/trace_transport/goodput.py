"""Backlogged verified-goodput task; deliberately separate from joint-v1 scoring."""
from collections import Counter
from dataclasses import replace
import math
import random

from .joint_session import JointTransportEnv,WorkloadConfig,SENDER_FEATURES
from .messages import Message
from .simulator import NetworkConfig

DOMAIN=dict(version='goodput-domain-v2.1',families=['steady','wireless-loss','contention','ack-bottleneck','handoff','churn','loss-cause-switch'],
    clients=[1,3],forward_bps=[500000,8000000],rtt_seconds=[.008,.06],
    rate_limit_bytes_per_second=2000000,window_limit_bytes=262144,
    reward='verified-goodput-v2.1',deadline_penalty=50.,measurement='completed object bytes during fixed production window')


def sample_case(seed,*,production=6.,drain=2.,stage='full',ood=False):
    rng=random.Random(seed);family=rng.choice(DOMAIN['families']) if stage=='full' else 'steady'
    clients=rng.randint(1,3) if stage=='full' else 1
    rate=math.exp(rng.uniform(math.log(500000),math.log(8000000)))
    reverse=max(256000,rate*.5);rtt=rng.uniform(.008,.06)
    changes=feedback=loss_changes=rtts=()
    loss=0.;t1=production*rng.uniform(.15,.3);t2=production*rng.uniform(.55,.7)
    if family=='wireless-loss':loss=rng.uniform(.01,.05)
    if family=='contention':
        changes=((t1,rate*.5),(t2,rate),(production*.85,rate*.7))
    if family=='ack-bottleneck':
        reverse=128000;feedback=((t1,256000),(t2,128000))
    if family=='handoff':
        changes=((t1,rate*.4),(t2,rate*1.3))
        rtts=((t1,rtt+.04),(t2,rtt))
    if family=='loss-cause-switch':
        loss=.03
        loss_changes=((t1,0.,0.),(t2,.03,.03))
        changes=((t1,rate*.45),(t2,rate))
    if ood:
        family='heldout-coupled-handoff';clients=4
        changes=((production*.15,rate*.35),(production*.4,rate*1.5),(production*.7,rate*.7))
        feedback=((production*.3,128000),(production*.6,512000))
        rtts=((production*.2,.1),(production*.5,.025));loss=.02
    cfg=NetworkConfig(bandwidth_bps=rate,feedback_bandwidth_bps=reverse,base_rtt=rtt,
        jitter=.001,loss=loss,ack_loss=loss,queue_bytes=rng.choice([4096,8192,16384]),feedback_queue_bytes=4096,
        capacity_changes=changes,feedback_capacity_changes=feedback,rtt_changes=rtts,loss_changes=loss_changes,feedback_delay_budget=.25,
        deadline=production+drain)
    workload=WorkloadConfig(clients=clients,production_seconds=production,drain_seconds=drain,
        interactive_bytes=128,interactive_hz=10,image_bytes=16384,map_bytes=65536,image_slots=2,map_slots=1,
        byte_limit=262144,interactive_slots=32,image_hz=1,map_interval=5,producer_phase_jitter=True)
    return cfg,workload,dict(version=DOMAIN['version'],family=family,churn=family=='churn',churn_times=[production*rng.uniform(.3,.45),production*rng.uniform(.65,.8)],seed=seed)


class GoodputEnv(JointTransportEnv):
    rate_limit=2000000.
    window_limit=262144.
    def __init__(self,network,workload,*,mode='mixed',churn=False,churn_times=None):
        if mode not in ('mixed','bulk'):raise ValueError('unknown goodput mode')
        super().__init__(network,workload,auto_producers=False)
        self.mode=mode;self.churn=churn;self.churn_times=churn_times
    def reset(self,*,seed=0):
        super().reset(seed=seed);self.active=set();self.verified=Counter();self.confirmed=Counter();self.host_verified=Counter();self.drained_verified=0
        for client in range(self.workload.clients):
            join=.0 if not self.churn else .15*client
            self._schedule(join,'application','join',2*client)
            if self.churn:
                leave,join_again=self.churn_times or (self.workload.production_seconds*.4,self.workload.production_seconds*.6)
                self._schedule(leave,'application','leave',2*client)
                self._schedule(join_again,'application','join',2*client)
        # Admit initial demand before the first actor decision.
        self._advance(0.)
        return self.observe(),self.info()
    def _application_event(self,kind,index):
        if kind=='leave':self.active.discard(index);self.active.discard(index^1);return
        self.active.update([index,index^1]);self._refill(index)
        if self.mode=='mixed':
            for i in [index,index^1]:self._schedule(self.now,'offer',i,'interactive',1/self.workload.interactive_hz,self.workload.interactive_bytes)
    def _offer(self,index,*args):
        if index in self.active:super()._offer(index,*args)
    def _refill(self,index):
        if index not in self.active or index%2 or self.now>=self.workload.production_seconds-1e-9:return
        s=self.sessions[index];w=self.workload
        for cls,size,slots in [('image',w.image_bytes,w.image_slots),('map',w.map_bytes,w.map_slots)]:
            while sum(s.messages[mid].message.traffic_class==cls for mid in s.live)<slots:
                if sum(len(s.messages[mid].message.payload) for mid in s.live)+size>w.byte_limit:break
                key=(index,cls,size)
                if key not in self.payload_cache:
                    unit=self.identity_rng.randbytes(256);self.payload_cache[key]=(unit*((size+255)//256))[:size]
                self.desired[cls]+=1;self._admit(index,Message(self.payload_cache[key],cls),self.now)
    def on_verified_delivery(self,index,mid,message):
        if self.now<=self.workload.production_seconds+1e-9:
            self.verified[message.message.traffic_class]+=len(message.message.payload)
            if message.message.traffic_class in ('image','map'):self.host_verified[index//2]+=len(message.message.payload)
        elif message.message.traffic_class in ('image','map'):
            self.drained_verified+=len(message.message.payload)
    def on_confirmed_delivery(self,index,mid,message):
        if self.now<=self.workload.production_seconds+1e-9:
            self.confirmed[message.message.traffic_class]+=len(message.message.payload)
        self._refill(index)
    def done(self):return self.now>=self.horizon-1e-9
    def observe(self):
        obs=super().observe();obs['sender'][:,[SENDER_FEATURES.index(name) for name in ('image_hz','map_hz')]]=0
        if getattr(self,'mode','mixed')=='bulk':obs['sender'][:,SENDER_FEATURES.index('interactive_hz')]=0
        return obs
    def step(self,sender_actions,receiver_actions):
        before=sum(self.verified[c] for c in ('image','map'))
        drained=self.drained_verified
        misses=self.reward_counts['missed'];wire=self.reward_counts['wire']
        obs,_,done,truncated,info=super().step(sender_actions,receiver_actions)
        useful=sum(self.verified[c] for c in ('image','map'))-before
        # Fixed task scale aligns reward with useful bytes, not per-case utilization.
        scale=262144.*self.workload.production_seconds
        expected=max(1,self.count*self.workload.production_seconds*self.workload.interactive_hz)
        reward=(useful+.1*(self.drained_verified-drained))/scale-.02*(self.reward_counts['wire']-wire)/scale
        reward-=DOMAIN['deadline_penalty']*(self.reward_counts['missed']-misses)/expected
        return obs,float(reward),done,truncated,info
    def metrics(self):
        out=super().metrics();w=self.workload
        out.update(profile='goodput-v2',mode=self.mode,measurement_seconds=w.production_seconds,
            drain_verified_bulk_bytes=self.drained_verified,
            verified_bulk_bytes=sum(self.verified[c] for c in ('image','map')),
            confirmed_bulk_bytes=sum(self.confirmed[c] for c in ('image','map')),
            verified_image_bytes=self.verified['image'],verified_map_bytes=self.verified['map'],
            per_host_verified_bytes=[self.host_verified[i] for i in range(w.clients)],
            measurement_wire_bytes=sum(s.wire+s.feedback_wire for s in self.sessions))
        # Wire counter covers measurement plus drain; retain that conservative cost.
        return out
