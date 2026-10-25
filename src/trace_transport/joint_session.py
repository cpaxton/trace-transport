"""Persistent binary sessions with typed events, bounded producers and local actors.

The router never exposes capacities, impairment schedules, or RNG state through
actor observations. Legacy TRC1 simulations remain independent of this backend.
"""
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass,field,replace,asdict
import copy
import heapq
import math
import random

import numpy as np

from .messages import Message,BinaryResult,CLASSES,encode_frame,decode_frame
from .protocol import Sender,Receiver,RATE_MIN,RATE_MAX,WINDOW_MIN,WINDOW_MAX
from .simulator import NetworkConfig,FifoLink
from .session_receiver import SessionReceiver
from .wire import DATA,ACK_RECEIPT
ACK_ACTIONS=((1,0.),)+tuple((b,d) for b in (2,4,8) for d in (.001,.005,.01,.025))
ACK_ACTIONS+=tuple((b,d) for b in (8,16,32) for d in (.025,.1,.25) if (b,d) not in ACK_ACTIONS)
SENDER_FEATURES=('rate','window','flight','ack_rate','wire_rate','rtt','min_rtt','loss_rate',
 'interactive_bytes','image_bytes','map_bytes','bulk_bytes','oldest_age','min_slack',
 'pending','feedback_age','last_rate_action','last_window_action','pressure','age',
 'interactive_hz','image_hz','map_hz','local_ack_pending','local_ack_age')
RECEIVER_FEATURES=('arrivals','unique_rate','duplicates','reordered','pending_acks','oldest_ack',
 'last_gap','arrival_age','interactive_seen','image_seen','map_seen','bulk_seen',
 'completed','emitted','ack_age','last_batch','last_delay','received_bytes','open_messages','age','received_slack','local_interactive_bytes','local_interactive_slack')
QUEUE_FEATURES=('interactive','image','map','bulk','remaining','age','slack','attempted','eligible')
MAX_CANDIDATES=64


class TransmissionMap(dict):
    def __init__(self):super().__init__();self.removed=[]
    def __delitem__(self,key):
        super().__delitem__(key);self.removed.append(key)
    def drain(self):
        removed=self.removed;self.removed=[];return removed


class SessionSender(Sender):
    """Reference reliability semantics with indexed ready and expiry queues."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.in_flight=TransmissionMap()
        self.offsets=list(self.segments);self.unacknowledged=self.offsets.copy();self.fresh_cursor=0;self.retries=[];self.expiries=[]
    def _receipt_coverage(self,prefix,ranges):
        # Already acknowledged segments have attempts>0 and need no further
        # validation or credit. Only scan still-unacknowledged offsets in ranges.
        covered=[]
        for low,high in [(0,prefix),*ranges]:
            start=bisect_left(self.unacknowledged,low)
            end=bisect_right(self.unacknowledged,high)
            covered.extend(self.segments[off] for off in self.unacknowledged[start:end]
                           if off+len(self.segments[off].payload)<=high)
        return covered
    def _ack_index(self,segment):
        position=bisect_left(self.unacknowledged,segment.offset)
        if position<len(self.unacknowledged) and self.unacknowledged[position]==segment.offset:
            self.unacknowledged.pop(position)
    def _expire(self,now):
        while self.expiries and self.expiries[0][0]<=now+1e-10:
            _,tx=heapq.heappop(self.expiries)
            if tx not in self.in_flight:continue
            seg=self.in_flight[tx];del self.in_flight[tx]
            self.flight_bytes-=seg.wire_bytes;seg.transmission_id=None
            self.declared_lost_bytes+=len(seg.payload);heapq.heappush(self.retries,seg.offset)
    def next_timeout(self):
        while self.expiries and self.expiries[0][1] not in self.in_flight:heapq.heappop(self.expiries)
        return self.expiries[0][0] if self.expiries else math.inf
    def poll(self,now):
        packet=super().poll(now)
        if packet is not None:heapq.heappush(self.expiries,(self.segments[packet.offset].timeout_at,packet.transmission_id))
        return packet
    def _eligible(self):
        while self.retries and (self.segments[self.retries[0]].acknowledged or self.segments[self.retries[0]].transmission_id is not None):
            heapq.heappop(self.retries)
        while self.fresh_cursor<len(self.offsets) and self.segments[self.offsets[self.fresh_cursor]].attempts:
            self.fresh_cursor+=1
        seg=self.segments[self.retries[0]] if self.retries else (self.segments[self.offsets[self.fresh_cursor]] if self.fresh_cursor<len(self.offsets) else None)
        if seg is None:return None
        if self.flight_bytes and self.flight_bytes+len(seg.payload)+64>self.cwnd:return None
        if not seg.attempts and self.receiver_credit<len(self.payload):
            outstanding=sum(len(s.payload) for s in self.in_flight.values())
            if outstanding+len(seg.payload)>self.receiver_credit:return None
        return seg


@dataclass(frozen=True)
class WorkloadConfig:
    clients:int=1
    production_seconds:float=20.
    drain_seconds:float=20.
    interactive_bytes:int=1024
    interactive_hz:float=20.
    image_bytes:int=131072
    image_hz:float=2.
    map_bytes:int=524288
    map_interval:float=5.
    interactive_deadline_ms:float=100.
    byte_limit:int=32*1024*1024
    interactive_slots:int=128
    image_slots:int=2
    map_slots:int=1
    shared_airtime:bool=False
    producer_phase_jitter:bool=False

    def __post_init__(self):
        if not 1<=self.clients<=16:raise ValueError('clients must be in [1,16]')
        for key in ('production_seconds','drain_seconds','interactive_hz','image_hz','map_interval','interactive_deadline_ms'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<=0:raise ValueError('invalid workload timing')
        for key in ('interactive_bytes','image_bytes','map_bytes','byte_limit','interactive_slots','image_slots','map_slots'):
            if type(getattr(self,key)) is not int or getattr(self,key)<=0:raise ValueError('invalid workload capacity')
        if max(self.interactive_bytes,self.image_bytes,self.map_bytes)>min(self.byte_limit,16*1024*1024):
            raise ValueError('message cannot fit producer credit')


@dataclass
class MessageState:
    message:Message
    offered:float
    sender:Sender
    delivered:float|None=None
    deadline_counted:bool=False


@dataclass
class Session:
    nonce:int
    direction:int
    messages:dict=field(default_factory=dict)
    live:set=field(default_factory=set)
    inbound:dict=field(default_factory=dict)
    hints:dict=field(default_factory=dict)
    pending:dict=field(default_factory=dict)
    pending_at:dict=field(default_factory=dict)
    ack_generations:dict=field(default_factory=dict)
    completion_seen:set=field(default_factory=set)
    flight:dict=field(default_factory=dict)
    flight_bytes:int=0
    rate:float=128000.
    window:float=16384.
    next_send:float=0.
    wake_generation:int=0
    preferred:int|None=None
    scheduler:str='edf'
    rate_control:str='fixed'
    cooldown:float=0.
    srtt:float=0.
    min_rtt:float=0.
    rttvar:float=0.
    last_feedback:float|None=None
    last_arrival:float|None=None
    last_gap:float=0.
    arrivals:int=0
    duplicates:int=0
    reordered:int=0
    previous_offset:int=0
    max_offsets:dict=field(default_factory=dict)
    wire:int=0
    feedback_wire:int=0
    acked:int=0
    unique:int=0
    lost:int=0
    emitted:int=0
    last_emit:float=0.
    batch:int=8
    delay:float=.025
    ack_mode:str='constant'
    baseline_batch:int=8
    baseline_delay:float=.025
    baseline_ack_mode:str='constant'
    previous_action:tuple=(0.,0.)
    peak_ack_rate:float=0.
    drr_class:int=0
    drr_remaining:int=0
    pressure:int=0
    before:tuple=(0,0,0,0)


class JointTransportEnv:
    def __init__(self,network=None,workload=None,*,auto_producers=True,sender_factory=None):
        self.sender_factory=sender_factory or SessionSender
        self.auto_producers=auto_producers
        self.network=replace(network or NetworkConfig(),feedback_delay_budget=.25)
        self.workload=workload or WorkloadConfig()
        self.horizon=self.workload.production_seconds+self.workload.drain_seconds
        self.count=2*self.workload.clients

    rate_limit=RATE_MAX
    window_limit=WINDOW_MAX

    def on_receiver_control(self,index):pass
    def on_accepted_payload(self,index,mid,traffic_class,size):pass
    def on_verified_delivery(self,index,mid,message):pass
    def on_confirmed_delivery(self,index,mid,message):pass
    def _application_event(self,*args):raise ValueError("unsupported application event")

    def reset(self,*,seed=0):
        self.seed=seed;self.now=0.;self.sequence=0;self.events=[];self.next_message=1
        self.identity_rng=random.Random(seed);self.streams=[random.Random(seed*2+1),random.Random(seed*2+2)]
        self.sessions=[Session(self.identity_rng.getrandbits(64),i%2,rate=self.network.initial_rate,
                         window=self.network.initial_cwnd) for i in range(self.count)]
        self.links=[FifoLink(self.network.bandwidth_bps,self.network.queue_bytes,self.network.capacity_changes),
                    FifoLink(self.network.feedback_bandwidth_bps,self.network.feedback_queue_bytes,self.network.feedback_capacity_changes)]
        self.medium_next=0.;self.airtime_intervals=[];self.processed_events=0
        self.desired=Counter();self.blocked=Counter();self.admitted=Counter();self.payload_cache={};self.unavoidable=0
        self.reward_counts=dict(on_time=0,missed=0,image=0,map=0,wire=0,fresh=0)
        self.receipt_seen=set();self.logs=[]
        w=self.workload
        for index in range(self.count) if self.auto_producers else []:
            phase=self.identity_rng.uniform(0,1/w.interactive_hz) if w.producer_phase_jitter else 0.
            self._schedule(phase,'offer',index,'interactive',1/w.interactive_hz,w.interactive_bytes)
            if index%2==0:
                phase=self.identity_rng.uniform(0,1/w.image_hz) if w.producer_phase_jitter else 0.
                self._schedule(phase,'offer',index,'image',1/w.image_hz,w.image_bytes)
                self._schedule(w.map_interval/2,'offer',index,'map',w.map_interval,w.map_bytes)
        return self.observe(),self.info()

    def _schedule(self,time,kind,*args):
        if math.isfinite(time) and time<=self.horizon+1e-9:
            self.sequence+=1;heapq.heappush(self.events,(max(self.now,time),self.sequence,kind,args))

    def clone(self):
        return copy.deepcopy(self)

    def snapshot(self):
        return copy.deepcopy(self.__dict__)

    def restore(self,snapshot):
        self.__dict__=copy.deepcopy(snapshot)

    def _path_impossible(self,index,size,offered,deadline):
        end=offered+deadline;direction=self.sessions[index].direction
        base=self.network.bandwidth_bps if direction==0 else self.network.feedback_bandwidth_bps
        changes=self.network.capacity_changes if direction==0 else self.network.feedback_capacity_changes
        rates=[base];rtts=[self.network.base_rtt]
        for t,rate in changes:
            if t<=offered:rates=[rate]
            elif t<=end:rates.append(rate)
        for t,rtt in self.network.rtt_changes:
            if t<=offered:rtts=[rtt]
            elif t<=end:rtts.append(rtt)
        minimum=max(0.,min(rtts)/2-self.network.jitter)+size*8/max(rates)
        return minimum>deadline or any(a<=offered and b>=end for a,b in self.network.outages)

    def _offer(self,index,cls,period,size):
        if self.now>=self.workload.production_seconds-1e-9:return
        s=self.sessions[index];self.desired[cls]+=1
        if cls=='interactive' and self._path_impossible(index,size,self.now,self.workload.interactive_deadline_ms/1000):self.unavoidable+=1
        live=[s.messages[mid] for mid in s.live]
        slots={'interactive':self.workload.interactive_slots,'image':self.workload.image_slots,'map':self.workload.map_slots}[cls]
        if sum(m.message.traffic_class==cls for m in live)>=slots or sum(len(m.message.payload) for m in live)+size>self.workload.byte_limit:
            self.blocked[cls]+=1;s.pressure+=1
            if cls=='interactive':self.reward_counts['missed']+=1
        else:
            key=(index,cls,size)
            if key not in self.payload_cache:
                unit=self.identity_rng.randbytes(64)
                if cls=='interactive':
                    unit=('status '+unit.hex()+' 🌐\r\n').encode('utf-8')
                    raw=(unit*((size+len(unit)-1)//len(unit)))[:size]
                    valid=raw.decode('utf-8',errors='ignore').encode('utf-8')
                    self.payload_cache[key]=valid+b' '*(size-len(valid))
                else:self.payload_cache[key]=(unit*((size+63)//64))[:size]
            message=Message(self.payload_cache[key],cls,self.workload.interactive_deadline_ms if cls=='interactive' else None)
            self._admit(index,message,self.now)
        self._schedule(self.now+period,'offer',index,cls,period,size)

    def admit(self,index,message):
        """Explicit binary application admission, with backpressure rather than eviction."""
        if not isinstance(message,Message):raise TypeError('expected Message')
        s=self.sessions[index]
        self.desired[message.traffic_class]+=1
        live=[s.messages[mid] for mid in s.live]
        slots={'interactive':self.workload.interactive_slots,'image':self.workload.image_slots,
               'map':self.workload.map_slots,'bulk':MAX_CANDIDATES}[message.traffic_class]
        if sum(len(m.message.payload) for m in live)+len(message.payload)>self.workload.byte_limit or sum(m.message.traffic_class==message.traffic_class for m in live)>=slots:
            self.blocked[message.traffic_class]+=1
            if message.traffic_class=='interactive':self.reward_counts['missed']+=1
            raise BufferError('session producer credit exhausted')
        return self._admit(index,message,self.now)

    def _admit(self,index,message,offered):
        mid=self.next_message;self.next_message+=1;s=self.sessions[index]
        sender=self.sender_factory(message.payload,mid,payload_size=self.network.payload_size,rate=RATE_MAX,
                      cwnd=WINDOW_MAX,feedback_delay_budget=self.network.feedback_delay_budget)
        if s.srtt:sender.srtt, sender.rttvar,sender.min_rtt=s.srtt,s.rttvar,s.min_rtt
        s.messages[mid]=MessageState(message,offered,sender);s.live.add(mid)
        self.admitted[message.traffic_class]+=1;self._wake_at(index)
        if message.deadline_ms is not None:self._schedule(offered+message.deadline_ms/1000,'deadline',index,mid)
        return mid

    def _deadline(self,index,mid):
        m=self.sessions[index].messages[mid]
        if m.delivered is None and not m.deadline_counted:
            self.reward_counts['missed']+=1;m.deadline_counted=True

    def _release(self,s,mid):
        for tx in s.messages[mid].sender.in_flight.drain():
            s.flight_bytes-=s.flight.pop((mid,tx),0)

    def _refresh(self,s):
        for mid in s.live:
            m=s.messages[mid]
            before=m.sender.declared_lost_bytes;m.sender._expire(self.now)
            s.lost+=m.sender.declared_lost_bytes-before;self._release(s,mid)

    def _eligible(self,s):
        return [(mid,s.messages[mid]) for mid in sorted(s.live) if s.messages[mid].sender._eligible() is not None]

    def _choose(self,s,ready):
        if not ready:return None
        if s.preferred is not None:
            for mid,m in ready:
                if mid==s.preferred:return mid,m
        if s.scheduler=='fifo':return min(ready,key=lambda x:(x[1].offered,x[0]))
        if s.scheduler=='drr':
            groups=('interactive','image','map','bulk');quanta=(8192,2048,1024,1024)
            for _ in range(8):
                cls=groups[s.drr_class]
                group=[x for x in ready if x[1].message.traffic_class==cls]
                if group:
                    if s.drr_remaining<=0:s.drr_remaining=quanta[s.drr_class]
                    selected=min(group,key=lambda x:(x[1].offered,x[0]))
                    return selected
                s.drr_class=(s.drr_class+1)%4;s.drr_remaining=0
        return min(ready,key=lambda x:(x[1].offered+x[1].message.deadline_ms/1000
                    if x[1].message.deadline_ms is not None else math.inf,x[1].offered,x[0]))

    def _wake_at(self,index):
        s=self.sessions[index];s.wake_generation+=1;self._refresh(s)
        timeout=min((s.messages[mid].sender.next_timeout() for mid in s.live),default=math.inf)
        ready=self._eligible(s)
        flight=s.flight_bytes
        when=max(self.now,s.next_send) if ready and (not flight or flight+self.network.payload_size+512<=s.window) else math.inf
        self._schedule(min(when,timeout),'wake',index,s.wake_generation)

    def _wake(self,index,generation):
        s=self.sessions[index]
        if generation!=s.wake_generation:return
        self._refresh(s)
        selected=self._choose(s,self._eligible(s))
        if selected is not None and self.now>=s.next_send-1e-9:
            mid,m=selected;flight=s.flight_bytes
            if not flight or flight+self.network.payload_size+512<=s.window:
                packet=m.sender.poll(self.now)
                if packet is not None:
                    meta={'class':m.message.traffic_class}
                    if m.message.deadline_ms is not None:meta['remaining_ms']=m.message.deadline_ms-(self.now-m.offered)*1000
                    packet=replace(packet,metadata=meta)
                    wire=encode_frame(s.nonce,packet);size=len(wire)
                    s.flight[(mid,packet.transmission_id)]=size;s.flight_bytes+=size;s.next_send=self.now+size/s.rate
                    s.wire+=size;self.reward_counts['wire']+=size
                    if m.sender.segments[packet.offset].attempts==1:self.reward_counts['fresh']+=len(packet.payload)
                    if s.scheduler=='drr':
                        s.drr_remaining-=size
                        if s.drr_remaining<=0:s.drr_class=(s.drr_class+1)%4
                    self._transmit(index,False,wire)
        self._wake_at(index)

    def _outage(self):
        return any(a<=self.now<b for a,b in self.network.outages)

    def _transmit(self,index,feedback,wire):
        s=self.sessions[index];direction=1-s.direction if feedback else s.direction
        link=self.links[direction];rng=self.streams[direction];cfg=self.network
        copies=1+int(rng.random()<cfg.duplicate);link.counters.generated_duplicates+=copies-1
        for _ in range(copies):
            if self.workload.shared_airtime:link.next_free=max(link.next_free,self.medium_next)
            start=max(self.now,link.next_free);finish=link.enqueue(self.now,len(wire))
            if finish is None:continue
            if self.workload.shared_airtime:
                self.medium_next=finish;self.airtime_intervals.append((start,finish,direction))
            loss=cfg.ack_loss if direction else cfg.loss
            for time,a,b in cfg.loss_changes:
                if time>self.now:break
                loss=b if direction else a
            if self._outage():link.counters.outage_drops+=1;continue
            if rng.random()<loss:link.counters.random_drops+=1;continue
            out=wire;corruption=cfg.ack_corruption if direction else cfg.corruption
            if corruption and rng.random()<corruption:
                damaged=bytearray(wire);damaged[rng.randrange(len(out))]^=1<<rng.randrange(8)
                out=bytes(damaged);link.counters.corrupted_packets+=1
            rtt=cfg.base_rtt
            for time,value in cfg.rtt_changes:
                if time>self.now:break
                rtt=value
            delay=max(0.,rtt/2+rng.uniform(-cfg.jitter,cfg.jitter))
            self._schedule(finish+delay,'deliver',index,feedback,out)

    def _deliver(self,index,feedback,wire):
        s=self.sessions[index];link=self.links[1-s.direction if feedback else s.direction]
        if self._outage():link.counters.outage_drops+=1;return
        try:
            nonce,packet=decode_frame(wire)
            if nonce!=s.nonce:raise ValueError('wrong session')
        except ValueError:
            link.counters.corruption_rejections+=1;return
        link.counters.delivered_packets+=1;mid=packet.message_id
        if feedback:
            m=s.messages.get(mid)
            if m is None:return
            before=m.sender.acknowledged_bytes;n=len(m.sender.rtt_samples)
            if m.sender.on_ack(packet,self.now):
                s.acked+=m.sender.acknowledged_bytes-before;s.last_feedback=self.now
                if len(m.sender.rtt_samples)>n:
                    s.srtt,s.rttvar,s.min_rtt=m.sender.srtt,m.sender.rttvar,m.sender.min_rtt
                    for other in s.live:
                        peer=s.messages[other].sender
                        peer.srtt,peer.rttvar,peer.min_rtt=s.srtt,s.rttvar,s.min_rtt
            self._release(s,mid)
            if m.sender.complete and mid in s.live:
                s.live.discard(mid);self.on_confirmed_delivery(index,mid,m)
            self._wake_at(index)
        else:
            if packet.kind!=DATA:return
            cls=packet.metadata.get('class')
            if cls not in CLASSES:return
            if mid not in s.inbound:
                allocated=sum(r.total or 0 for r in s.inbound.values() if not r.complete)
                if allocated+packet.total>self.workload.byte_limit:return
                s.inbound[mid]=SessionReceiver()
                s.hints[mid]=dict(traffic_class=cls,remaining_ms=packet.metadata.get('remaining_ms'),arrived=self.now)
            r=s.inbound[mid];before=r.received_bytes;was_complete=r.complete
            ack=r.on_data(packet,self.now)
            if ack is None:return
            s.hints[mid].update(remaining_ms=packet.metadata.get('remaining_ms'),arrived=self.now)
            was_reordered=packet.offset<s.max_offsets.get(mid,0)
            s.max_offsets[mid]=max(packet.offset,s.max_offsets.get(mid,0))
            s.arrivals+=1;s.duplicates+=before==r.received_bytes;s.reordered+=was_reordered
            s.previous_offset=packet.offset;s.last_gap=self.now-s.last_arrival if s.last_arrival is not None else 0.;s.last_arrival=self.now
            delta=r.received_bytes-before;s.unique+=delta
            if delta:self.on_accepted_payload(index,mid,cls,delta)
            if cls in ('image','map'):self.reward_counts[cls]+=delta
            if r.complete and not was_complete:
                payload=r.reconstruct()
                # This assertion is evaluation truth, never actor input.
                expected=s.messages[mid].message.payload
                if payload!=expected:raise AssertionError('corrupt application delivery')
                m=s.messages[mid];m.delivered=self.now
                self.on_verified_delivery(index,mid,m)
                if m.message.deadline_ms is not None and self.now-m.offered<=m.message.deadline_ms/1000+1e-9:
                    if m.deadline_counted:self.reward_counts['missed']-=1;m.deadline_counted=False
                    self.reward_counts['on_time']+=1
            first=ack.metadata['complete'] and mid not in s.completion_seen
            if ack.metadata['complete']:s.completion_seen.add(mid)
            s.pending.setdefault(mid,{})[packet.offset]=(ack,self.now)
            s.pending_at.setdefault(mid,self.now)
            batch,delay=s.batch,s.delay
            if s.ack_mode=='deadline' and (cls=='interactive' or was_reordered or packet.offset>ack.metadata.get('prefix',0)):batch,delay=1,0.
            if first or len(s.pending[mid])>=batch or delay==0:self._flush(index,mid)
            elif len(s.pending[mid])==1:
                generation=s.ack_generations.get(mid,0)
                self._schedule(self.now+delay,'flush',index,mid,generation)

    def _flush(self,index,mid,generation=None):
        s=self.sessions[index]
        if generation is not None and generation!=s.ack_generations.get(mid,0):return
        pending=s.pending.pop(mid,None)
        if not pending:return
        ack,time=max(pending.values(),key=lambda x:x[1]);s.pending_at.pop(mid,None)
        s.ack_generations[mid]=s.ack_generations.get(mid,0)+1
        packet=replace(ack,metadata={**ack.metadata,'ack_delay':min(.25,max(0.,self.now-time))})
        wire=encode_frame(s.nonce,packet);s.feedback_wire+=len(wire);s.emitted+=1;s.last_emit=self.now
        self.reward_counts['wire']+=len(wire);self._transmit(index,True,wire)

    def _advance(self,target):
        while self.events and self.events[0][0]<=target+1e-10:
            time,_,kind,args=heapq.heappop(self.events);self.now=time
            {'offer':self._offer,'wake':self._wake,'deliver':self._deliver,'flush':self._flush,'deadline':self._deadline,'application':self._application_event}[kind](*args)
            self.processed_events+=1
        self.now=target

    def done(self):
        return self.now>=self.horizon-1e-9 or (self.now>=self.workload.production_seconds and
            all(not s.live for s in self.sessions))

    def candidates(self,s):
        live=[(mid,s.messages[mid]) for mid in s.live]
        return sorted(live,key=lambda x:(x[1].message.traffic_class!='interactive',x[1].offered,x[0]))[:MAX_CANDIDATES]

    def baseline(self,index,rate_control='fixed',scheduler='edf',feedback='budgeted'):
        s=self.sessions[index];dt=self.network.control_interval
        ack_rate=(s.acked-s.before[0])/dt;loss=s.lost>s.before[3]
        a=np.zeros(2,dtype=np.float32)
        if rate_control in ('aimd','bdp','deadline'):
            inflated=s.srtt>max(.02,1.5*s.min_rtt)
            if self.now>=s.cooldown and (loss or inflated):
                a[0]=-1.;s.cooldown=self.now+max(dt,s.srtt)
            elif ack_rate>0:a[0]=min(1.,math.log1p(2000/max(s.rate,1))/.2)
        if rate_control=='bdp':
            s.peak_ack_rate=max(.95*s.peak_ack_rate,ack_rate)
            if s.last_feedback is not None:
                target=min(WINDOW_MAX,max(16384,2*s.peak_ack_rate*s.min_rtt+8192))
                a[1]=np.clip(math.log(target/s.window)/.2,-1,1)
        if rate_control=='delay' and s.last_feedback is not None:
            # Delay-target baseline inspired by Copa's rate/queue-delay relation;
            # not a reproduction of Copa's complete Internet coexistence logic.
            qdelay=max(.001,s.srtt-s.min_rtt)
            target=min(self.rate_limit,max(RATE_MIN,self.network.payload_size/(.5*qdelay)))
            a[0]=np.clip(math.log(target/s.rate)/.2,-1,1)
            window=min(self.window_limit,max(WINDOW_MIN,1.5*target*s.min_rtt+2*self.network.payload_size))
            a[1]=np.clip(math.log(window/s.window)/.2,-1,1)
        if rate_control=='deadline':
            # Endpoint-local arrival-rate estimate; no router or link capacity.
            measured=s.acked/max(self.now,max(s.min_rtt,.02))
            if s.last_feedback is not None:
                target=min(WINDOW_MAX,max(WINDOW_MIN,measured*(s.min_rtt+.02)+self.network.payload_size))
                a[1]=np.clip(math.log(target/s.window)/.2,-1,1)
        return a

    def step(self,sender_actions,receiver_actions):
        if self.done():raise RuntimeError('session finished')
        actions=np.asarray(sender_actions,dtype=float);rx=np.asarray(receiver_actions)
        if actions.shape!=(self.count,3) or not np.isfinite(actions).all():raise ValueError('invalid sender actions')
        if rx.shape!=(self.count,) or not np.isfinite(rx).all() or np.any(rx!=rx.astype(int)) or np.any(rx<0) or np.any(rx>len(ACK_ACTIONS)):
            raise ValueError('invalid receiver actions')
        before=self.reward_counts.copy();old=self.now
        for i,s in enumerate(self.sessions):
            base=self.baseline(i,s.rate_control)
            delta=np.clip(base+2*np.clip(actions[i,:2],-1,1),-1,1)
            s.rate=float(np.clip(s.rate*math.exp(.2*delta[0]),RATE_MIN,self.rate_limit))
            s.window=float(np.clip(s.window*math.exp(.2*delta[1]),WINDOW_MIN,self.window_limit))
            s.previous_action=tuple(delta)
            if actions[i,2]!=int(actions[i,2]):raise ValueError("scheduling choice must be an integer")
            choice=int(actions[i,2]);candidates=self.candidates(s)
            if choice<0 or choice>len(candidates):raise ValueError('invalid scheduling action')
            s.preferred=candidates[choice-1][0] if choice else None
            if rx[i]:s.batch,s.delay=ACK_ACTIONS[int(rx[i])-1];s.ack_mode='constant'
            else:s.batch,s.delay,s.ack_mode=s.baseline_batch,s.baseline_delay,s.baseline_ack_mode
            self.on_receiver_control(i)
            s.before=(s.acked,s.wire,s.unique,s.lost);self._wake_at(i)
        self._advance(min(self.horizon,self.now+self.network.control_interval))
        d={k:self.reward_counts[k]-before[k] for k in before};w=self.workload
        expected=max(1,self.count*w.production_seconds*w.interactive_hz)
        reward=(d['on_time']-4*d['missed'])/expected
        reward+=.25*d['image']/max(1,w.clients*w.production_seconds*w.image_hz*w.image_bytes)
        reward+=.25*d['map']/max(1,w.clients*w.production_seconds/w.map_interval*w.map_bytes)
        target_bytes=expected*w.interactive_bytes+w.clients*w.production_seconds*(w.image_hz*w.image_bytes+w.map_bytes/w.map_interval)
        reward-=.1*max(0,d['wire']-d['fresh'])/max(1,target_bytes)
        waiting=sum(min(10,(self.now-m.offered)/(w.interactive_deadline_ms/1000)) for s in self.sessions for m in s.messages.values()
                    if m.message.traffic_class=='interactive' and m.delivered is None)
        reward-=.05*waiting*(self.now-old)/expected
        terminal=self.done()
        if terminal:
            for cls in CLASSES:
                admitted=sum(len(m.message.payload) for s in self.sessions for m in s.messages.values() if m.message.traffic_class==cls)
                missing=sum(len(m.message.payload) for s in self.sessions for m in s.messages.values() if m.message.traffic_class==cls and not m.sender.complete)
                reward-=2*missing/max(1,admitted)
        return self.observe(),float(reward),terminal,False,self.info()

    def observe(self):
        send=[];recv=[];queues=np.zeros((self.count,MAX_CANDIDATES,9),np.float32);mask=np.zeros((self.count,MAX_CANDIDATES),bool)
        sa=[];ra=[]
        log=lambda x,scale=1:math.log1p(max(0.,x)/scale)
        dt=self.network.control_interval
        for i,s in enumerate(self.sessions):
            local=self.sessions[i^1]
            local_interactive=[local.messages[mid] for mid in local.live if local.messages[mid].message.traffic_class=='interactive']
            live=[s.messages[mid] for mid in s.live]
            bytes_by=[sum(len(m.message.payload)-m.sender.acknowledged_bytes for m in live if m.message.traffic_class==cls) for cls in CLASSES]
            ages=[self.now-m.offered for m in live]
            slack=[(m.offered+m.message.deadline_ms/1000-self.now) for m in live if m.message.deadline_ms is not None]
            send.append([log(s.rate,128000),log(s.window,16384),log(s.flight_bytes,16384),
                log((s.acked-s.before[0])/dt,128000),log((s.wire-s.before[1])/dt,128000),s.srtt/.1,s.min_rtt/.1,
                log((s.lost-s.before[3])/dt,16384),*[log(b,65536) for b in bytes_by],
                log(max(ages,default=0)),np.clip(min(slack,default=1.)/.1,-10,10),log(len(live)),
                log(self.now-s.last_feedback if s.last_feedback is not None else self.now),*s.previous_action,log(s.pressure),log(self.now),log(self.workload.interactive_hz,20),
                log(self.workload.image_hz if i%2==0 else 0,2),log(1/self.workload.map_interval if i%2==0 else 0,.2),log(sum(len(x) for x in local.pending.values()),8),
                min(10,max((self.now-t for t in local.pending_at.values()),default=0.)/.25)])
            seen=Counter(h['traffic_class'] for h in s.hints.values())
            recv.append([log(s.arrivals,32),log((s.unique-s.before[2])/dt,128000),log(s.duplicates,8),log(s.reordered,8),
                log(sum(len(x) for x in s.pending.values()),8),max((self.now-t for t in s.pending_at.values()),default=0.)/.25,
                min(10,s.last_gap/.02),min(10,(self.now-s.last_arrival if s.last_arrival is not None else self.now)/.1),
                *[log(seen[c],16) for c in CLASSES],log(sum(r.complete for r in s.inbound.values()),16),log(s.emitted,32),
                min(10,(self.now-s.last_emit)/.1),s.batch/32,s.delay/.25,log(s.unique,65536),log(sum(not r.complete for r in s.inbound.values()),8),log(self.now),
                np.clip(min(((h['remaining_ms']/1000-(self.now-h['arrived']))/.1 for mid,h in s.hints.items()
                    if h['remaining_ms'] is not None and mid in s.inbound and (not s.inbound[mid].complete or mid in s.pending)),default=10),-10,10),log(sum(len(m.message.payload) for m in local_interactive),4096),
                np.clip(min(((m.offered+m.message.deadline_ms/1000-self.now)/.1 for m in local_interactive),default=10),-10,10)])
            for j,(mid,m) in enumerate(self.candidates(s)):
                cls=m.message.traffic_class;remaining=len(m.message.payload)-m.sender.acknowledged_bytes
                queues[i,j]=[*[float(cls==c) for c in CLASSES],log(remaining,65536),log(self.now-m.offered),
                    np.clip((m.offered+m.message.deadline_ms/1000-self.now)/.1,-10,10) if m.message.deadline_ms else 10,
                    float(m.sender.next_transmission_id>1),float(m.sender._eligible() is not None)]
                mask[i,j]=bool(m.sender._eligible() is not None)
            sa.append(bool(live));ra.append(bool(s.pending or any(not r.complete for r in s.inbound.values())))
        return dict(sender=np.asarray(send,np.float32),receiver=np.asarray(recv,np.float32),queue=queues,queue_mask=mask,
                    sender_active=np.asarray(sa,bool),receiver_active=np.asarray(ra,bool))

    def critic_state(self,obs=None):
        """Current hidden state only; never upcoming schedules or RNG values."""
        obs=self.observe() if obs is None else obs;network=[]
        for direction,link in enumerate(self.links):
            rate=link.rate
            for time,value in link.changes:
                if time>self.now:break
                rate=value
            network.extend([math.log1p(rate/1000000),link.occupancy/65536])
        return np.asarray([*obs['sender'].mean(0),*obs['receiver'].mean(0),*network,self.now/self.horizon,self.count/32],np.float32)

    def info(self):
        return dict(time=self.now,active_sessions=self.count,events=self.processed_events)

    def results(self):
        result=[]
        for s in self.sessions:
            for mid,m in s.messages.items():
                r=s.inbound.get(mid);payload=r.reconstruct() if r is not None and r.complete else None
                if payload is not None and payload!=m.message.payload:raise AssertionError('wrong binary payload')
                result.append(BinaryResult(mid,m.message.traffic_class,payload,m.sender.complete and payload is not None,
                    payload is not None,m.offered,m.offered,m.delivered,m.sender.completed_at,m.message.deadline_ms))
        return result

    def metrics(self):
        rows=[r.summary() for r in self.results()];w=self.workload
        interactive=[r for r in rows if r['traffic_class']=='interactive']
        desired=self.desired['interactive'];on_time=sum(r['on_time'] for r in interactive)
        pending=sum(not r['receiver_complete'] and self.now<r['offered_at']+r['deadline_ms']/1000-1e-9 for r in interactive)
        out=dict(desired=dict(self.desired),blocked=dict(self.blocked),admitted=dict(self.admitted),
            messages=len(rows),completed=sum(r['completed'] for r in rows),wrong=0,
            interactive_miss_rate=(desired-on_time-pending)/max(1,desired),interactive_on_time=on_time,interactive_pending_deadlines=pending,
            interactive_unavoidable_bound=self.unavoidable,interactive_avoidable_misses=max(0,desired-on_time-pending-self.unavoidable),
            image_bytes=sum(r['delivered_bytes'] for r in rows if r['traffic_class']=='image'),
            map_bytes=sum(r['delivered_bytes'] for r in rows if r['traffic_class']=='map'),
            delivered_bytes=sum(r['delivered_bytes'] for r in rows),
            total_wire_bytes=sum(s.wire+s.feedback_wire for s in self.sessions),
            feedback_bytes=sum(s.feedback_wire for s in self.sessions),elapsed=self.now,
            actual_image_fps=sum(r['receiver_complete'] for r in rows if r['traffic_class']=='image')/max(1,w.clients*self.now),
            admitted_image_fps=self.admitted['image']/max(1,w.clients*w.production_seconds),
            pending_confirmation_bytes=sum(len(m.message.payload)-m.sender.acknowledged_bytes for s in self.sessions for mid,m in s.messages.items() if mid in s.live),
            pending_messages=sum(len(s.live) for s in self.sessions),events=self.processed_events,
            messages_detail=rows,network=[asdict(link.counters) for link in self.links])
        latencies=[r['delivered_at']-r['offered_at'] for r in interactive if r['receiver_complete']]
        out['interactive_p95']=float(np.quantile(latencies,.95)) if latencies else None
        return out
