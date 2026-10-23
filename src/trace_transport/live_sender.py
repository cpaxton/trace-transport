"""Experimental local sender actor for single-file TRC2 transfers.

This is sender-only cross-play with a deterministic receiver. Single-file traffic
is outside the mixed persistent workload used for training; report correctness
and performance separately. Queue selection is trivial with one pending message.
"""
import math
import numpy as np
from .joint_models import load_checkpoint
from .joint_session import CLASSES,MAX_CANDIDATES,SENDER_FEATURES


def sender_observation(sender,now,traffic_class,before,dt,deadline_ms=None):
    log=lambda x,scale=1:math.log1p(max(0.,x)/scale)
    remaining=len(sender.payload)-sender.acknowledged_bytes
    slack=np.clip((deadline_ms/1000-now)/.1,-10,10) if deadline_ms is not None else 10.
    features=[log(sender.rate,128000),log(sender.cwnd,16384),log(sender.flight_bytes,16384),
        log((sender.acknowledged_bytes-before[0])/dt,128000),log((sender.wire_sent-before[1])/dt,128000),
        sender.srtt/.1,sender.min_rtt/.1,log((sender.declared_lost_bytes-before[2])/dt,16384),
        *[log(remaining if traffic_class==c else 0,65536) for c in CLASSES],
        log(now),slack,log(1),log(now-sender.last_ack_at if sender.last_ack_at is not None else now),
        *sender.previous_action,0.,log(now),0.,0.,0.,0.,0.]
    queue=np.zeros((1,MAX_CANDIDATES,9),np.float32);mask=np.zeros((1,MAX_CANDIDATES),bool)
    eligible=sender._eligible() is not None
    queue[0,0]=[*[float(traffic_class==c) for c in CLASSES],log(remaining,65536),log(now),slack,
                  float(sender.next_transmission_id>1),float(eligible)]
    mask[0,0]=eligible
    return np.asarray(features,np.float32),queue,mask


class JointFileSender:
    def __init__(self,checkpoint,traffic_class,device='cpu'):
        import torch
        torch.set_num_threads(1)
        self.actor,_,self.config,metadata=load_checkpoint(checkpoint,device)
        if metadata.get('profile','deadline-v1')!='deadline-v1':
            raise ValueError('this live adapter does not compose the goodput-v2 adaptive teacher')
        self.device=device;self.traffic_class=traffic_class
        self.history=np.zeros((1,self.config.history,len(SENDER_FEATURES)),np.float32)
        self.valid=np.zeros((1,self.config.history),bool)
        self.before=(0,0,0);self.last=None;self.decisions=0
    def tick(self,sender,now):
        import torch
        if self.last is not None and now-self.last<.02:return
        dt=max(.02,now-self.last) if self.last is not None else .02
        features,queue,mask=sender_observation(sender,now,self.traffic_class,self.before,dt)
        self.history[:,:-1]=self.history[:,1:].copy();self.history[:,-1]=features
        self.valid[:,:-1]=self.valid[:,1:].copy();self.valid[:,-1]=True
        with torch.inference_mode():
            action=self.actor.act(*[torch.as_tensor(x,device=self.device) for x in (self.history,self.valid,queue,mask)])[0].cpu().numpy()
        sender.apply_action(np.clip(2*action[:2],-1,1))
        self.before=(sender.acknowledged_bytes,sender.wire_sent,sender.declared_lost_bytes)
        self.last=now;self.decisions+=1
