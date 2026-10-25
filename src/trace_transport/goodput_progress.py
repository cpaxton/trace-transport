"""V5 diagnostics and optional potential shaping; exact object scoring preserved."""
from collections import Counter

from .goodput_deployment import DOMAIN as V4_DOMAIN,sample_case as v4_case,GoodputDeploymentEnv
from .learning_diagnostics import completion_headroom
from .messages import CLASSES

DOMAIN={**V4_DOMAIN,'version':'goodput-domain-v5.0','reward':'verified-goodput-progress-v5.0',
        'training_progress':'optional discounted potential of CRC-accepted incomplete bulk payload',
        'gamma_reference_seconds':.02,'gamma':.999,
        'evaluation_contract':'whole-object goodput unchanged; service guard uses accepted bulk bytes; report whole-object and class completion separately',
        'host_guard':'zero accepted bulk bytes per configured host during measurement',
        'fairness_margin_absolute':.05}


def sample_case(seed,**kwargs):
    cfg,w,label=v4_case(seed,**kwargs);label['version']=DOMAIN['version'];return cfg,w,label


class GoodputProgressEnv(GoodputDeploymentEnv):
    def __init__(self,*args,progress_shaping=False,**kwargs):
        super().__init__(*args,**kwargs);self.progress_shaping=progress_shaping
    def reset(self,*,seed=0):
        self.accepted_bulk=0;self.accepted_by_host=Counter();self.incomplete_bulk=0
        return super().reset(seed=seed)
    def on_accepted_payload(self,index,mid,traffic_class,size):
        if traffic_class not in ('image','map'):return
        self.incomplete_bulk+=size
        if self.now<=self.workload.production_seconds+1e-9:
            self.accepted_bulk+=size;self.accepted_by_host[index//2]+=size
    def on_verified_delivery(self,index,mid,message):
        super().on_verified_delivery(index,mid,message)
        if message.message.traffic_class in ('image','map'):
            self.incomplete_bulk-=len(message.message.payload)
            if self.incomplete_bulk<0:raise AssertionError('incomplete payload counter underflow')
    def potential(self):
        # Same fixed scale as the raw goodput reward; never physical capacity.
        return 0. if self.done() else self.incomplete_bulk/(262144.*self.workload.production_seconds)
    def step(self,sender_actions,receiver_actions):
        before=self.potential();obs,reward,done,truncated,info=super().step(sender_actions,receiver_actions)
        if self.progress_shaping:
            discount=DOMAIN['gamma']**(obs['transition_seconds']/DOMAIN['gamma_reference_seconds'])
            reward+=discount*(0. if done else self.potential())-before
        return obs,float(reward),done,truncated,info
    def metrics(self):
        out=super().metrics();rows=self.results()
        out['class_completion_after_drain']={c:dict(admitted=sum(r.traffic_class==c for r in rows),
            receiver_complete=sum(r.traffic_class==c and r.receiver_complete for r in rows),
            confirmed=sum(r.traffic_class==c and r.completed for r in rows)) for c in CLASSES}
        out.update(profile='goodput-v5',
            accepted_bulk_payload_bytes=self.accepted_bulk,
            accepted_bulk_payload_by_host=[self.accepted_by_host[i] for i in range(self.workload.clients)],
            measurement_incomplete_bulk_payload_bytes=self.accepted_bulk-out['verified_bulk_bytes'],
            completion_headroom=completion_headroom(self.network,self.workload))
        return out
