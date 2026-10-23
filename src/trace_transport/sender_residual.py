"""Learn a bounded correction to endpoint-local AIMD rate/window decisions.

Receiver receipt scheduling is frozen at batch-8/25-ms. This isolates sender
control after measuring the limited receiver-only action-space headroom.
"""
from dataclasses import asdict
import math
from pathlib import Path
import numpy as np
import torch

from .benchmark import benchmark_text
from .controllers import AimdController,FixedController
from .environment import TransportEnv,FEATURE_INDEX as I,FEATURE_NAMES
from .learning import HistoryTransformer,ModelConfig,empty_history,append_history

SCALE=.5


class BdpAimdController(AimdController):
    """AIMD rate plus window derived from local ACK-rate and minimum RTT.

    A simple additional transport baseline, not an implementation of TCP/BBR.
    """
    def reset(self,observations):
        super().reset(observations)
        self.peak=np.zeros(self.flows)

    def act(self,observations):
        actions=super().act(observations)
        for i,row in enumerate(observations):
            ack_rate=math.expm1(float(row[I['acked_payload_rate']]))*128000
            self.peak[i]=max(.95*self.peak[i],ack_rate)
            if row[I['feedback_seen']]>0:
                target=min(4*1024*1024,max(16384,2*self.peak[i]*float(row[I['minimum_rtt']])*.1+8192))
                window=math.expm1(float(row[I['congestion_window']]))*16384
                actions[i,1]=np.clip(math.log(target/max(window,1))/.2,-1,1)
        return actions


def make_baseline(name,flows):
    cls={'aimd':AimdController,'fixed':FixedController,'bdp-aimd':BdpAimdController}[name]
    return cls(flows,action_dimensions=2)


def make_model(seed=0,history=8):
    torch.manual_seed(seed)
    return HistoryTransformer(ModelConfig(history=history,width=32,heads=4,layers=1,action_dimensions=2))


def save(path,model,training):
    torch.save(dict(format_version='sender-residual-v1',feature_names=list(FEATURE_NAMES),
                    model_config=asdict(model.config),state_dict=model.state_dict(),training=training,
                    baseline='bdp-aimd',residual_scale=SCALE),Path(path))


def load(path):
    d=torch.load(path,map_location='cpu',weights_only=True)
    if d.get('format_version')!='sender-residual-v1' or d.get('feature_names')!=list(FEATURE_NAMES):
        raise ValueError('incompatible residual sender checkpoint')
    if d.get('baseline')!='bdp-aimd' or d.get('residual_scale')!=SCALE:
        raise ValueError('incompatible residual control mapping')
    model=HistoryTransformer(ModelConfig(**d['model_config']))
    model.load_state_dict(d['state_dict'])
    return model.eval()


def episode(config,*,seed,payload_bytes=32768,flows=2,model=None,baseline='bdp-aimd',stochastic=False,texts=None,start_times=None):
    texts=[benchmark_text(payload_bytes)]*flows if texts is None else list(texts)
    flows=len(texts)
    env=TransportEnv(texts,config,action_dimensions=2,start_times=start_times,reward_profile='completion',
                     receiver_feedback=True,receipt_feedback=True)
    obs,_=env.reset(seed=seed)
    controller=make_baseline(baseline,flows);controller.reset(obs)
    h,v=empty_history(flows,model.config) if model else (None,None)
    trajectory=[];score=np.zeros(flows)
    while not env.simulator.finished:
        actions=controller.act(obs)
        if model:
            append_history(h,v,obs)
            with torch.no_grad():
                history,valid=torch.from_numpy(h.copy()),torch.from_numpy(v.copy())
                action,latent,logp,value=model.sample(history,valid)
                if not stochastic:
                    action=model(history,valid)[0].tanh()
            actions=np.clip(actions+SCALE*action.numpy(),-1,1)
        for feedback in env.simulator.feedback_schedulers:
            feedback.batch_size,feedback.delay=8,.025
        active=torch.tensor([not s.complete for s in env.simulator.senders])
        obs,reward,terminated,truncated,_=env.step(actions)
        score+=reward
        if model and stochastic:
            trajectory.append((history,valid,latent,logp,value,torch.from_numpy(reward),
                               torch.from_numpy(terminated|truncated),active))
    return env.simulator.results(),score,trajectory
