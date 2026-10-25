"""Separate local actors and a centralized training-only critic."""
from dataclasses import dataclass,asdict
from pathlib import Path
import math

import torch
from torch import nn
from torch.nn import functional as F

from .learning import AttentionBlock
from .joint_session import SENDER_FEATURES,RECEIVER_FEATURES,QUEUE_FEATURES,MAX_CANDIDATES,ACK_ACTIONS


@dataclass(frozen=True)
class JointModelConfig:
    history:int=32
    width:int=64
    layers:int=2
    heads:int=4
    def __post_init__(self):
        if self.history<1 or self.width<1 or self.layers<1 or self.heads<1 or self.width%self.heads:
            raise ValueError('invalid joint model dimensions')


class LocalEncoder(nn.Module):
    def __init__(self,features,config):
        super().__init__();self.config=config
        self.input=nn.Linear(features,config.width)
        self.position=nn.Parameter(torch.randn(config.history,config.width)*.02)
        self.blocks=nn.ModuleList([AttentionBlock(config.width,config.heads) for _ in range(config.layers)])
        self.norm=nn.LayerNorm(config.width)
    def forward(self,h,valid):
        n=h.shape[1];causal=torch.ones(n,n,dtype=torch.bool,device=h.device).tril()
        allowed=causal[None]&valid[:,None,:]
        allowed=allowed|(torch.eye(n,dtype=torch.bool,device=h.device)[None]&~valid[:,:,None])
        x=self.input(h.masked_fill(~valid[:,:,None],0))+self.position
        for block in self.blocks:x=block(x,allowed)
        return self.norm(x[:,-1])


class SenderActor(nn.Module):
    def __init__(self,config):
        super().__init__();self.encoder=LocalEncoder(len(SENDER_FEATURES),config)
        self.control=nn.Linear(config.width,2);nn.init.zeros_(self.control.weight);nn.init.zeros_(self.control.bias)
        self.log_std=nn.Parameter(torch.full((2,),-2.3))
        self.queue=nn.Sequential(nn.Linear(len(QUEUE_FEATURES),config.width),nn.GELU(),nn.Linear(config.width,config.width))
        self.query=nn.Linear(config.width,config.width);nn.init.zeros_(self.query.weight);nn.init.zeros_(self.query.bias)
        self.follow=nn.Linear(config.width,1);nn.init.zeros_(self.follow.weight);nn.init.constant_(self.follow.bias,.5)
    def forward(self,h,valid,queue,mask):
        state=self.encoder(h,valid);mean=self.control(state)
        logits=(self.queue(queue)*self.query(state)[:,None]).sum(-1)/math.sqrt(state.shape[-1])
        logits=logits.masked_fill(~mask,-1e9)
        return mean,torch.cat([self.follow(state),logits],-1)
    def distributions(self,h,valid,q,mask):
        mean,logits=self(h,valid,q,mask)
        return torch.distributions.Normal(mean,self.log_std.clamp(-5,0).exp()),torch.distributions.Categorical(logits=logits)
    def sample(self,h,valid,q,mask):
        normal,cat=self.distributions(h,valid,q,mask);latent=normal.sample();choice=cat.sample()
        logp=self._logp(normal,latent)+cat.log_prob(choice)
        return torch.cat([latent.tanh(),choice[:,None]],-1),latent,choice,logp
    @staticmethod
    def _logp(normal,latent):
        correction=2*(math.log(2)-latent-F.softplus(-2*latent))
        return (normal.log_prob(latent)-correction).sum(-1)
    def evaluate(self,h,valid,q,mask,latent,choice):
        normal,cat=self.distributions(h,valid,q,mask)
        return self._logp(normal,latent)+cat.log_prob(choice),normal.entropy().sum(-1)+cat.entropy()
    def act(self,h,valid,q,mask):
        mean,logits=self(h,valid,q,mask)
        return torch.cat([mean.tanh(),logits.argmax(-1)[:,None]],-1)


class ReceiverActor(nn.Module):
    def __init__(self,config):
        super().__init__();self.encoder=LocalEncoder(len(RECEIVER_FEATURES),config)
        self.masked_choice=None
        self.head=nn.Linear(config.width,1+len(ACK_ACTIONS));nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():self.head.bias[0]=.5
    def mask_teacher_alias(self,batch,delay):
        self.masked_choice=next((i+1 for i,pair in enumerate(ACK_ACTIONS) if pair==(batch,delay)),None)
    def forward(self,h,valid):
        logits=self.head(self.encoder(h,valid))
        if self.masked_choice is not None:
            logits=logits.clone();logits[:,self.masked_choice]=-1e9
        return logits
    def sample(self,h,valid):
        d=torch.distributions.Categorical(logits=self(h,valid));a=d.sample();return a,d.log_prob(a)
    def evaluate(self,h,valid,a):
        d=torch.distributions.Categorical(logits=self(h,valid));return d.log_prob(a),d.entropy()
    def act(self,h,valid):return self(h,valid).argmax(-1)


class CentralCritic(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(len(SENDER_FEATURES)+len(RECEIVER_FEATURES)+6,128),
            nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
    def forward(self,state):return self.net(state).squeeze(-1)


def save_checkpoint(path,sender,receiver,critic,config,metadata,optimizers=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    torch.save(dict(format_version='joint-v1',config=asdict(config),
        sender_features=list(SENDER_FEATURES),receiver_features=list(RECEIVER_FEATURES),queue_features=list(QUEUE_FEATURES),
        ack_actions=list(ACK_ACTIONS),sender=sender.state_dict(),receiver=receiver.state_dict(),critic=critic.state_dict(),
        metadata=metadata,optimizers=[o.state_dict() for o in optimizers] if optimizers else None),temp)
    temp.replace(path)


def load_checkpoint(path,device='cpu'):
    d=torch.load(path,map_location='cpu',weights_only=True)
    if d.get('format_version') not in ('joint-v1','joint-actors-v1') or d.get('sender_features')!=list(SENDER_FEATURES) or d.get('receiver_features')!=list(RECEIVER_FEATURES) or d.get('queue_features')!=list(QUEUE_FEATURES) or d.get('ack_actions')!=list(ACK_ACTIONS):
        raise ValueError('incompatible joint checkpoint')
    config=JointModelConfig(**d['config']);sender=SenderActor(config).to(device);receiver=ReceiverActor(config).to(device)
    sender.load_state_dict(d['sender']);receiver.load_state_dict(d['receiver'])
    if d['metadata'].get('training',{}).get('mask_receiver_alias',False):
        teacher=d['metadata']['baseline'];receiver.mask_teacher_alias(teacher['batch'],teacher['delay'])
    return sender.eval(),receiver.eval(),config,d['metadata']


def export_actors(source,destination):
    """Export local policies and schemas, excluding critic, optimizers and RNG."""
    data=torch.load(source,map_location='cpu',weights_only=True)
    load_checkpoint(source)
    data['format_version']='joint-actors-v1'
    data.pop('critic',None);data.pop('optimizers',None)
    data['metadata']={k:v for k,v in data['metadata'].items() if k not in ('torch_rng','cuda_rng','episode_counters')}
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True);torch.save(data,destination)
