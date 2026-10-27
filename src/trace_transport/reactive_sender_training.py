"""Small producer-only macro MAPPO pilot; receiver and reverse roles are fixed."""
from dataclasses import asdict,dataclass
from pathlib import Path
import json
import math
import os
import time
import numpy as np
import torch
from torch import nn
from .experiment_io import write
from .joint_session import SENDER_FEATURES,RECEIVER_FEATURES
from .learning import time_generalized_advantages
from .reactive_sender import BIASES,CONTRACT,FEATURES,KEEP,MacroPool,CRITIC_EXTRA


@dataclass(frozen=True)
class Config:
    decisions:int=2000
    environments:int=4
    rollout:int=16
    epochs:int=3
    minibatch:int=256
    learning_rate:float=.0001
    seed:int=42
    kind:str='current-mlp'
    def __post_init__(self):
        if min(self.decisions,self.environments,self.rollout,self.epochs,self.minibatch)<1 or self.environments>4:raise ValueError('invalid bounded training budget')
        if self.kind not in ('current-mlp','history-mlp') or not math.isfinite(self.learning_rate) or self.learning_rate<=0:raise ValueError('invalid policy configuration')


class Policy(nn.Module):
    def __init__(self,kind='current-mlp'):
        super().__init__();self.kind=kind
        if kind=='current-mlp':self.encoder=nn.Sequential(nn.Linear(len(FEATURES),64),nn.Tanh(),nn.Linear(64,64),nn.Tanh());width=64
        elif kind=='history-mlp':self.encoder=nn.Sequential(nn.Linear(CONTRACT['bins']*(len(FEATURES)+1),16),nn.Tanh(),nn.Linear(16,64),nn.Tanh());width=64
        else:raise ValueError('unknown policy')
        self.head=nn.Linear(width,len(BIASES));nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        with torch.no_grad():self.head.bias[1]=math.log((len(BIASES)-1)*CONTRACT['initial_teacher_prior']/(1-CONTRACT['initial_teacher_prior']))
    def forward(self,h,valid):
        h=h.masked_fill(~valid[:,:,None],0)
        x=h[:,-1] if self.kind=='current-mlp' else torch.cat([h,valid[:,:,None].float()],-1).flatten(1)
        return self.head(self.encoder(x))


class Critic(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(len(SENDER_FEATURES)+len(RECEIVER_FEATURES)+6+len(CRITIC_EXTRA),128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def save(path,policy,critic,optimizers,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    torch.save(dict(format_version='reactive-sender-v7',contract=CONTRACT,features=list(FEATURES),
        kind=policy.kind,policy=policy.state_dict(),critic=critic.state_dict(),
        optimizers=[o.state_dict() for o in optimizers],torch_rng=torch.get_rng_state(),metadata=json.loads(json.dumps(metadata))),temp)
    temp.replace(path)


def load(path):
    d=torch.load(path,map_location='cpu',weights_only=True)
    if d.get('format_version')!='reactive-sender-v7' or d.get('contract')!=CONTRACT or d.get('features')!=list(FEATURES):raise ValueError('incompatible reactive sender checkpoint')
    policy=Policy(d['kind']);policy.load_state_dict(d['policy']);policy.eval()
    if not all(torch.isfinite(p).all() for p in policy.parameters()):raise ValueError('nonfinite policy')
    return policy,d['metadata']


def train(output,config,seeds,source,*,on_update=None):
    output=Path(output)
    if (output/'config.json').exists():raise ValueError('fresh output required; no silent resume')
    if not seeds or len(seeds)!=len(set(seeds)):raise ValueError('nonempty unique training seeds required')
    torch.set_num_threads(1);torch.manual_seed(config.seed);rng=np.random.default_rng(config.seed)
    policy=Policy(config.kind)
    torch.manual_seed(config.seed+10000);critic=Critic()
    torch.manual_seed(config.seed+20000)
    optimizers=[torch.optim.Adam(policy.parameters(),lr=config.learning_rate),torch.optim.Adam(critic.parameters(),lr=config.learning_rate)]
    metadata=dict(training=asdict(config),contract=CONTRACT,training_seeds=seeds,source=source,
        initialization=dict(policy_seed=config.seed,critic_seed=config.seed+10000,sampling_seed=config.seed+20000),
        reward='raw goodput-v6, micro rewards discounted in physical time within macro transitions; no shaping/subtracted teacher',
        runtime=dict(torch=str(torch.__version__),numpy=str(np.__version__)),accepted_win=False)
    write(output/'config.json',metadata);save(output/'initial.pt',policy,critic,optimizers,metadata)
    start=time.monotonic();decisions=world_steps=kernel_steps=updates=0;episodes=[];visited=set();inference=[]
    pool=MacroPool(config.environments);obs=[]
    def reset(i):
        seed=int(rng.choice(seeds));visited.add(seed);return pool.reset(i,seed)
    try:
        obs=[reset(i) for i in range(config.environments)]
        while decisions<config.decisions:
            records=[];rewards=[];values=[];terminals=[];elapsed=[];states=[];raw=[]
            for step in range(config.rollout):
                state=torch.tensor(np.stack([o['critic'] for o in obs]));states.append(state)
                mapping=[(e,int(i)) for e,o in enumerate(obs) for i in np.flatnonzero(o['eligible'])]
                actions=[np.full(len(o['eligible']),KEEP,int) for o in obs]
                with torch.no_grad():
                    values.append(critic(state))
                    if mapping:
                        h=torch.tensor(np.stack([obs[e]['history'][i] for e,i in mapping]));valid=torch.tensor(np.stack([obs[e]['valid'][i] for e,i in mapping]))
                        begin=time.perf_counter();distribution=torch.distributions.Categorical(logits=policy(h,valid));choice=distribution.sample();oldp=distribution.log_prob(choice)
                        inference.append((time.perf_counter()-begin)*1000)
                        records.append(dict(h=h,valid=valid,choice=choice,oldp=oldp,world=torch.tensor([step*config.environments+e for e,_ in mapping])))
                        for (e,i),a in zip(mapping,choice.tolist()):actions[e][i]=a
                for pipe,a in zip(pool.pipes,actions):pipe.send(('step',a))
                results=[pool.receive(i) for i in range(config.environments)]
                rewards.append(torch.tensor([r[1] for r in results]));terminals.append(torch.tensor([r[2] for r in results]));elapsed.append(torch.tensor([r[4] for r in results]))
                raw.extend(r[6] for r in results);decisions+=len(mapping);world_steps+=config.environments;kernel_steps+=sum(r[5] for r in results)
                for i,result in enumerate(results):
                    if result[2]:episodes.append(result[3]);obs[i]=reset(i)
                    else:obs[i]=result[0]
                if decisions>=config.decisions:break
            with torch.no_grad():bootstrap=critic(torch.tensor(np.stack([o['critic'] for o in obs])))
            adv,target=time_generalized_advantages(torch.stack(rewards),torch.stack(values),torch.stack(terminals),bootstrap,torch.stack(elapsed),gamma=.999,lam=.98)
            actor_stats=dict(samples=0,gradient_norm=0.,kl=0.)
            if records:
                fields={k:torch.cat([r[k] for r in records]) for k in records[0]};advantages=adv.flatten()[fields.pop('world')]
                advantages=(advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-6)
                stop=False
                for _ in range(config.epochs):
                    for ids in torch.randperm(len(advantages)).split(config.minibatch):
                        d=torch.distributions.Categorical(logits=policy(fields['h'][ids],fields['valid'][ids]));logp=d.log_prob(fields['choice'][ids])
                        ratio=(logp-fields['oldp'][ids]).exp();kl=((ratio-1)-(logp-fields['oldp'][ids])).mean()
                        actor_stats['kl']=max(actor_stats['kl'],float(kl.detach()))
                        if kl>.02:stop=True;break
                        loss=-torch.minimum(ratio*advantages[ids],ratio.clamp(.8,1.2)*advantages[ids]).mean()-.001*d.entropy().mean()
                        optimizers[0].zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizers[0].step()
                        actor_stats['gradient_norm']=max(actor_stats['gradient_norm'],float(norm));actor_stats['samples']=len(advantages)
                    if stop:break
                actor_stats['choices']=torch.bincount(fields['choice'],minlength=len(BIASES)).tolist()
            cs=torch.cat(states);targets=target.flatten()
            for _ in range(config.epochs):
                for ids in torch.randperm(len(cs)).split(config.minibatch):
                    loss=.5*(critic(cs[ids])-targets[ids]).square().mean();optimizers[1].zero_grad();loss.backward();nn.utils.clip_grad_norm_(critic.parameters(),.5);optimizers[1].step()
            updates+=1;duration=time.monotonic()-start
            progress=dict(phase='training',pid=os.getpid(),macro_decisions=decisions,world_macro_steps=world_steps,kernel_steps=kernel_steps,
                updates=updates,elapsed_seconds=duration,macro_decisions_per_second=decisions/max(duration,1e-9),completed_episodes=len(episodes),
                distinct_training_worlds=len(visited),mean_raw_macro_reward=float(np.mean(raw)),actor=actor_stats,
                actor_batch_ms_p50_p95=np.quantile(inference[-100:],[.5,.95]).tolist() if inference else None,
                note='training reward, not validation performance; all actor samples are eligible bulk producers')
            write(output/'progress.json',progress);write(output/'episodes.json',episodes)
            with (output/'training.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
            save(output/'latest.pt',policy,critic,optimizers,{**metadata,**progress,**{'visited_training_worlds':sorted(visited)}})
            if on_update:on_update(progress)
            print(json.dumps(progress),flush=True)
        save(output/'final.pt',policy,critic,optimizers,{**metadata,**progress,**{'visited_training_worlds':sorted(visited)}})
        write(output/'progress.json',{**progress,'phase':'complete'});return policy
    finally:pool.close()
