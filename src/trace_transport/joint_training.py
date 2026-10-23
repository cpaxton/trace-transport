"""Joint MAPPO: two local actors, shared return, separate centralized critic.

CPU packet workers run in parallel. CUDA batches actor inference and updates;
no router state enters an actor tensor. Session histories survive message ends.
"""
from dataclasses import asdict,dataclass
import json
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import torch

from .joint_curriculum import DOMAIN
from .joint_models import JointModelConfig,SenderActor,ReceiverActor,CentralCritic,save_checkpoint
from .joint_session import SENDER_FEATURES,RECEIVER_FEATURES
from .joint_workers import EnvironmentPool
from .learning import generalized_advantages


@dataclass(frozen=True)
class JointTrainConfig:
    decisions:int=5_000_000
    environments:int=16
    rollout:int=256
    epochs:int=4
    minibatch:int=1024
    seed:int=42
    device:str='cpu'
    learning_rate:float=1e-4
    checkpoint_every:int=250000
    production_seconds:float=20.
    drain_seconds:float=20.
    easy_fraction:float=.2
    learn_sender:bool=True
    learn_receiver:bool=True
    def __post_init__(self):
        if min(self.decisions,self.environments,self.rollout,self.epochs,self.minibatch,self.checkpoint_every)<1:
            raise ValueError('training budgets must be positive')
        if self.learning_rate<=0 or not 0<=self.easy_fraction<=1:raise ValueError('invalid optimizer/curriculum')


class Histories:
    def __init__(self,obs,cfg):
        self.sender=np.zeros((len(obs['sender']),cfg.history,len(SENDER_FEATURES)),np.float32)
        self.receiver=np.zeros((len(obs['receiver']),cfg.history,len(RECEIVER_FEATURES)),np.float32)
        self.valid=np.zeros((len(obs['sender']),cfg.history),bool)
    def append(self,obs):
        self.sender[:,:-1]=self.sender[:,1:].copy();self.sender[:,-1]=obs['sender']
        self.receiver[:,:-1]=self.receiver[:,1:].copy();self.receiver[:,-1]=obs['receiver']
        self.valid[:,:-1]=self.valid[:,1:].copy();self.valid[:,-1]=True


def _seed(root,episode):
    words=np.random.SeedSequence([0x4A4F494E,root,episode]).generate_state(2)
    return int(words[0])|(int(words[1])<<32)


def _tensor(array,device):return torch.as_tensor(array,device=device)


def _pack(observations,histories,role,device):
    rows=[];mapping=[]
    for e,(obs,h) in enumerate(zip(observations,histories)):
        ids=np.flatnonzero(obs[role+'_active'])
        for i in ids:mapping.append((e,int(i)))
        if len(ids):
            fields=[getattr(h,role)[ids].copy(),h.valid[ids].copy()]
            if role=='sender':fields.extend([obs['queue'][ids].copy(),obs['queue_mask'][ids].copy()])
            rows.append(fields)
    if not rows:return None,mapping
    data=[_tensor(np.concatenate([row[i] for row in rows]),device) for i in range(len(rows[0]))]
    return data,mapping


def _actor_update(model,optimizer,records,advantages,cfg,device,role):
    if not records:return dict(gradient_norm=0.,kl=0.,samples=0,steps=0)
    fields={key:torch.cat([row[key] for row in records]).to(device) for key in records[0]}
    adv=advantages.flatten()[fields.pop('world')]
    adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
    maximum=0.;max_kl=0.;steps=0;stop=False
    for _ in range(cfg.epochs):
        for ids in torch.randperm(len(adv),device=device).split(cfg.minibatch):
            if role=='sender':
                logp,entropy=model.evaluate(fields['h'][ids],fields['v'][ids],fields['q'][ids],fields['mask'][ids],fields['latent'][ids],fields['choice'][ids])
            else:logp,entropy=model.evaluate(fields['h'][ids],fields['v'][ids],fields['choice'][ids])
            difference=logp-fields['oldp'][ids];ratio=difference.exp()
            kl=((ratio-1)-difference).mean();max_kl=max(max_kl,float(kl.detach()))
            if kl.item()>.015:stop=True;break
            loss=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()-.001*entropy.mean()
            optimizer.zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5);maximum=max(maximum,float(norm))
            optimizer.step();steps+=1
        if stop:break
    return dict(gradient_norm=maximum,kl=max_kl,samples=len(adv),steps=steps)


def train(output,cfg=JointTrainConfig(),model_cfg=JointModelConfig(),resume=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if resume is None and (output/'config.json').exists():raise ValueError('training output already exists; use a new run directory')
    torch.set_num_threads(1);torch.manual_seed(cfg.seed)
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('requested CUDA is unavailable')
    sender=SenderActor(model_cfg).to(cfg.device);receiver=ReceiverActor(model_cfg).to(cfg.device);critic=CentralCritic().to(cfg.device)
    optimizers=[torch.optim.Adam(m.parameters(),lr=cfg.learning_rate) for m in (sender,receiver,critic)]
    obs=[];states=[];histories=[];episodes=list(range(cfg.environments));next_episode=cfg.environments
    decisions=0;world_steps=0;updates=0;last_checkpoint=0;start=time.monotonic();completed=[]
    metadata=dict(training=asdict(cfg),model=asdict(model_cfg),domain=DOMAIN,reward='joint-deadline-v1',baseline=dict(rate='fixed',scheduler='edf',feedback='budgeted'))
    if resume is not None:
        resume=Path(resume)
        data=torch.load(resume,map_location='cpu',weights_only=True)
        old=data['metadata']
        if data.get('format_version')!='joint-v1' or old['training']!=asdict(cfg) or old['model']!=asdict(model_cfg) or old['domain']!=DOMAIN or old['reward']!=metadata['reward']:
            raise ValueError('resume requires matching training, model, domain and reward configuration')
        sender.load_state_dict(data['sender']);receiver.load_state_dict(data['receiver']);critic.load_state_dict(data['critic'])
        for optimizer,state in zip(optimizers,data['optimizers']):optimizer.load_state_dict(state)
        decisions=old['decisions'];world_steps=old['world_steps'];updates=old['updates'];last_checkpoint=decisions
        if decisions>=cfg.decisions:raise ValueError('resume checkpoint has already reached training budget')
        next_episode=max(old['episode_counters'])+1
        episodes=list(range(next_episode,next_episode+cfg.environments));next_episode+=cfg.environments
        torch.set_rng_state(old['torch_rng'])
        if cfg.device.startswith('cuda'):torch.cuda.set_rng_state_all(old['cuda_rng'])
        metadata['resume_history']=old.get('resume_history',[])+([old['resume']] if 'resume' in old else [])
        metadata['resume']={'source':str(resume),'sha256':hashlib.sha256(resume.read_bytes()).hexdigest(),
            'decisions':decisions,'collector_reset':True,
            'description':'Optimizer and RNG continuation; in-flight worlds and local histories reset with fresh episode seeds.'}
        previous=output/'progress.json'
        if previous.exists():start-=json.loads(previous.read_text()).get('elapsed_seconds',0)
        # Interrupted episodes are deliberately excluded from completed-episode metrics.
        if (output/'episodes.json').exists():completed=json.loads((output/'episodes.json').read_text())
    (output/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    def checkpoint(name):
        save_checkpoint(output/name,sender,receiver,critic,model_cfg,
            {**metadata,'decisions':decisions,'world_steps':world_steps,'updates':updates,'episode_counters':episodes,
             'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if cfg.device.startswith('cuda') else []},optimizers)
    def reset(index,episode):
        stage='easy' if decisions<cfg.decisions*cfg.easy_fraction else 'full'
        o,state,_=pool.reset(index,_seed(cfg.seed,episode),cfg.production_seconds,cfg.drain_seconds,stage)
        return o,state,Histories(o,model_cfg)
    pool=EnvironmentPool(cfg.environments)
    try:
        for i in range(cfg.environments):
            o,state,h=reset(i,episodes[i]);obs.append(o);states.append(state);histories.append(h)
        if resume is None:checkpoint('initial.pt')
        while decisions<cfg.decisions:
            sender_records=[];receiver_records=[];rewards=[];values=[];terminals=[];critic_states=[]
            collection_start=time.monotonic();rollout_decisions=0;events=0
            for step in range(cfg.rollout):
                for o,h in zip(obs,histories):h.append(o)
                sp,sm=_pack(obs,histories,'sender',cfg.device);rp,rm=_pack(obs,histories,'receiver',cfg.device)
                actions=[(np.zeros((len(o['sender']),3),np.float32),np.zeros(len(o['receiver']),np.int64)) for o in obs]
                with torch.no_grad():
                    state_tensor=_tensor(np.asarray(states),cfg.device);value=critic(state_tensor)
                    if sp is not None and cfg.learn_sender:
                        action,latent,choice,logp=sender.sample(*sp)
                        row=dict(h=sp[0].cpu(),v=sp[1].cpu(),q=sp[2].cpu(),mask=sp[3].cpu(),
                            latent=latent.cpu(),choice=choice.cpu(),oldp=logp.cpu(),
                            world=torch.tensor([step*cfg.environments+e for e,_ in sm]))
                        sender_records.append(row)
                        for (e,i),a in zip(sm,action.cpu().numpy()):actions[e][0][i]=a
                    if rp is not None and cfg.learn_receiver:
                        choice,logp=receiver.sample(*rp)
                        receiver_records.append(dict(h=rp[0].cpu(),v=rp[1].cpu(),choice=choice.cpu(),oldp=logp.cpu(),
                            world=torch.tensor([step*cfg.environments+e for e,_ in rm])))
                        for (e,i),a in zip(rm,choice.cpu().numpy()):actions[e][1][i]=a
                critic_states.append(state_tensor.cpu());values.append(value.cpu())
                pool.send_steps(actions);result=pool.collect()
                rewards.append(torch.tensor([r[2] for r in result],dtype=torch.float32));terminals.append(torch.tensor([r[3] for r in result]))
                n=len(sm)+len(rm);decisions+=n;rollout_decisions+=n;world_steps+=cfg.environments
                for i,(o,state,_,done,metrics,event_count) in enumerate(result):
                    events=max(events,event_count)
                    if done:
                        completed.append(metrics);episodes[i]=next_episode;next_episode+=1
                        obs[i],states[i],histories[i]=reset(i,episodes[i])
                    else:obs[i],states[i]=o,state
                if decisions>=cfg.decisions:break
            with torch.no_grad():bootstrap=critic(_tensor(np.asarray(states),cfg.device)).cpu()
            advantages,returns=generalized_advantages(torch.stack(rewards),torch.stack(values),torch.stack(terminals),bootstrap,gamma=.999)
            collection_seconds=time.monotonic()-collection_start
            s_stats=_actor_update(sender,optimizers[0],sender_records,advantages.to(cfg.device),cfg,cfg.device,'sender')
            r_stats=_actor_update(receiver,optimizers[1],receiver_records,advantages.to(cfg.device),cfg,cfg.device,'receiver')
            cs=torch.cat(critic_states).to(cfg.device);target=returns.flatten().to(cfg.device)
            for _ in range(cfg.epochs):
                for ids in torch.randperm(len(cs),device=cfg.device).split(cfg.minibatch):
                    loss=.5*(critic(cs[ids])-target[ids]).square().mean()
                    optimizers[2].zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),.5);optimizers[2].step()
            updates+=1
            elapsed=time.monotonic()-start
            log=dict(phase="training",pid=os.getpid(),update=updates,decisions=decisions,world_steps=world_steps,elapsed_seconds=elapsed,
                decisions_per_second=decisions/max(elapsed,1e-9),collection_seconds=collection_seconds,
                rollout_decisions=rollout_decisions,sender=s_stats,receiver=r_stats,
                mean_team_reward=float(torch.stack(rewards).mean()),completed_episodes=len(completed),
                recent_miss_rate=float(np.mean([x['interactive_miss_rate'] for x in completed[-16:]])) if completed else None)
            with (output/'training.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
            temp=output/'progress.tmp';temp.write_text(json.dumps(log,indent=2)+'\n');temp.replace(output/'progress.json')
            print(json.dumps(log),flush=True)
            if decisions-last_checkpoint>=cfg.checkpoint_every:
                checkpoint(f'd{decisions:09d}.pt');last_checkpoint=decisions
            checkpoint('latest.pt')
        checkpoint('final.pt')
        log.update(phase='complete')
        temp=output/'progress.tmp';temp.write_text(json.dumps(log,indent=2)+'\n');temp.replace(output/'progress.json')
        (output/'episodes.json').write_text(json.dumps(completed,indent=2)+'\n')
        return output/'final.pt'
    except BaseException as exc:
        failure=dict(phase='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
            pid=os.getpid(),decisions=decisions,world_steps=world_steps,error=repr(exc),
            checkpoint=str(output/'latest.pt'))
        temp=output/'failure.tmp';temp.write_text(json.dumps(failure,indent=2)+'\n');temp.replace(output/'failure.json')
        raise
    finally:pool.close()
