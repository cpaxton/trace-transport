"""Joint MAPPO: two local actors, shared return, separate centralized critic.

CPU packet workers run in parallel. CUDA batches actor inference and updates;
no router state enters an actor tensor. Session histories survive message ends.
"""
from dataclasses import asdict,dataclass
import json
import hashlib
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .joint_curriculum import DOMAIN
from .joint_models import JointModelConfig,SenderActor,ReceiverActor,CentralCritic,save_checkpoint
from .joint_session import SENDER_FEATURES,RECEIVER_FEATURES,ACK_ACTIONS
from .joint_workers import EnvironmentPool
from .learning import generalized_advantages,time_generalized_advantages


@dataclass(frozen=True)
class JointTrainConfig:
    profile:str="deadline-v1"
    teacher_scheduler:str="edf"
    teacher_rate:str="fixed"
    teacher_initial_rate:float=128000.
    teacher_batch:int=8
    teacher_delay:float=.025
    paired_teacher:bool=False
    receiver_teacher_prior:float|None=None
    teacher_prior:float=.9
    mask_receiver_alias:bool=False
    progress_shaping:bool=False
    time_aware_gae:bool=False
    gae_lambda:float=.95
    gpu_memory_fraction:float=.15
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
        if self.teacher_batch not in (1,2,4,8,16,32) or not 0<=self.teacher_delay<=.25:raise ValueError('invalid teacher feedback')
        if not 2000<=self.teacher_initial_rate<=2000000:raise ValueError('invalid initial pacing rate')
        if self.profile not in ('deadline-v1','goodput-v2','goodput-v3','goodput-v4','goodput-v5','goodput-v6') or self.teacher_rate not in ('fixed','aimd','bdp','delay','delay-fast'):raise ValueError('invalid profile/teacher')
        if self.mask_receiver_alias and self.profile not in ('goodput-v5','goodput-v6'):raise ValueError('receiver alias masking is versioned for v5')
        if self.progress_shaping and (self.profile not in ('goodput-v5','goodput-v6') or not self.time_aware_gae or self.paired_teacher):raise ValueError('progress shaping requires v5, time-aware GAE and raw task reward')
        if self.time_aware_gae and self.profile not in ('goodput-v4','goodput-v5','goodput-v6'):raise ValueError('elapsed-time GAE requires v4 transition timing')
        if self.teacher_scheduler not in ('edf','drr'):raise ValueError('invalid teacher scheduler')
        if self.receiver_teacher_prior is not None and not 0<self.receiver_teacher_prior<1:raise ValueError('invalid receiver prior')
        if self.teacher_rate=='delay-fast' and self.profile not in ('goodput-v3','goodput-v4','goodput-v5','goodput-v6'):raise ValueError('delay-fast requires v3')
        if not 0<self.teacher_prior<1 or not 0<self.gpu_memory_fraction<=.5 or not 0<=self.gae_lambda<=1:raise ValueError('invalid memory/GAE limits')
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


def receiver_choice_diagnostics(choices,advantages):
    count=1+len(ACK_ACTIONS)
    if ((choices<0)|(choices>=count)).any():raise ValueError('receiver choice outside current action table')
    return dict(choice_counts=torch.bincount(choices,minlength=count).tolist(),
        choice_mean_normalized_advantage=[float(advantages[choices==i].mean()) if (choices==i).any() else None for i in range(count)])


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
    diagnostic=receiver_choice_diagnostics(fields['choice'].detach().cpu(),adv.detach().cpu()) if role=='receiver' else {}
    return dict(gradient_norm=maximum,kl=max_kl,samples=len(adv),steps=steps,**diagnostic)


def train(output,cfg=JointTrainConfig(),model_cfg=JointModelConfig(),resume=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if resume is None and (output/'config.json').exists():raise ValueError('training output already exists; use a new run directory')
    torch.set_num_threads(1);torch.manual_seed(cfg.seed)
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('requested CUDA is unavailable')
    if cfg.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(cfg.gpu_memory_fraction)
    if cfg.profile=='goodput-v6':
        from .goodput_reactive import DOMAIN as domain
    elif cfg.profile=='goodput-v5':
        from .goodput_progress import DOMAIN as domain
    elif cfg.profile=='goodput-v4':
        from .goodput_deployment import DOMAIN as domain
    elif cfg.profile=='goodput-v3':
        from .goodput_distribution import DOMAIN as domain
    elif cfg.profile=='goodput-v2':
        from .goodput import DOMAIN as domain
    else:domain=DOMAIN
    baseline=dict(rate=cfg.teacher_rate,scheduler=cfg.teacher_scheduler,feedback='budgeted')
    if cfg.profile.startswith('goodput-'):baseline.update(initial_rate=cfg.teacher_initial_rate,batch=cfg.teacher_batch,delay=cfg.teacher_delay)
    if cfg.profile in ('goodput-v5','goodput-v6'):baseline['progress_shaping']=cfg.progress_shaping
    sender=SenderActor(model_cfg).to(cfg.device);receiver=ReceiverActor(model_cfg).to(cfg.device);critic=CentralCritic().to(cfg.device)
    if cfg.mask_receiver_alias:receiver.mask_teacher_alias(cfg.teacher_batch,cfg.teacher_delay)
    if cfg.profile.startswith('goodput-'):
        # Greedy teacher equivalence is insufficient: sampling must start close
        # to it too, rather than uniformly random scheduling/ACK decisions.
        from .joint_session import MAX_CANDIDATES,ACK_ACTIONS
        odds=cfg.teacher_prior/(1-cfg.teacher_prior)
        with torch.no_grad():
            sender.follow.bias.fill_(math.log(MAX_CANDIDATES*odds))
            rx_prior=cfg.receiver_teacher_prior if cfg.receiver_teacher_prior is not None else cfg.teacher_prior
            receiver.head.bias[0]=math.log((len(ACK_ACTIONS)-int(receiver.masked_choice is not None))*rx_prior/(1-rx_prior))
            sender.log_std.fill_(-3.)
    optimizers=[torch.optim.Adam(m.parameters(),lr=cfg.learning_rate) for m in (sender,receiver,critic)]
    obs=[];states=[];histories=[];episodes=list(range(cfg.environments));next_episode=cfg.environments
    decisions=0;world_steps=0;updates=0;last_checkpoint=0;start=time.monotonic();completed=[]
    source_hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['goodput.py','goodput_distribution.py','goodput_deployment.py','goodput_progress.py','goodput_reactive.py','learning_diagnostics.py','joint_session.py','joint_models.py','joint_workers.py','joint_training.py','protocol.py','wire.py','session_receiver.py','learning.py']} if cfg.profile.startswith('goodput-') else {}
    metadata=dict(algorithm='mappo-time-gae-progress-v1' if cfg.progress_shaping else ('mappo-time-gae-v1' if cfg.time_aware_gae else 'mappo-step-gae-v1'),teacher_baseline_coupling=('same initial RNG; action-dependent draws can diverge; no guaranteed variance reduction' if cfg.paired_teacher else 'disabled; raw task reward'),training=asdict(cfg),model=asdict(model_cfg),domain=domain,profile=cfg.profile,
        reward=domain['reward'] if cfg.profile.startswith('goodput-') else 'joint-deadline-v1',baseline=baseline,source_sha256=source_hashes)
    if resume is not None:
        resume=Path(resume)
        data=torch.load(resume,map_location='cpu',weights_only=True)
        old=data['metadata']
        if data.get('format_version')!='joint-v1' or JointTrainConfig(**old['training'])!=cfg or old['model']!=asdict(model_cfg) or old['domain']!=domain or old['reward']!=metadata['reward']:
            raise ValueError('resume requires matching training, model, domain and reward configuration')
        if cfg.profile.startswith('goodput-') and old.get('source_sha256')!=source_hashes:raise ValueError('goodput resume source changed; version the next experiment')
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
        o,state,_=pool.reset(index,_seed(cfg.seed,episode),cfg.production_seconds,cfg.drain_seconds,stage,False,baseline,None,cfg.profile,cfg.paired_teacher)
        return o,state,Histories(o,model_cfg)
    pool=EnvironmentPool(cfg.environments)
    try:
        for i in range(cfg.environments):
            o,state,h=reset(i,episodes[i]);obs.append(o);states.append(state);histories.append(h)
        if resume is None:checkpoint('initial.pt')
        while decisions<cfg.decisions:
            sender_records=[];receiver_records=[];rewards=[];values=[];terminals=[];critic_states=[];elapsed_transitions=[]
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
                if cfg.time_aware_gae:elapsed_transitions.append(torch.tensor([r[0]['transition_seconds'] for r in result],dtype=torch.float32))
                n=len(sm)+len(rm);decisions+=n;rollout_decisions+=n;world_steps+=cfg.environments
                for i,(o,state,_,done,metrics,event_count) in enumerate(result):
                    events=max(events,event_count)
                    if done:
                        completed.append(metrics);episodes[i]=next_episode;next_episode+=1
                        obs[i],states[i],histories[i]=reset(i,episodes[i])
                    else:obs[i],states[i]=o,state
                if decisions>=cfg.decisions:break
            with torch.no_grad():bootstrap=critic(_tensor(np.asarray(states),cfg.device)).cpu()
            if cfg.time_aware_gae:
                advantages,returns=time_generalized_advantages(torch.stack(rewards),torch.stack(values),torch.stack(terminals),bootstrap,torch.stack(elapsed_transitions),gamma=.999,lam=cfg.gae_lambda)
            else:
                advantages,returns=generalized_advantages(torch.stack(rewards),torch.stack(values),torch.stack(terminals),bootstrap,gamma=.999,lam=cfg.gae_lambda)
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
