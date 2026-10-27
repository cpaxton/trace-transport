"""Frozen-policy deployment and conditional action-value diagnostics for v7.

No training, optimizer changes, model selection or acceptance claim. Actor inputs
remain local. Critics/future rollout outcomes are analysis-only data.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import random
import numpy as np
import torch
from trace_transport.experiment_io import write
from trace_transport.goodput_evaluation import summarize,compare
from trace_transport.learning import time_generalized_advantages
from trace_transport.reactive_sender import BIASES,KEEP,MacroWorld
from trace_transport.reactive_sender_training import Policy,Critic,load

GAMMA=.999
REFERENCE_SECONDS=.02
LAM=.98
_CACHE={}


def seed_value(*parts):
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1,dtype=np.uint64)[0])


def generators(seed,case,count):
    return [torch.Generator().manual_seed(seed_value(0x5637,seed,case,i)) for i in range(count)]


def model(path,*,need_critic=False,expected_core=None):
    actor,metadata=load(path);critic=None
    if expected_core is not None:
        recorded={k:v for k,v in metadata.get('source',{}).items() if k.startswith('src/')}
        if recorded!=expected_core:raise ValueError('checkpoint/simulator source mismatch')
    if need_critic:
        d=torch.load(path,map_location='cpu',weights_only=True)
        if 'critic' not in d:raise ValueError('credit analysis requires the full private checkpoint, not an actor-only export')
        critic=Critic();critic.load_state_dict(d['critic']);critic.eval()
        if not all(torch.isfinite(p).all() for p in critic.parameters()):raise ValueError('nonfinite critic')
    if not all(torch.isfinite(p).all() for p in actor.parameters()):raise ValueError('nonfinite actor')
    return actor,critic


def actions(actor,obs,mode,rng):
    if mode not in ('teacher','greedy','sampled'):raise ValueError('unknown deployment mode')
    result=np.full(len(obs['eligible']),KEEP,int);ids=np.flatnonzero(obs['eligible'])
    if mode=='teacher' or not len(ids):return result,None
    if actor is None:raise ValueError('actor required')
    with torch.inference_mode():
        logits=actor(torch.as_tensor(obs['history']),torch.as_tensor(obs['valid']));probabilities=logits.softmax(-1)
        greedy=logits.argmax(-1)
        for i in ids:
            result[i]=int(greedy[i]) if mode=='greedy' else int(torch.multinomial(probabilities[i],1,generator=rng[i]))
    return result,probabilities[ids].numpy()


def run_episode(seed,actor,mode,sampling_seed,*,production=12.,drain=3.):
    world=MacroWorld(seed,production=production,drain=drain);rng=generators(sampling_seed,seed,world.env.count)
    counts=Counter();probability_sum=np.zeros(len(BIASES));entropy=0.;eligible=0;changes=0;steps=0
    while not world.env.done():
        obs=world.observation();a,p=actions(actor,obs,mode,rng)
        counts.update(a[a!=KEEP].tolist())
        if p is not None:
            probability_sum+=p.sum(0);entropy+=float(-(p*np.log(np.maximum(p,1e-12))).sum());eligible+=len(p)
        before=world.env.target_bias.copy();world.step(a);changes+=int(np.count_nonzero(before!=world.env.target_bias));steps+=1
    row=world.env.metrics();row.pop('messages_detail',None);row.update(seed=seed,domain=world.label)
    return dict(record=row,eligible_choices=dict(counts),eligible_decisions=eligible,
        mean_probabilities=(probability_sum/eligible).tolist() if eligible else None,
        mean_entropy=entropy/eligible if eligible else None,actual_bias_changes=changes,macro_steps=steps,
        scope='choices count scheduled eligible producer decisions; changes measure actual bias at macro boundaries')


def gae_first(rewards,values,elapsed,horizon,*,lam=LAM):
    n=min(horizon,len(rewards))
    if n<1 or len(values)!=len(rewards)+1:raise ValueError('complete reward/value trace required')
    terminal=torch.zeros(n,1,dtype=torch.bool);terminal[-1,0]=n==len(rewards)
    # Match the parent's float32 estimator, including truncated critic bootstrap.
    advantage,_=time_generalized_advantages(torch.tensor(rewards[:n],dtype=torch.float32)[:,None],
        torch.tensor(values[:n],dtype=torch.float32)[:,None],terminal,
        torch.tensor([values[n]],dtype=torch.float32),torch.tensor(elapsed[:n],dtype=torch.float32)[:,None],
        gamma=GAMMA,lam=lam,reference_seconds=REFERENCE_SECONDS)
    return float(advantage[0,0])


def conditional_rollout(snapshot,actor,critic,index,choice,replica_seed,*,horizons=(4,16),include_trace=False):
    obs=snapshot.observation()
    if not 0<=index<len(obs['eligible']) or not obs['eligible'][index]:raise ValueError('forced decision must be an eligible producer')
    trial=snapshot.clone();rng=generators(replica_seed,trial.label['seed'],trial.env.count)
    # Future packet and action streams are independent across arms/replicas.
    # Already-scheduled arrivals remain conditional on the shared prefix.
    trial.env.streams=[random.Random(seed_value(0x4E4554,replica_seed,d)) for d in range(2)]
    start=trial.env.now;discount=1.;total=0.;values=[];rewards=[];elapsed=[];first=True
    before=dict(verified_bulk_bytes=sum(trial.env.verified[c] for c in ['image','map']),misses=trial.env.reward_counts['missed'],wire=trial.env.reward_counts['wire'])
    while not trial.env.done():
        o=trial.observation()
        with torch.inference_mode():values.append(float(critic(torch.as_tensor(o['critic'])[None])[0]))
        a,_=actions(actor,o,'sampled',rng)
        if first:a[index]=choice;first=False
        result=trial.step(a);rewards.append(result[1]);elapsed.append(result[4])
        total+=discount*result[1];discount*=GAMMA**(result[4]/REFERENCE_SECONDS)
    values.append(0.) # True terminal, not a time-limit truncation bootstrap.
    out=dict(discounted_return=total,critic_value=values[0],mc_advantage=total-values[0],
        gae_advantages={str(h):gae_first(rewards,values,elapsed,h) for h in horizons},
        verified_bulk_delta=sum(trial.env.verified[c] for c in ['image','map'])-before['verified_bulk_bytes'],
        missed_delta=trial.env.reward_counts['missed']-before['misses'],wire_delta=trial.env.reward_counts['wire']-before['wire'],
        continuation_seconds=trial.env.now-start,macro_steps=len(rewards),replica_seed=replica_seed)
    if include_trace:out['trace']=dict(rewards=rewards,values=values,elapsed=elapsed)
    return out


def difference_ci(x,y,seed):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<2 or len(y)<2 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('at least two finite replicas per arm required')
    rng=np.random.default_rng(seed)
    differences=x[rng.integers(0,len(x),(2000,len(x)))].mean(1)-y[rng.integers(0,len(y),(2000,len(y)))].mean(1)
    return np.quantile(differences,[.025,.975]).tolist()


def logit_gradient(probabilities,action_values):
    p=np.asarray(probabilities,float);q=np.asarray(action_values,float)
    if not np.isfinite(p).all() or not np.isfinite(q).all() or p.ndim!=1 or p.shape!=q.shape or not np.isclose(p.sum(),1.) or (p<0).any():raise ValueError('invalid conditional policy probabilities')
    return (p*(q-np.dot(p,q))).tolist()


def credit_case(seed,actor,critic,replicas,*,prefix_seed=701,after=2.,deadline=6.,index=0):
    world=MacroWorld(seed);rng=generators(prefix_seed,seed,world.env.count)
    while world.env.now<deadline and not world.env.done():
        o=world.observation()
        if world.env.now+1e-9>=after and o['eligible'][index]:break
        a,_=actions(actor,o,'sampled',rng);world.step(a)
    else:return dict(seed=seed,status='skipped',reason='no eligible requested producer before snapshot deadline')
    o=world.observation()
    with torch.inference_mode():p=actor(torch.as_tensor(o['history']),torch.as_tensor(o['valid'])).softmax(-1)[index].tolist()
    arms=[]
    for choice in range(len(BIASES)):
        rows=[conditional_rollout(world,actor,critic,index,choice,seed_value(prefix_seed,seed,index,choice,r,0x4352)) for r in range(replicas)]
        arms.append(dict(choice=choice,bias=BIASES[choice],trials=rows,
            mean_return=float(np.mean([r['discounted_return'] for r in rows])),return_std=float(np.std([r['discounted_return'] for r in rows],ddof=1)),
            mean_mc_advantage=float(np.mean([r['mc_advantage'] for r in rows])),
            mean_gae_advantages={str(h):float(np.mean([r['gae_advantages'][str(h)] for r in rows])) for h in [4,16]}))
    base=arms[BIASES.index(1.)]
    for arm in arms:
        arm['return_difference_from_teacher']=arm['mean_return']-base['mean_return']
        arm['return_difference_ci95']=[0.,0.] if arm is base else difference_ci(
            [r['discounted_return'] for r in arm['trials']],[r['discounted_return'] for r in base['trials']],seed_value(seed,arm['choice'],0x4349))
    expected=float(np.dot(p,[a['mean_return'] for a in arms]))
    return dict(seed=seed,status='complete',producer_index=index,snapshot_seconds=world.env.now,prefix_mode='sampled',prefix_seed=prefix_seed,
        probabilities=p,arms=arms,conditional_expected_mc_value=expected,critic_value=arms[0]['trials'][0]['critic_value'],
        mc_logit_gradient=logit_gradient(p,[a['mean_return'] for a in arms]),
        gae_logit_gradients={str(h):logit_gradient(p,[a['mean_gae_advantages'][str(h)] for a in arms]) for h in [4,16]},
        scope='conditional on-policy snapshot; independent future arms, not CRN; summary-critic aliasing, finite replicas and rollout-window placement limit attribution; per-agent logit contribution, not full shared-parameter PPO gradient')


def worker(task):
    torch.set_num_threads(1);path=task.get('checkpoint');need=task['type']=='credit'
    key=(path,need)
    if key not in _CACHE:
        if path:_CACHE[key]=model(path,need_critic=need)
        else:_CACHE[key]=(Policy().eval(),None) # Exact initialized probabilities; encoder irrelevant with zero head weights.
    actor,critic=_CACHE[key]
    if need:return credit_case(task['seed'],actor,critic,task['replicas'],prefix_seed=task['sampling_seed'])
    return run_episode(task['seed'],actor,task['mode'],task['sampling_seed'])


def validate_config(config,profile):
    spec=config['profiles'][profile]
    if not 1<=config['workers']<=4:raise ValueError('workers must be between one and four')
    if spec['credit_replicas']<2:raise ValueError('at least two credit replicas required')
    for key in ['performance_cases','sampling_seeds','performance_models','credit_cases','credit_models']:
        values=spec[key]
        if not values or len(values)!=len(set(values)):raise ValueError(f'empty or duplicate {key}')
    allowed={f'seed{seed}-{kind}' for seed in [42,43,44] for kind in ['current-mlp','history-mlp']}
    if not set(spec['performance_models']+spec['credit_models'])<=allowed:raise ValueError('unknown frozen model')
    for key in ['performance_cases','credit_cases','sampling_seeds']:
        if any(type(x) is not int or x<0 for x in spec[key]):raise ValueError(f'invalid {key}')


def build_tasks(config,root,profile):
    validate_config(config,profile)
    spec=config['profiles'][profile];cases=spec['performance_cases'];sampling=spec['sampling_seeds'];models=spec['performance_models']
    tasks=[dict(type='performance',tag='teacher',checkpoint=None,mode='teacher',seed=s,sampling_seed=0) for s in cases]
    for r in sampling:
        tasks.extend(dict(type='performance',tag=f'initial-sampled-{r}',checkpoint=None,mode='sampled',seed=s,sampling_seed=r) for s in cases)
    for name in models:
        path=str(root/name/'final.pt')
        tasks.extend(dict(type='performance',tag=f'{name}-greedy',checkpoint=path,mode='greedy',seed=s,sampling_seed=0) for s in cases)
        for r in sampling:tasks.extend(dict(type='performance',tag=f'{name}-sampled-{r}',checkpoint=path,mode='sampled',seed=s,sampling_seed=r) for s in cases)
    for name in spec['credit_models']:
        tasks.extend(dict(type='credit',tag=name,checkpoint=str(root/name/'final.pt'),seed=s,sampling_seed=config['prefix_sampling_seed'],replicas=spec['credit_replicas']) for s in spec['credit_cases'])
    return tasks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--models-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=['smoke','full'],default='smoke');args=p.parse_args();out=args.output
    if out.exists() and any(f.name!='service.log' for f in out.iterdir()):raise ValueError('new output directory required')
    config=json.loads(args.config.read_text());spec=config['profiles'][args.profile]
    validate_config(config,args.profile)
    torch.set_num_threads(1);source=json.loads(Path('source-manifest.json').read_text());core={k:v for k,v in source.items() if k.startswith('src/')}
    tasks=build_tasks(config,args.models_root,args.profile);hashes={}
    for name in sorted(set(spec['performance_models']+spec['credit_models'])):
        path=args.models_root/name/'final.pt';model(path,need_critic=name in spec['credit_models'],expected_core=core)
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    plan=dict(version='v7-policy-diagnostics-v1',started_utc=datetime.now(timezone.utc).isoformat(),config=config,profile=args.profile,
        checkpoint_sha256=hashes,source=source,task_count=len(tasks),gamma=GAMMA,gae_lambda=LAM,discount_reference_seconds=REFERENCE_SECONDS,
        scope='diagnostic only; unchanged checkpoints, no training or acceptance; case reuse and finite replica uncertainty explicit',accepted_win=False)
    write(out/'plan.json',plan);result=dict(plan=plan,performance={},credit={})
    # Canonical import supports the immutable-source runpy launcher under spawn.
    from scripts.v7_policy_diagnostics import worker as execute
    with ProcessPoolExecutor(max_workers=config['workers'],mp_context=get_context('spawn')) as pool:
        pending={pool.submit(execute,t):t for t in tasks}
        for completed,future in enumerate(as_completed(pending),1):
            task=pending[future];row=future.result();group='credit' if task['type']=='credit' else 'performance'
            result[group].setdefault(task['tag'],[]).append(row)
            write(out/'results.json',result);write(out/'progress.json',dict(phase='diagnostics',completed=completed,total=len(tasks),kind=task['type'],mode=task['tag']))
    result['performance_summary']={}
    reference=sorted([a['record'] for a in result['performance']['teacher']],key=lambda r:r['seed'])
    for tag,rows in result['performance'].items():
        records=sorted([a['record'] for a in rows],key=lambda r:r['seed'])
        if [r['seed'] for r in records]!=sorted(spec['performance_cases']):raise ValueError('incomplete performance mode')
        result['performance_summary'][tag]=dict(summary=summarize(records),comparison_to_teacher=compare(records,reference) if tag!='teacher' else None,
            note='paired case bootstrap, not CRN or independent sampling-seed cases; diagnostic, not acceptance')
    for tag,summary in result['performance_summary'].items():
        if '-sampled-' in tag and not tag.startswith('initial-'):
            sampling=tag.rsplit('-sampled-',1)[1]
            records=sorted([a['record'] for a in result['performance'][tag]],key=lambda r:r['seed'])
            initial=sorted([a['record'] for a in result['performance'][f'initial-sampled-{sampling}']],key=lambda r:r['seed'])
            summary['comparison_to_initial_sampled']=compare(records,initial)
    for tag,rows in result['credit'].items():
        if sorted(r['seed'] for r in rows)!=sorted(spec['credit_cases']):raise ValueError('incomplete credit mode')
    result['completed_utc']=datetime.now(timezone.utc).isoformat();write(out/'results.json',result)
    write(out/'progress.json',dict(phase='complete',accepted_win=False))


if __name__=='__main__':main()
