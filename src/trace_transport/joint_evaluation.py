"""Paired session evaluation, role ablations and conservative win conditions."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .joint_models import load_checkpoint
from .joint_training import Histories,_pack,_tensor
from .joint_workers import EnvironmentPool
from .joint_session import JointTransportEnv
from .joint_curriculum import sample_case

# Fixed tensor shapes are part of repeatable goodput inference, independent of worker timing.
MAX_GOODPUT_CLIENTS=4
MAX_GOODPUT_ENDPOINTS=2*MAX_GOODPUT_CLIENTS

BASELINES=[dict(rate=rate,scheduler=scheduler,feedback=feedback)
    for rate in ('fixed','aimd','bdp','deadline') for scheduler in ('edf','drr') for feedback in ('budgeted','sparse','deadline')]


def summarize(records):
    desired=sum(r['desired'].get('interactive',0) for r in records)
    ontime=sum(r['interactive_on_time'] for r in records)
    return dict(interactive_miss_rate=(desired-ontime)/max(1,desired),interactive_desired=desired,
        image_bytes=sum(r['image_bytes'] for r in records),map_bytes=sum(r['map_bytes'] for r in records),
        image_goodput=sum(r['image_bytes'] for r in records)/max(1e-9,sum(r['elapsed'] for r in records)),
        map_goodput=sum(r['map_bytes'] for r in records)/max(1e-9,sum(r['elapsed'] for r in records)),
        delivered_bytes=sum(r['delivered_bytes'] for r in records),completed=sum(r['completed'] for r in records),
        admitted=sum(r['messages'] for r in records),total_wire_bytes=sum(r['total_wire_bytes'] for r in records),
        wrong=sum(r['wrong'] for r in records),
        wire_per_delivered_byte=sum(r['total_wire_bytes'] for r in records)/max(1,sum(r['delivered_bytes'] for r in records)))


def baseline_rank(records,maximum_bulk):
    s=summarize(records);bulk=s['image_goodput']+s['map_goodput']
    return (bulk>=.95*maximum_bulk,-s['interactive_miss_rate'],-s['wire_per_delivered_byte'])


def constant_baseline_actor(actor,role):
    """Recognize exact initialized argmax/zero-residual policies, not approximate ones."""
    if not all(torch.isfinite(p).all() for p in actor.parameters()):return False
    zero=lambda p:bool(torch.count_nonzero(p)==0)
    if role=='sender':
        return all(zero(p) for layer in [actor.control,actor.query] for p in [layer.weight,layer.bias]) and zero(actor.follow.weight) and bool(actor.follow.bias[0]>0)
    return zero(actor.head.weight) and bool(actor.head.bias[0]>actor.head.bias[1:].max())


def evaluate(*,seeds,checkpoint=None,baseline=None,device='cpu',environments=8,production=20.,drain=20.,ood=False,
             sender_enabled=True,receiver_enabled=True,trace_network=None,on_record=None,asynchronous=True,profile='deadline-v1'):
    baseline=baseline or dict(rate='fixed',scheduler='edf',feedback='budgeted')
    torch.set_num_threads(1)
    sender=receiver=config=None
    if checkpoint:
        sender,receiver,config,metadata=load_checkpoint(checkpoint,device)
        recorded=metadata.get('profile','deadline-v1')
        if profile!='deadline-v1' and profile!=recorded:raise ValueError('checkpoint/profile mismatch')
        profile=recorded
        teacher_config=metadata.get('baseline',dict(rate='fixed',scheduler='edf',feedback='budgeted'))
    infer_sender=checkpoint is not None and sender_enabled and not constant_baseline_actor(sender,'sender')
    infer_receiver=checkpoint is not None and receiver_enabled and not constant_baseline_actor(receiver,'receiver')
    infer=infer_sender or infer_receiver
    pool=EnvironmentPool(min(environments,len(seeds)))
    obs=[];histories=[];assigned={};next_index=0;records=[]
    def reset(i,seed):
        teacher=dict(baseline)
        if checkpoint and sender_enabled:teacher.update(rate=teacher_config['rate'],scheduler=teacher_config['scheduler'])
        if checkpoint and sender_enabled and 'initial_rate' in teacher_config:teacher['initial_rate']=teacher_config['initial_rate']
        if checkpoint and receiver_enabled:
            teacher['feedback']=teacher_config['feedback']
            for key in ('batch','delay'):
                if key in teacher_config:teacher[key]=teacher_config[key]
        o,_,label=pool.reset(i,seed,production,drain,'full',ood,teacher,trace_network,profile)
        assigned[i]=(seed,label)
        return o,Histories(o,config) if infer else None
    try:
        for i in range(pool.count):
            o,h=reset(i,seeds[next_index]);next_index+=1;obs.append(o);histories.append(h)
        if not infer:
            # Deterministic/constant policies need no per-tick actor IPC. Workers
            # execute the identical control steps locally, preserving timer order.
            from multiprocessing.connection import wait
            by_pipe={pool.pipes[i]:i for i in assigned}
            for i in assigned:pool.pipes[i].send(('finish-baseline',None))
            while assigned:
                for pipe in wait([pool.pipes[i] for i in assigned]):
                    i=by_pipe[pipe];metrics=pool.receive(i);seed,label=assigned[i]
                    metrics.update(seed=seed,domain=label);records.append(metrics)
                    if on_record is not None:on_record(metrics)
                    if next_index<len(seeds):
                        obs[i],histories[i]=reset(i,seeds[next_index]);next_index+=1
                        pool.pipes[i].send(('finish-baseline',None))
                    else:del assigned[i]
            return sorted(records,key=lambda r:r['seed'])
        def stable_actions(active):
            result=[]
            # The same world always gets the same tensor shapes and row positions,
            # independent of which other CPU workers finish first.
            for i in active:
                o,h=obs[i],histories[i];h.append(o);n=len(o['sender'])
                if n>MAX_GOODPUT_ENDPOINTS:raise ValueError(f'goodput fixed-shape evaluator supports at most {MAX_GOODPUT_CLIENTS} clients')
                sender_action=np.zeros((n,3),np.float32);receiver_action=np.zeros(n,np.int64)
                valid=np.zeros((MAX_GOODPUT_ENDPOINTS,config.history),bool);valid[:n]=h.valid
                def tensor(x):return torch.as_tensor(x,device=device)
                with torch.inference_mode():
                    if infer_sender and o['sender_active'].any():
                        history=np.zeros((MAX_GOODPUT_ENDPOINTS,config.history,h.sender.shape[-1]),np.float32);history[:n]=h.sender
                        queue=np.zeros((MAX_GOODPUT_ENDPOINTS,*o['queue'].shape[1:]),np.float32);queue[:n]=o['queue']
                        mask=np.zeros((MAX_GOODPUT_ENDPOINTS,o['queue_mask'].shape[1]),bool);mask[:n]=o['queue_mask']
                        a=sender.act(tensor(history),tensor(valid),tensor(queue),tensor(mask)).cpu().numpy()
                        sender_action[o['sender_active']]=a[:n][o['sender_active']]
                    if infer_receiver and o['receiver_active'].any():
                        history=np.zeros((MAX_GOODPUT_ENDPOINTS,config.history,h.receiver.shape[-1]),np.float32);history[:n]=h.receiver
                        a=receiver.act(tensor(history),tensor(valid)).cpu().numpy()
                        receiver_action[o['receiver_active']]=a[:n][o['receiver_active']]
                result.append((sender_action,receiver_action))
            return result
        def actions_for(active):
            if profile.startswith('goodput-'):return stable_actions(active)
            current=[obs[i] for i in active]
            actions=[(np.zeros((len(o['sender']),3),np.float32),np.zeros(len(o['receiver']),np.int64)) for o in current]
            hs=[histories[i] for i in active]
            for o,h in zip(current,hs):h.append(o)
            with torch.inference_mode():
                if infer_sender:
                    packed,mapping=_pack(current,hs,'sender',device)
                    if packed is not None:
                        a=sender.act(*packed).cpu().numpy()
                        for (e,j),value in zip(mapping,a):actions[e][0][j]=value
                if infer_receiver:
                    packed,mapping=_pack(current,hs,'receiver',device)
                    if packed is not None:
                        a=receiver.act(*packed).cpu().numpy()
                        for (e,j),value in zip(mapping,a):actions[e][1][j]=value
            return actions
        def accept(i,result):
            nonlocal next_index
            o,_,_,done,metrics,_=result;obs[i]=o
            if done:
                seed,label=assigned[i];metrics.update(seed=seed,domain=label)
                records.append(metrics)
                if on_record is not None:on_record(metrics)
                if next_index<len(seeds):
                    obs[i],histories[i]=reset(i,seeds[next_index]);next_index+=1
                else:del assigned[i]

        if asynchronous:
            from multiprocessing.connection import wait
            by_pipe={pool.pipes[i]:i for i in assigned}
            ready=set(assigned);pending=set()
            while assigned:
                active=sorted(ready)
                if active:
                    actions=actions_for(active)
                    for i,action in zip(active,actions):pool.send_step(i,action)
                    pending.update(active);ready.clear()
                for pipe in wait([pool.pipes[i] for i in pending]):
                    i=by_pipe[pipe];pending.remove(i);accept(i,pool.receive(i))
                    if i in assigned:ready.add(i)
        else:
            while assigned:
                active=sorted(assigned);actions=actions_for(active)
                for i,action in zip(active,actions):pool.send_step(i,action)
                for i in active:accept(i,pool.receive(i))
        return sorted(records,key=lambda r:r['seed'])
    finally:pool.close()


def select_baseline(records_by_name):
    maximum=max(summarize(rows)['image_goodput']+summarize(rows)['map_goodput'] for rows in records_by_name.values())
    return max(records_by_name,key=lambda name:baseline_rank(records_by_name[name],maximum))


def compare(learned,reference):
    if [r['seed'] for r in learned]!=[r['seed'] for r in reference]:raise ValueError('paired seed mismatch')
    a,b=summarize(learned),summarize(reference)
    rng=np.random.default_rng(42);diff=[]
    for _ in range(2000):
        ids=rng.integers(0,len(learned),len(learned))
        x=summarize([learned[i] for i in ids]);y=summarize([reference[i] for i in ids])
        diff.append(x['interactive_miss_rate']-y['interactive_miss_rate'])
    ci=[float(x) for x in np.quantile(diff,[.025,.975])]
    reduction=(b['interactive_miss_rate']-a['interactive_miss_rate'])/b['interactive_miss_rate'] if b['interactive_miss_rate'] else None
    guards=dict(image_goodput=a['image_goodput']>=.95*b['image_goodput'],map_goodput=a['map_goodput']>=.95*b['map_goodput'],
        image_volume=a['image_bytes']>=.95*b['image_bytes'],map_volume=a['map_bytes']>=.95*b['map_bytes'],
        completion=a['completed']/max(1,a['admitted'])>=b['completed']/max(1,b['admitted']),
        wire=a['wire_per_delivered_byte']<=1.05*b['wire_per_delivered_byte'],integrity=a['wrong']==0)
    return dict(relative_miss_reduction=reduction,miss_difference_ci95=ci,guards=guards,
        win=reduction is not None and reduction>=.2 and ci[1]<0 and all(guards.values()),
        notes='seed-block bootstrap; zero-miss baseline is a ceiling, not a metric switch')


def headroom(seed=40000,steps=5,production=2.,drain=2.,lookahead=20):
    """Privileged one-step candidate search using exact clones; not an upper bound."""
    cfg,w,_=sample_case(seed,production=production,drain=drain,stage='full')
    env=JointTransportEnv(cfg,w);env.reset(seed=seed)
    decisions=[]
    for _ in range(steps):
        if env.done():break
        best=None
        # Global candidate actions; privileged reward/state selection is diagnostic only.
        for rate in (-.5,0.,.5):
            for ack in (0,9,21):
                fork=env.clone();a=np.zeros((env.count,3));a[:,0]=rate
                reward=0.
                for _ in range(lookahead):
                    if fork.done():break
                    _,r,_,_,_=fork.step(a,np.full(env.count,ack,dtype=int));reward+=r
                candidate=(reward,-rate*rate,-ack)
                if best is None or candidate>best[0]:best=(candidate,rate,ack)
        _,rate,ack=best;a=np.zeros((env.count,3));a[:,0]=rate
        _,reward,_,_,_=env.step(a,np.full(env.count,ack,dtype=int))
        decisions.append(dict(time=env.now,rate=rate,ack=ack,reward=reward))
    return dict(kind='privileged finite-candidate look-ahead probe, not an optimal bound',decisions=decisions,metrics=env.metrics())
