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


def evaluate(*,seeds,checkpoint=None,baseline=None,device='cpu',environments=8,production=20.,drain=20.,ood=False,
             sender_enabled=True,receiver_enabled=True,trace_network=None):
    baseline=baseline or dict(rate='fixed',scheduler='edf',feedback='budgeted')
    torch.set_num_threads(1)
    sender=receiver=config=None
    if checkpoint:sender,receiver,config,metadata=load_checkpoint(checkpoint,device)
    pool=EnvironmentPool(min(environments,len(seeds)))
    obs=[];histories=[];assigned={};next_index=0;records=[]
    def reset(i,seed):
        teacher=dict(baseline)
        if checkpoint and sender_enabled:teacher.update(rate='fixed',scheduler='edf')
        if checkpoint and receiver_enabled:teacher['feedback']='budgeted'
        o,_,label=pool.reset(i,seed,production,drain,'full',ood,teacher,trace_network)
        assigned[i]=(seed,label)
        return o,Histories(o,config) if checkpoint else None
    try:
        for i in range(pool.count):
            o,h=reset(i,seeds[next_index]);next_index+=1;obs.append(o);histories.append(h)
        while assigned:
            # Park completed workers at a zero-work step by sending only to active pipes.
            active=sorted(assigned);current=[obs[i] for i in active]
            actions=[(np.zeros((len(o['sender']),3),np.float32),np.zeros(len(o['receiver']),np.int64)) for o in current]
            if checkpoint:
                hs=[histories[i] for i in active]
                for o,h in zip(current,hs):h.append(o)
                with torch.inference_mode():
                    if sender_enabled:
                        packed,mapping=_pack(current,hs,'sender',device)
                        if packed is not None:
                            a=sender.act(*packed).cpu().numpy()
                            for (e,j),value in zip(mapping,a):actions[e][0][j]=value
                    if receiver_enabled:
                        packed,mapping=_pack(current,hs,'receiver',device)
                        if packed is not None:
                            a=receiver.act(*packed).cpu().numpy()
                            for (e,j),value in zip(mapping,a):actions[e][1][j]=value
            for i,action in zip(active,actions):pool.pipes[i].send(('step',action))
            for i in active:
                o,_,_,done,metrics,_=pool.receive(i);obs[i]=o
                if done:
                    seed,label=assigned[i];metrics.update(seed=seed,domain=label)
                    records.append(metrics)
                    if next_index<len(seeds):
                        obs[i],histories[i]=reset(i,seeds[next_index]);next_index+=1
                    else:del assigned[i]
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
