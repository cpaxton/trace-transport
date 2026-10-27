"""Registered v7 engineering/pilot training, gated before using Caliban compute."""
import argparse
import hashlib
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from trace_transport.experiment_io import write
from trace_transport.reactive_sender import CONTRACT,FEATURES,BIASES,KEEP,MacroWorld,MacroPool
from trace_transport.reactive_sender_training import Config,Policy,train,load
from trace_transport.goodput_reactive import sample_case
from trace_transport.goodput_evaluation import summarize,compare


def audit(config):
    chosen=[]
    for family in ['wired-lan','capacity-opportunities','wifi-contention','bufferbloat']:
        chosen.append(next(s for s in config['training_cases'] if sample_case(s)[2]['family']==family))
    cases=[];examples=[]
    for seed in chosen:
        world=MacroWorld(seed)
        while world.env.now<1.1:world.step(np.full(world.env.count,KEEP))
        eligible=np.flatnonzero(world.observation()['eligible'])
        for index in eligible[:2]:
            rows=[]
            for choice in range(len(BIASES)):
                returns=[];misses=[];verified=[]
                for replica in range(2):
                    trial=world.clone()
                    trial.env.streams=[random.Random(seed*10000+int(index)*100+choice*10+replica*2+d) for d in range(2)]
                    before=trial.env.reward_counts['missed'];bulk=sum(trial.env.verified[c] for c in ['image','map'])
                    total=0.;first=True;reverted=False
                    while trial.env.now<world.env.now+1.-1e-9 and not trial.env.done():
                        a=np.full(trial.env.count,KEEP)
                        if first:a[index]=choice;first=False
                        elif not reverted and trial.observation()['eligible'][index]:a[index]=1;reverted=True
                        total+=trial.step(a)[6]
                    returns.append(total);misses.append(trial.env.reward_counts['missed']-before)
                    verified.append(sum(trial.env.verified[c] for c in ['image','map'])-bulk)
                rows.append(dict(choice=choice,bias=BIASES[choice],raw_returns=returns,mean_return=float(np.mean(returns)),replica_std=float(np.std(returns)),mean_misses=float(np.mean(misses)),mean_verified=float(np.mean(verified))))
            teacher=rows[1]
            feasible=[r for r in rows if r['mean_misses']<=teacher['mean_misses'] and r['mean_verified']>=teacher['mean_verified']]
            best=max(feasible,key=lambda r:r['mean_return']);signal=best['mean_return']-teacher['mean_return']
            positive=best['choice']!=1 and signal>2*max(best['replica_std'],teacher['replica_std'],1e-10)
            observation=world.observation();examples.append((observation['history'][index],observation['valid'][index],best['choice']))
            cases.append(dict(seed=seed,family=sample_case(seed)[2]['family'],producer_index=int(index),actions=rows,guarded_positive_signal=positive))
    if not examples:return dict(version='reactive-sender-v7-development-gate-v1',cases=cases,tiny_fit_accuracy={},positive_states=0,advance=False,note='no eligible development states found')
    fits={};torch.set_num_threads(1)
    for kind in ['current-mlp','history-mlp']:
        torch.manual_seed(17);model=Policy(kind);optimizer=torch.optim.Adam(model.parameters(),lr=.003)
        h=torch.tensor(np.stack([x[0] for x in examples]));valid=torch.tensor(np.stack([x[1] for x in examples]));y=torch.tensor([x[2] for x in examples])
        for _ in range(300):
            optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(h,valid),y);loss.backward();optimizer.step()
        with torch.no_grad():fits[kind]=float((model(h,valid).argmax(-1)==y).float().mean())
    return dict(version='reactive-sender-v7-development-gate-v1',completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='small finite-horizon development sensitivity gate; independent post-state packet-stream replicas, not CRN; teacher continuation, noisy labels, not optimal action values or held-out performance',
        cases=cases,tiny_fit_accuracy=fits,positive_states=sum(c['guarded_positive_signal'] for c in cases),
        advance=any(c['guarded_positive_signal'] for c in cases) and all(v==1. for v in fits.values()))


def mirror(root,target):
    target.mkdir(parents=True,exist_ok=True)
    paths=list(root.glob('*.json'))
    for d in root.glob('seed*'):paths.extend(p for p in d.iterdir() if p.suffix in ('.json','.pt'))
    if sum(p.stat().st_size for p in paths)>128*1024**2:raise RuntimeError('mirror exceeds 128MiB')
    for p in paths:
        if shutil.disk_usage(target).free-p.stat().st_size<1024**3:raise RuntimeError('preserve 1GiB disk free')
        dest=target/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix(dest.suffix+'.copying');shutil.copyfile(p,tmp);tmp.replace(dest)


def evaluate(policy,seeds):
    pool=MacroPool(2);rows=[]
    try:
        for offset in range(0,len(seeds),2):
            active={i:pool.reset(i,s) for i,s in enumerate(seeds[offset:offset+2])}
            while active:
                for i,o in active.items():
                    a=np.full(len(o['eligible']),KEEP)
                    if policy is not None:
                        with torch.inference_mode():pred=policy(torch.tensor(o['history']),torch.tensor(o['valid'])).argmax(-1).numpy()
                        a[o['eligible']]=pred[o['eligible']]
                    pool.pipes[i].send(('step',a))
                for i in list(active):
                    result=pool.receive(i)
                    if result[2]:rows.append(result[3]);del active[i]
                    else:active[i]=result[0]
    finally:pool.close()
    return sorted(rows,key=lambda r:r['seed'])


def inference_benchmark(kinds):
    output={}
    h=torch.zeros(16,CONTRACT['bins'],len(FEATURES));valid=torch.ones(16,CONTRACT['bins'],dtype=torch.bool)
    for kind in kinds:
        actor=Policy(kind).eval();samples=[]
        with torch.inference_mode():
            for _ in range(64):
                start=time.perf_counter();torch.distributions.Categorical(logits=actor(h,valid)).sample();samples.append((time.perf_counter()-start)*1000)
        output[kind]=dict(first_ms=samples[0],warm_ms_p50_p95=np.quantile(samples[1:],[.5,.95]).tolist())
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--persistent',type=Path)
    p.add_argument('--gate-only',action='store_true');p.add_argument('--gate',type=Path)
    args=p.parse_args();config=json.loads(args.config.read_text());root=args.root
    torch.set_num_threads(1)
    if args.gate_only:
        gate=audit(config);base=Path(__file__).resolve().parents[1]
        paths=list((base/'src').rglob('*.py'))+[Path(__file__).resolve()]
        gate['source_sha256']={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        gate['config_sha256']=hashlib.sha256(args.config.read_bytes()).hexdigest()
        write(root/'gate.json',gate);return
    if (root/'plan.json').exists():raise ValueError('new run required')
    gate=json.loads(args.gate.read_text())
    source=json.loads(Path('source-manifest.json').read_text())
    expected={k:v for k,v in source.items() if k.startswith('src/') or k=='scripts/reactive_sender_pilot.py'}
    if gate.get('source_sha256')!=expected:raise ValueError('gate/source mismatch')
    if gate.get('config_sha256')!=hashlib.sha256(args.config.read_bytes()).hexdigest():raise ValueError('gate/config mismatch')
    latency=inference_benchmark(config['kinds'])
    latency_ok=all(v['warm_ms_p50_p95'][1]<=CONTRACT['inference_delay_seconds']*1000 for v in latency.values())
    advance=gate['advance'] and latency_ok
    budget=config['pilot_decisions'] if advance else config['engineering_decisions']
    actual=[Config(decisions=budget,seed=seed,kind=kind) for kind in config['kinds'] for seed in config['optimizer_seeds']]
    plan=dict(version='reactive-sender-v7-pilot-v1',started_utc=datetime.now(timezone.utc).isoformat(),
        contract=CONTRACT,actual_training=[asdict(c) for c in actual],config=config,gate=gate,inference_benchmark=latency,latency_gate_pass=latency_ok,
        source=json.loads(Path('source-manifest.json').read_text()),scope='bounded training pilot' if advance else 'engineering training only; sensitivity or latency gate did not justify scaling',accepted_win=False)
    write(root/'plan.json',plan)
    def sync():
        if args.persistent:mirror(root,args.persistent)
    try:
        sync()
        reference=None;result=dict(plan=plan,modes={})
        for cfg in actual:
            name=f'seed{cfg.seed}-{cfg.kind}';write(root/'progress.json',dict(phase='training',run=name))
            train(root/name,cfg,config['training_cases'],plan['source'],on_update=lambda _:sync())
            policy,_=load(root/name/'final.pt')
            if reference is None:
                write(root/'progress.json',dict(phase='baseline-reference'))
                reference=evaluate(None,config['validation_cases']);write(root/'reference.json',dict(summary=summarize(reference),records=reference));sync()
            write(root/'progress.json',dict(phase='validation',run=name))
            rows=evaluate(policy,config['validation_cases'])
            result['modes'][name]=dict(summary=summarize(rows),comparison=compare(rows,reference),records=rows)
            write(root/'results.json',result);sync()
        result['completed_utc']=datetime.now(timezone.utc).isoformat();write(root/'results.json',result)
        write(root/'progress.json',dict(phase='complete',accepted_win=False));sync()
    except BaseException as exc:
        write(root/'failure.json',dict(error=repr(exc),utc=datetime.now(timezone.utc).isoformat()));sync();raise


if __name__=='__main__':main()
