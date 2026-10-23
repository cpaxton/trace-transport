"""Baseline-initialized categorical PPO, validation-selected snapshots.

Three seeds, 320 episodes each. Initial snapshots are eligible, so training must
beat its strong starting policy on validation to be selected. No test selection.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from trace_transport.learning import generalized_advantages
from trace_transport.receiver_curriculum import DOMAIN,sample_receiver_episode
from trace_transport.receiver_discrete import ACTIONS,WIDE_ACTIONS,make_categorical
from trace_transport.receiver_learning import episode,save,load
from trace_transport.receiver_experiment import evaluate,summary,paired_comparison


def train(seed,episodes,output,history=8,wide=False):
    model=make_categorical(seed,history,wide=wide)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    rng=random.Random(seed)
    buffer=[];logs=[];paths=[]
    def snapshot(index):
        path=output/f'receiver-v3-s{seed}-e{index:04d}.pt'
        save(path,model,dict(seed=seed,episodes=index,profile='persistent-randomized',
                            policy='categorical',initial_prior='8% batch8/delay250ms' if wide else '90% batch8/delay25ms'))
        paths.append(path)
    snapshot(0)
    start=time.monotonic()
    for index in range(episodes):
        cfg,flows,size=sample_receiver_episode(rng)
        if wide:cfg=replace(cfg,feedback_delay_budget=.25)
        result,score,trajectory=episode(cfg,seed=8000000+seed*10000+index,
            payload_bytes=size,flows=flows,model=model,stochastic=True,receipt_feedback=True)
        h,v,action,oldp,values,rewards,terminal,active=[torch.stack(x) for x in zip(*trajectory)]
        adv,returns=generalized_advantages(rewards,values,terminal,torch.zeros(flows),gamma=.999)
        mask=active.flatten()
        buffer.append((h.flatten(0,1)[mask],v.flatten(0,1)[mask],action.flatten()[mask],
                       oldp.flatten()[mask],adv.flatten()[mask],returns.flatten()[mask]))
        if len(buffer)==4 or index==episodes-1:
            hist,valid,action,oldp,adv,returns=[torch.cat(x) for x in zip(*buffer)]
            adv=(adv-adv.mean())/adv.std().clamp_min(1e-6)
            stop=False
            for _ in range(4):
                for ids in torch.randperm(len(adv)).split(256):
                    logp,entropy,value=model.evaluate(hist[ids],valid[ids],action[ids])
                    ratio=(logp-oldp[ids]).exp()
                    kl=((ratio-1)-(logp-oldp[ids])).mean()
                    if kl.item()>.015:
                        stop=True;break
                    loss=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
                    loss=loss+.5*(value-returns[ids]).square().mean()-.001*entropy.mean()
                    optimizer.zero_grad();loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
                if stop:break
            buffer.clear()
        logs.append(dict(episode=index,network=asdict(cfg),flows=flows,payload_bytes=size,
                         completed=sum(r.completed for r in result),decisions=int(active.sum()),
                         score=float(score.mean())))
        if (index+1)%40==0 or index==episodes-1:
            snapshot(index+1)
            print(f'seed={seed} episodes={index+1}/{episodes}',flush=True)
    (output/f'train-s{seed}.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,episodes=logs),indent=2)+'\n')
    return paths


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wide',action='store_true',help='learn a wider action bank with a configured .25s delay budget')
    p.add_argument('--episodes',type=int,default=320)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--validation-trials',type=int,default=5)
    p.add_argument('--test-trials',type=int,default=20)
    args=p.parse_args(argv)
    if min(args.episodes,args.validation_trials,args.test_trials)<1 or max(args.validation_trials,args.test_trials)>1000:
        p.error('invalid budgets')
    args.output=args.output or Path('runs/receiver-wide-v1' if args.wide else 'runs/receiver-v3')
    budget=.25 if args.wide else None
    validation_start,test_start=(21000,22000) if args.wide else (8000,9000)
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    plan=dict(version='receiver-wide-v1' if args.wide else 'receiver-v3',seeds=args.seeds,episodes=args.episodes,actions=WIDE_ACTIONS if args.wide else ACTIONS,
              feedback_delay_budget=budget,wide=args.wide,
              history=8,training_domain=DOMAIN,validation_seeds=list(range(validation_start,validation_start+args.validation_trials)),
              test_seeds=list(range(test_start,test_start+args.test_trials)),snapshot_every=40,
              selection='validation completion first, then common transport-v2 return; initial snapshot eligible',
              optimizer=dict(lr=1e-4,epochs=4,episodes_per_update=4,minibatch=256,clip=.2,target_kl=.015,entropy=.001),
              initial_prior='8% batch8/delay250ms; rest uniform' if args.wide else '90% batch8/delay25ms, other controls share 10%')
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    paths=[path for seed in args.seeds for path in train(seed,args.episodes,args.output,wide=args.wide)]
    validation={}
    # Initial models are identical in behavior; retain one initial snapshot to
    # avoid redundant validation, while every trained checkpoint remains eligible.
    paths=[p for p in paths if 'e0000' not in p.stem or f's{args.seeds[0]}-' in p.stem]
    for path in paths:
        rows=evaluate(load(path),None,plan['validation_seeds'],receipt_feedback=True,feedback_delay_budget=budget)
        validation[path.name]=dict(summary=summary(rows),records=rows)
        print('validation',path.name,validation[path.name]['summary'],flush=True)
    winner=max(validation,key=lambda name:(validation[name]['summary']['completed'],validation[name]['summary']['mean_score']))
    baseline=(8,.025)
    baseline_validation=[]
    if args.wide:
        for action in WIDE_ACTIONS:
            rows=evaluate(None,action,plan['validation_seeds'],receipt_feedback=True,feedback_delay_budget=budget)
            baseline_validation.append(dict(action=action,summary=summary(rows),records=rows))
        baseline=max(baseline_validation,key=lambda r:(r['summary']['completed'],r['summary']['mean_score']))['action']
    (args.output/'selection.json').write_text(json.dumps(dict(winner=winner,baseline=baseline,
        baseline_validation=baseline_validation,validation=validation),indent=2)+'\n')
    final={}
    for name,model,control in [('selected',load(args.output/winner),None),('initial',load(paths[0]),None),('deterministic',None,baseline)]:
        rows=evaluate(model,control,plan['test_seeds'],receipt_feedback=True,feedback_delay_budget=budget)
        final[name]=dict(summary=summary(rows),records=rows)
        print('test',name,final[name]['summary'],flush=True)
    best=args.output/'receiver-v3-best.pt';best.write_bytes((args.output/winner).read_bytes())
    report=dict(plan=plan,winner=winner,baseline=baseline,baseline_validation=baseline_validation,validation=validation,final_test=final,
                comparison=paired_comparison(final['selected']['records'],final['deterministic']['records']),
                sha256=hashlib.sha256(best.read_bytes()).hexdigest())
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
