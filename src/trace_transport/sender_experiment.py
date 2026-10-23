"""Explore the larger sender control space after receiver-v3.

Residual correction to local BDP-window/AIMD decisions; frozen strong receipt
scheduler. Initial and intermediate snapshots compete on validation only.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch

from trace_transport.learning import generalized_advantages
from trace_transport.receiver_curriculum import DOMAIN,sample_receiver_episode
from trace_transport.receiver_experiment import scenarios,summary,paired_comparison
from trace_transport.sender_residual import make_model,episode,save,load


def evaluate(model,baseline,seeds):
    rows=[]
    for name,cfg in scenarios().items():
        for seed in seeds:
            result,score,_=episode(cfg,seed=seed,model=model,baseline=baseline)
            rows.append(dict(scenario=name,seed=seed,completed=sum(r.completed for r in result),
                transfers=len(result),wrong=0,elapsed=float(np.mean([r.elapsed_seconds for r in result])),
                feedback_bytes=sum(r.feedback_wire_bytes for r in result),data_bytes=sum(r.data_wire_bytes for r in result),
                retries=sum(r.retransmitted_payload_bytes for r in result),score=float(score.mean())))
    return rows


def train(seed,episodes,output):
    model=make_model(seed);optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    rng=random.Random(seed);paths=[];buffer=[];logs=[]
    def snapshot(index):
        path=output/f'sender-residual-s{seed}-e{index:04d}.pt'
        save(path,model,dict(seed=seed,episodes=index,profile='persistent-randomized'))
        paths.append(path)
    snapshot(0)
    for index in range(episodes):
        cfg,flows,size=sample_receiver_episode(rng)
        result,score,traj=episode(cfg,seed=12000000+seed*10000+index,payload_bytes=size,flows=flows,model=model,stochastic=True)
        h,v,latent,oldp,values,rewards,terminal,active=[torch.stack(x) for x in zip(*traj)]
        adv,returns=generalized_advantages(rewards,values,terminal,torch.zeros(flows),gamma=.999)
        mask=active.flatten()
        buffer.append((h.flatten(0,1)[mask],v.flatten(0,1)[mask],latent.flatten(0,1)[mask],
                       oldp.flatten()[mask],adv.flatten()[mask],returns.flatten()[mask]))
        if len(buffer)==4 or index==episodes-1:
            hist,valid,latent,oldp,adv,returns=[torch.cat(x) for x in zip(*buffer)]
            adv=(adv-adv.mean())/adv.std().clamp_min(1e-6)
            stop=False
            for _ in range(4):
                for ids in torch.randperm(len(adv)).split(256):
                    logp,entropy,value=model.evaluate(hist[ids],valid[ids],latent[ids])
                    ratio=(logp-oldp[ids]).exp()
                    if ((ratio-1)-(logp-oldp[ids])).mean().item()>.015:stop=True;break
                    loss=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
                    loss=loss+.5*(value-returns[ids]).square().mean()-.001*entropy.mean()
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
                if stop:break
            buffer.clear()
        logs.append(dict(episode=index,network=asdict(cfg),flows=flows,payload_bytes=size,
                         completed=sum(r.completed for r in result),score=float(score.mean())))
        if (index+1)%80==0 or index==episodes-1:
            snapshot(index+1);print(f'seed={seed} episodes={index+1}/{episodes}',flush=True)
    (output/f'train-s{seed}.json').write_text(json.dumps(logs,indent=2)+'\n')
    return paths


def sender_comparison(learned,baseline):
    # A sender can trade ACK reductions for data retransmissions. Use total wire
    # cost for the criterion, while preserving the feedback-only breakdown.
    total=lambda rows:[{**r,'feedback_bytes':r['feedback_bytes']+r['data_bytes']} for r in rows]
    result=paired_comparison(total(learned),total(baseline))
    result['metrics']['total_wire_bytes']=result['metrics'].pop('feedback_bytes')
    result['metrics']['feedback_bytes']=paired_comparison(learned,baseline)['metrics']['feedback_bytes']
    result['criterion']='exact completion plus latency/total-wire Pareto condition; not ACK bytes alone'
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=320)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--output',type=Path,default=Path('runs/sender-residual-v1'))
    p.add_argument('--validation-trials',type=int,default=5)
    p.add_argument('--test-trials',type=int,default=20)
    args=p.parse_args(argv)
    if min(args.episodes,args.validation_trials,args.test_trials)<1 or max(args.validation_trials,args.test_trials)>1000:p.error('invalid budgets')
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    plan=dict(version='sender-residual-v1',episodes=args.episodes,seeds=args.seeds,domain=DOMAIN,
        validation_seeds=list(range(10000,10000+args.validation_trials)),test_seeds=list(range(11000,11000+args.test_trials)),
        baseline='local BDP-window + AIMD rate',residual_scale=.5,receiver='receipt batch8/25ms',history=8,
        selection='validation completion, then common transport-v2 return; initial snapshot eligible',
        optimizer=dict(lr=1e-4,epochs=4,minibatch=256,episodes_per_update=4,clip=.2,target_kl=.015,entropy=.001))
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    paths=[path for seed in args.seeds for path in train(seed,args.episodes,args.output)]
    paths=[path for path in paths if 'e0000' not in path.stem or f's{args.seeds[0]}-' in path.stem]
    validation={}
    for path in paths:
        rows=evaluate(load(path),'bdp-aimd',plan['validation_seeds'])
        validation[path.name]=dict(summary=summary(rows),records=rows)
        print('validation',path.name,validation[path.name]['summary'],flush=True)
    baselines={}
    for name in ['aimd','bdp-aimd','fixed']:
        rows=evaluate(None,name,plan['validation_seeds'])
        baselines[name]=dict(summary=summary(rows),records=rows)
    rank=lambda d:(d['summary']['completed'],d['summary']['mean_score'])
    winner=max(validation,key=lambda name:rank(validation[name]));baseline=max(baselines,key=lambda name:rank(baselines[name]))
    (args.output/'selection.json').write_text(json.dumps(dict(winner=winner,baseline=baseline,validation=validation,baselines=baselines),indent=2)+'\n')
    final={}
    for name,model,base in [('selected',load(args.output/winner),'bdp-aimd'),*[(name,None,name) for name in baselines]]:
        rows=evaluate(model,base,plan['test_seeds']);final[name]=dict(summary=summary(rows),records=rows)
        print('test',name,final[name]['summary'],flush=True)
    best=args.output/'sender-residual-best.pt';best.write_bytes((args.output/winner).read_bytes())
    report=dict(plan=plan,winner=winner,baseline=baseline,validation=validation,baseline_validation=baselines,final_test=final,
        comparison=sender_comparison(final['selected']['records'],final[baseline]['records']))
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
