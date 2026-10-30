"""Fixed-budget 2x2 rollout/GAE pilot; unchanged task, actor and reward."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import torch
from trace_transport.experiment_io import write
from trace_transport.reactive_sender_training import Config,train
from trace_transport.goodput_evaluation import summarize,compare
from scripts.reactive_sender_pilot import mirror,inference_benchmark
from scripts.v7_policy_diagnostics import worker


def configurations(config,profile):
    spec=config['profiles'][profile]
    for key in ['training_cases','validation_cases','optimizer_seeds','sampling_seeds']:
        xs=spec[key]
        if not xs or len(xs)!=len(set(xs)) or any(type(x) is not int or x<0 for x in xs):raise ValueError(f'invalid {key}')
    if set(spec['training_cases'])&set(spec['validation_cases']):raise ValueError('training/evaluation overlap')
    arms=config['arms']
    if {(a['rollout'],a['gae_lambda']) for a in arms}!={(16,.98),(64,.98),(16,.998),(64,.998)} or len(arms)!=4:raise ValueError('complete unique 2x2 required')
    if len({a['name'] for a in arms})!=4 or any(not a['name'].replace('-','').isalnum() for a in arms):raise ValueError('invalid arm names')
    if not 1<=spec['decisions']<=50000:raise ValueError('bounded per-run budget required')
    return [(f"seed{seed}-{a['name']}",Config(decisions=spec['decisions'],seed=seed,kind='current-mlp',rollout=a['rollout'],gae_lambda=a['gae_lambda']))
            for seed in spec['optimizer_seeds'] for a in arms]


def evaluation(root,name,checkpoint,cases,sampling,*,baselines=False):
    tasks=[]
    if baselines:
        tasks.extend(dict(type='performance',tag='teacher',checkpoint=None,mode='teacher',seed=s,sampling_seed=0) for s in cases)
        for r in sampling:tasks.extend(dict(type='performance',tag=f'initial-sampled-{r}',checkpoint=None,mode='sampled',seed=s,sampling_seed=r) for s in cases)
    else:
        tasks.extend(dict(type='performance',tag='greedy',checkpoint=str(checkpoint),mode='greedy',seed=s,sampling_seed=0) for s in cases)
        for r in sampling:tasks.extend(dict(type='performance',tag=f'sampled-{r}',checkpoint=str(checkpoint),mode='sampled',seed=s,sampling_seed=r) for s in cases)
    rows={}
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as pool:
        pending={pool.submit(worker,t):t for t in tasks}
        for future in as_completed(pending):
            task=pending[future];rows.setdefault(task['tag'],[]).append(future.result())
            write(root/'progress.json',dict(phase='evaluation',run=name,completed=sum(map(len,rows.values())),total=len(tasks)))
    for tag,records in rows.items():
        records.sort(key=lambda r:r['record']['seed'])
        if [r['record']['seed'] for r in records]!=sorted(cases):raise ValueError('incomplete evaluation')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--persistent',type=Path,required=True)
    p.add_argument('--profile',choices=['engineering','pilot'],default='engineering');args=p.parse_args()
    config=json.loads(args.config.read_text());spec=config['profiles'][args.profile];runs=configurations(config,args.profile);root=args.root
    if root.exists() and any(f.name!='service.log' for f in root.iterdir()):raise ValueError('new run directory required')
    if args.persistent.exists() and any(args.persistent.iterdir()):raise ValueError('new persistent directory required')
    torch.set_num_threads(1);source=json.loads(Path('source-manifest.json').read_text())
    for rel,digest in source.items():
        if hashlib.sha256(Path(rel).read_bytes()).hexdigest()!=digest:raise ValueError('source manifest mismatch')
    latency=inference_benchmark(['current-mlp'])
    if latency['current-mlp']['warm_ms_p50_p95'][1]>20:raise ValueError('modeled inference delay exceeded')
    plan=dict(version='v7-long-credit-v1',started_utc=datetime.now(timezone.utc).isoformat(),source=source,
        config=config,profile=args.profile,actual_training={name:asdict(cfg) for name,cfg in runs},inference_benchmark=latency,
        gamma=.999,discount_reference_seconds=.02,scope='development ablation; reused validation; no model selection or reserved acceptance',accepted_win=False)
    write(root/'plan.json',plan);result=dict(plan=plan,runs={})
    def sync():mirror(root,args.persistent)
    try:
        sync()
        baseline=evaluation(root,'baselines',None,spec['validation_cases'],spec['sampling_seeds'],baselines=True)
        result['baselines']=baseline;write(root/'results.json',result);sync()
        teacher=[r['record'] for r in baseline['teacher']]
        for name,cfg in runs:
            write(root/'progress.json',dict(phase='training',run=name))
            train(root/name,cfg,spec['training_cases'],source,on_update=lambda _:sync())
            rows=evaluation(root,name,root/name/'final.pt',spec['validation_cases'],spec['sampling_seeds'])
            summaries={}
            for tag,entries in rows.items():
                records=[r['record'] for r in entries]
                summaries[tag]=dict(summary=summarize(records),comparison_to_teacher=compare(records,teacher),
                    scope='case bootstrap only; action sampling seeds separate; diagnostic not acceptance')
                if tag.startswith('sampled-'):
                    initial=[r['record'] for r in baseline['initial-'+tag]]
                    summaries[tag]['comparison_to_initial_sampled']=compare(records,initial)
            result['runs'][name]=dict(training=asdict(cfg),checkpoint_sha256=hashlib.sha256((root/name/'final.pt').read_bytes()).hexdigest(),
                progress=json.loads((root/name/'progress.json').read_text()),performance=rows,summaries=summaries)
            write(root/'results.json',result);sync()
        result['completed_utc']=datetime.now(timezone.utc).isoformat();write(root/'results.json',result)
        write(root/'progress.json',dict(phase='complete',accepted_win=False));sync()
    except BaseException as exc:
        write(root/'failure.json',dict(error=repr(exc),utc=datetime.now(timezone.utc).isoformat()));sync();raise


if __name__=='__main__':main()
