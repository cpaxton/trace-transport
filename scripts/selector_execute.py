"""Execute endpoint-local probe-then-select controllers, including the probe cost.

All declared models/seeds are evaluated, without validation-based selection.
No coordination oracle or hindsight candidate lookup controls the network.
"""
import argparse
import ast
import zipfile
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from trace_transport.experiment_io import write
from trace_transport.goodput_reactive import GoodputReactiveEnv,sample_case
from trace_transport.goodput_evaluation import summarize,compare
from trace_transport.joint_session import SENDER_FEATURES
from scripts.selector_learning import Selector,KINDS


BASELINE=dict(rate='delay-fast',scheduler='edf',feedback='budgeted',initial_rate=512000.,batch=1,delay=0.)


def load_selector(path):
    d=torch.load(path,map_location='cpu',weights_only=True)
    if d.get('sender_features')!=list(SENDER_FEATURES) or d.get('kind') not in KINDS or d['choices']!=len(d['controls']):raise ValueError('selector schema mismatch')
    if d['mean'].shape!=(len(SENDER_FEATURES),) or d['scale'].shape!=d['mean'].shape or not torch.isfinite(d['mean']).all() or not torch.isfinite(d['scale']).all() or not (d['scale']>0).all():raise ValueError('invalid normalizer')
    model=Selector(d['kind'],history=d['history'],choices=d['choices']);model.load_state_dict(d['state']);model.eval()
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite selector')
    return model,d


def execute_case(seed,*,selector=None,heuristic=False,production=12.,drain=3.):
    model,meta=selector if selector is not None else (None,None)
    cfg,w,label=sample_case(seed,production=production,drain=drain)
    env=GoodputReactiveEnv(replace(cfg,initial_rate=BASELINE['initial_rate']),w,churn=label.get('churn',False),churn_times=label.get('churn_times'))
    env.reset(seed=seed)
    def install(s,b):
        # Retain the current rate/window, cooldown and measurement state. No reset,
        # shadow controller, free rate jump or extra coordination is introduced.
        s.rate_control=b['rate'];s.scheduler=b['scheduler']
        s.batch=s.baseline_batch=b['batch'];s.delay=s.baseline_delay=b['delay']
        s.ack_mode=s.baseline_ack_mode='constant'
    for s in env.sessions:install(s,BASELINE)
    obs=env.observe();history=meta['history'] if meta is not None else 64
    h=np.zeros((env.count,history,len(SENDER_FEATURES)),np.float32);valid=np.zeros((env.count,history),bool)
    zero=(np.zeros((env.count,3)),np.zeros(env.count,int));choices=None;inference_ms=None
    probe_steps=max(1,math.floor(min(1.,production)/cfg.control_interval))
    step=0
    while not env.done():
        if step<probe_steps:
            h[:,:-1]=h[:,1:].copy();h[:,-1]=obs['sender'];valid[:,:-1]=valid[:,1:].copy();valid[:,-1]=True
        if step==probe_steps and (model is not None or heuristic):
            if heuristic:
                rtt=SENDER_FEATURES.index('min_rtt');ack=SENDER_FEATURES.index('ack_rate')
                choices=[int(row[-1,rtt]<np.log1p(5.) and row[-1,ack]>0) for row in h]
                controls=[BASELINE,{**BASELINE,'rate':'bdp'}]
            else:
                x=((torch.as_tensor(h)-meta['mean'])/meta['scale']).clamp(-10,10)
                start=time.perf_counter()
                with torch.inference_mode():choices=model(x,torch.as_tensor(valid)).argmax(-1).tolist()
                inference_ms=(time.perf_counter()-start)*1000
                controls=[c['baseline'] for c in meta['controls']]
            for s,i in zip(env.sessions,choices):install(s,controls[i])
        obs,_,_,_,_=env.step(*zero);step+=1
    row=env.metrics();row.pop('messages_detail',None);row.update(seed=seed,domain=label)
    return dict(record=row,endpoint_choices=choices,inference_ms=inference_ms,
        probe_seconds=probe_steps*cfg.control_interval,scope='local decision once after baseline probe; inference wall time measured but not charged in simulated time')


_CACHE={}


def worker_trial(task):
    seed,path,heuristic=task
    torch.set_num_threads(1)
    if path is not None and path not in _CACHE:_CACHE[path]=load_selector(path)
    return execute_case(seed,selector=_CACHE.get(path),heuristic=heuristic)


def verified_baselines(prior_path,archive,seeds,source):
    prior=json.loads(Path(prior_path).read_text())
    if prior['plan']['seeds']!=seeds or set(prior['modes'])-{'adaptive','low-rtt-bdp-rule'}:raise ValueError('only identical complete baseline modes can be reused')
    core=lambda manifest:{k:v for k,v in manifest.items() if k.startswith('src/')}
    if core(prior['plan']['source'])!=core(source):raise ValueError('baseline simulator changed')
    with zipfile.ZipFile(archive) as z:old=z.read('scripts/selector_execute.py')
    if hashlib.sha256(old).hexdigest()!=prior['plan']['source']['scripts/selector_execute.py']:raise ValueError('baseline source archive mismatch')
    def policy(text):
        nodes=ast.parse(text).body
        return [ast.dump(n,include_attributes=False) for n in nodes if
            isinstance(n,ast.FunctionDef) and n.name=='execute_case' or
            isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BASELINE' for t in n.targets)]
    if policy(old)!=policy(Path(__file__).read_text()):raise ValueError('baseline execution policy changed')
    for mode in prior['modes'].values():
        observed=[a['record']['seed'] for a in mode['audits']]
        if sorted(observed)!=sorted(seeds):raise ValueError('baseline cases incomplete')
    return prior['modes']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prior-baselines',type=Path);p.add_argument('--prior-source',type=Path)
    args=p.parse_args();out=args.output
    if out.exists() and any(f.name!='service.log' for f in out.iterdir()):raise ValueError('new output required')
    torch.set_num_threads(1);data=json.loads(args.dataset.read_text());learning=json.loads((args.models/'results.json').read_text())
    if 'completed_utc' not in data or 'completed_utc' not in learning or learning['plan']['tiny_only']:raise ValueError('completed collection and full fitting required')
    if hashlib.sha256(args.dataset.read_bytes()).hexdigest()!=learning['plan']['dataset_sha256']:raise ValueError('dataset changed after fitting')
    seeds=data['plan']['splits']['validation'];files=[args.models/f'{kind}-{seed}.pt' for kind in KINDS for seed in [1,2,3]]
    if any(not f.exists() for f in files):raise ValueError('missing declared model')
    plan=dict(version='selector-execute-v1',started_utc=datetime.now(timezone.utc).isoformat(),seeds=seeds,
        model_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        source=json.loads(Path('source-manifest.json').read_text()),
        workers=4,scope='exploratory validation, all 12 final models and simple rule; no snapshot/model selection, reserved tests unopened; probe charged, measured inference latency not simulated')
    modes={}
    if bool(args.prior_baselines)!=bool(args.prior_source):raise ValueError('both prior baseline provenance paths required')
    if args.prior_baselines:
        modes=verified_baselines(args.prior_baselines,args.prior_source,seeds,plan['source'])
        plan['reused_baseline_provenance']=dict(results_sha256=hashlib.sha256(args.prior_baselines.read_bytes()).hexdigest(),source_archive_sha256=hashlib.sha256(args.prior_source.read_bytes()).hexdigest(),modes=list(modes),verification='exact policy AST, baseline constant, simulator source and seed memberships; no learned modes reused')
    write(out/'plan.json',plan);result=dict(plan=plan,modes=modes)
    # Canonical import remains pickleable when launched through a runpy bootstrap.
    from scripts.selector_execute import worker_trial as task_worker
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as pool:
        for name,path in [('adaptive',None),('low-rtt-bdp-rule',None)]+[(f.stem,f) for f in files]:
            if name in result['modes']:continue
            audits=[]
            jobs={pool.submit(task_worker,(seed,str(path) if path else None,name=='low-rtt-bdp-rule')):seed for seed in seeds}
            for future in as_completed(jobs):
                audits.append(future.result())
                write(out/'progress.json',dict(phase='executed-validation',mode=name,seed=jobs[future],completed=len(audits),total=len(seeds)))
            audits.sort(key=lambda a:a['record']['seed']);rows=[a['record'] for a in audits]
            result['modes'][name]=dict(summary=summarize(rows),audits=audits,
                comparison=compare(rows,[a['record'] for a in result['modes']['adaptive']['audits']]) if name!='adaptive' else None)
            write(out/'results.json',result)
    result['completed_utc']=datetime.now(timezone.utc).isoformat();write(out/'results.json',result)
    write(out/'progress.json',dict(phase='complete',accepted_win=False))


if __name__=='__main__':main()
