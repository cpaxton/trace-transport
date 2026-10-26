"""Collect independent development/validation control outcomes and causal probes.

Dataset construction only: hindsight labels must never become actor inputs.
No learned policy, reserved acceptance tests, or real-device performance claim.
"""
import argparse
from dataclasses import replace
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import platform
import shutil
import numpy as np
import torch
from trace_transport.experiment_io import write
from trace_transport.goodput_reactive import DOMAIN,GoodputReactiveEnv,sample_case
from trace_transport.goodput_evaluation import summarize
from trace_transport.joint_evaluation import evaluate
from trace_transport.joint_session import SENDER_FEATURES
from scripts.reactive_diagnostics import bank


def probe_history(seed,baseline,seconds=1.,history=64):
    cfg,w,label=sample_case(seed,production=12.,drain=3.)
    env=GoodputReactiveEnv(replace(cfg,initial_rate=baseline['initial_rate']),w,
        churn=label.get('churn',False),churn_times=label.get('churn_times'))
    env.reset(seed=seed)
    for s in env.sessions:
        s.rate_control=baseline['rate'];s.scheduler=baseline['scheduler']
        s.batch=s.baseline_batch=baseline['batch'];s.delay=s.baseline_delay=baseline['delay']
        s.ack_mode=s.baseline_ack_mode='constant'
    obs=env.observe();h=np.zeros((history,len(SENDER_FEATURES)),np.float32);valid=np.zeros(history,bool)
    # Observe only the first sender's endpoint-local feedback along baseline actions.
    # Neither family nor future outcomes enter the feature tensor.
    for _ in range(max(1,math.floor(seconds/cfg.control_interval))):
        h[:-1]=h[1:].copy();h[-1]=obs['sender'][0];valid[:-1]=valid[1:].copy();valid[-1]=True
        obs,_,_,_,_=env.step(np.zeros((env.count,3)),np.zeros(env.count,int))
    return dict(seed=seed,history=h.tolist(),valid=valid.tolist(),probe_elapsed=env.now,
        schema=list(SENDER_FEATURES),endpoint=0,scope='causal baseline probe only; labels stored separately')


def persist(root,target):
    target.mkdir(parents=True,exist_ok=True);paths=list(root.glob('*.json'))
    total=sum(p.stat().st_size for p in paths)
    if total>128*1024**2:raise RuntimeError('JSON mirror exceeds 128MiB')
    for p in paths:
        if shutil.disk_usage(target).free-p.stat().st_size<1024**3:raise RuntimeError('preserve 1GiB disk free')
        dest=target/p.name;tmp=dest.with_suffix('.copying');shutil.copyfile(p,tmp);tmp.replace(dest)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--persistent',type=Path,required=True);args=p.parse_args();out=args.output
    if (out/'plan.json').exists():raise ValueError('new collection directory required')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    controls=[dict(name=f'{rate}-b{batch}-d{delay}',baseline=dict(rate=rate,scheduler='edf',feedback='budgeted',initial_rate=512000.,batch=batch,delay=delay))
        for rate in ['delay-fast','aimd','bdp'] for batch,delay in [(1,0.),(8,.005),(32,.025)]]
    splits=dict(development=bank(170000,4),validation=bank(180000,2))
    if set(splits['development'])&set(splits['validation']):raise ValueError('overlapping splits')
    plan=dict(version='selector-coverage-v1',started_utc=datetime.now(timezone.utc).isoformat(),
        domain=DOMAIN,splits=splits,controls=controls,production=12.,drain=3.,probe_seconds=1.,history=64,
        source=json.loads(Path('source-manifest.json').read_text()),machine=platform.machine(),
        runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
        scope='72 independent synthetic cases, 648 full control trials plus 72 baseline probes; diagnostic data only; no reserved tests; no training or online selection performance claim',
        next_gate='fit tiny dataset then compare MLP/GRU/transformer selection using only local probes, with labels and family excluded from inputs; validate executed policy separately before claiming transport gains')
    write(out/'plan.json',plan);persist(out,args.persistent)
    result=dict(plan=plan,splits={})
    try:
        for split,seeds in splits.items():
            result['splits'][split]=dict(controls={},probes=[])
            for seed in seeds:
                write(out/'progress.json',dict(phase='probes',split=split,seed=seed))
                result['splits'][split]['probes'].append(probe_history(seed,controls[0]['baseline']))
                write(out/'results.json',result)
            for control in controls:
                write(out/'progress.json',dict(phase='control-outcomes',split=split,control=control['name']))
                rows=evaluate(seeds=seeds,baseline=control['baseline'],environments=4,production=12.,drain=3.,profile='goodput-v6')
                result['splits'][split]['controls'][control['name']]=dict(records=rows,summary=summarize(rows))
                write(out/'results.json',result);persist(out,args.persistent)
        result['completed_utc']=datetime.now(timezone.utc).isoformat();write(out/'results.json',result)
        write(out/'progress.json',dict(phase='complete',reserved_tests='unopened'));persist(out,args.persistent)
    except BaseException as exc:
        write(out/'failure.json',dict(error=repr(exc),utc=datetime.now(timezone.utc).isoformat()))
        persist(out,args.persistent);raise


if __name__=='__main__':main()
