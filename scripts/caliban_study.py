"""Bounded CPU scaling benchmark and longer multi-seed v4 training study.

The long study never opens reserved test seeds. Persistent mirroring stores a
bounded latest checkpoint per seed; runtime environments and raw logs stay local.
"""
import argparse
import _thread
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import threading
import time
import uuid

import numpy as np
import torch
from trace_transport.experiment_io import write
from trace_transport.goodput_deployment import DOMAIN,FAMILIES,sample_case
from trace_transport.goodput_evaluation import summarize,compare
from trace_transport.joint_evaluation import evaluate
from trace_transport.joint_models import JointModelConfig,SenderActor,ReceiverActor,load_checkpoint
from trace_transport.joint_session import SENDER_FEATURES,RECEIVER_FEATURES,QUEUE_FEATURES,MAX_CANDIDATES
from trace_transport.joint_training import JointTrainConfig,train


def model_config():return JointModelConfig(history=64,width=64,layers=2)


def config(seed,decisions,envs,baseline=None):
    b=baseline or dict(rate='delay-fast',scheduler='edf',batch=16,delay=.025)
    return JointTrainConfig(profile='goodput-v4',teacher_rate=b['rate'],teacher_scheduler=b['scheduler'],
        teacher_initial_rate=b.get('initial_rate',512000.),teacher_batch=b['batch'],teacher_delay=b['delay'],
        receiver_teacher_prior=.5,paired_teacher=False,decisions=decisions,environments=envs,rollout=128,
        epochs=3,checkpoint_every=500000,production_seconds=12.,drain_seconds=3.,
        easy_fraction=.05,gae_lambda=.98,device='cpu',seed=seed)


def bank(start,per_family=2):
    if not isinstance(per_family,int) or per_family<1:raise ValueError('positive cases per family required')
    seeds={f:[] for f in FAMILIES}
    for seed in range(start,start+10000):
        family=sample_case(seed)[2]['family']
        if len(seeds[family])<per_family:seeds[family].append(seed)
        if all(len(s)==per_family for s in seeds.values()):break
    missing={f:per_family-len(rows) for f,rows in seeds.items() if len(rows)!=per_family}
    if missing:raise ValueError('seed bank exhausted with uncovered strata: '+str(missing))
    return sorted(s for rows in seeds.values() for s in rows)


def validate_resume_plan(prior,plan):
    keys=['domain','training_seeds','decisions_per_seed','environments','model','source_sha256',
          'calibration_seeds','validation_seeds','reserved_test','reserved_ood']
    for key in keys:
        if prior[key]!=plan[key]:raise ValueError('resume plan/source mismatch: '+key)


def benchmark(output,decisions):
    torch.set_num_threads(1);output.mkdir(parents=True,exist_ok=True)
    result=dict(version='caliban-v4-benchmark-v1',started_utc=datetime.now(timezone.utc).isoformat(),
                machine=platform.machine(),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                cuda_available=torch.cuda.is_available(),affinity=sorted(os.sched_getaffinity(0)),trials=[])
    for envs in [1,2,4]:
        cfg=config(42,decisions,envs)
        # Full distribution rather than easy-only startup throughput.
        cfg=JointTrainConfig(**{**asdict(cfg),'easy_fraction':0.,'production_seconds':2.,'drain_seconds':.5})
        train(output/f'env{envs}',cfg,model_config())
        p=json.loads((output/f'env{envs}/progress.json').read_text())
        result['trials'].append(dict(environments=envs,decisions=p['decisions'],seconds=p['elapsed_seconds'],
                                     decisions_per_second=p['decisions_per_second']))
        write(output/'results.json',result)
    model=model_config();s=SenderActor(model);r=ReceiverActor(model)
    h=torch.zeros(8,model.history,len(SENDER_FEATURES));v=torch.ones(8,model.history,dtype=torch.bool)
    rh=torch.zeros(8,model.history,len(RECEIVER_FEATURES));q=torch.zeros(8,MAX_CANDIDATES,len(QUEUE_FEATURES));m=torch.ones(8,MAX_CANDIDATES,dtype=torch.bool)
    elapsed=[]
    with torch.inference_mode():
        for i in range(120):
            t=time.perf_counter();s.act(h,v,q,m);r.act(rh,v)
            if i>=20:elapsed.append((time.perf_counter()-t)*1000)
    result['eight_endpoint_actor_pair_ms_p50_p95']=np.quantile(elapsed,[.5,.95]).tolist()
    result['completed_utc']=datetime.now(timezone.utc).isoformat();write(output/'results.json',result)


def mirror_once(root,persistent):
    if shutil.disk_usage(persistent).free<1024**3:raise RuntimeError('persistent disk has <1GiB free; stop rather than fill it')
    allowed=list(root.glob('*.json'))
    for run in root.glob('seed*'):
        allowed.extend(run/name for name in ['config.json','progress.json','failure.json'] if (run/name).exists())
        allowed.extend(run.glob('*.pt'))
    size=sum(p.stat().st_size for p in allowed)
    if size>256*1024**2:raise RuntimeError('persistent checkpoint mirror exceeds 256MiB budget')
    for path in allowed:
        dest=persistent/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_suffix(dest.suffix+'.'+uuid.uuid4().hex+'.copying');shutil.copyfile(path,temp);temp.replace(dest)


def study(root,persistent,decisions,seed_list,resume=False):
    root.mkdir(parents=True,exist_ok=True);persistent.mkdir(parents=True,exist_ok=True)
    prior=json.loads((root/'plan.json').read_text()) if (root/'plan.json').exists() else None
    if prior is not None and not resume:raise ValueError('study already exists; use --resume explicitly')
    if resume and prior is None:raise ValueError('restore persistent study files before resuming')
    bench=prior['benchmark'] if prior else json.loads((root.parent/'benchmark/results.json').read_text())
    envs=max(bench['trials'],key=lambda x:x['decisions_per_second'])['environments']
    plan=dict(version='caliban-v4-study-v1',started_utc=datetime.now(timezone.utc).isoformat(),domain=DOMAIN,
        training_seeds=seed_list,decisions_per_seed=decisions,environments=envs,model=asdict(model_config()),
        calibration_seeds=bank(120000,1),validation_seeds=bank(121000,2),reserved_test=[123000,123999],reserved_ood=[124000,124999],
        selection='initial and every 500k snapshot, guards first then goodput, validation only',
        checkpoint_persistence='full optimizer snapshots and source archive; bounded 256MiB persistent mirror',
        benchmark=bench,source_sha256=json.loads((Path(__file__).resolve().parents[1]/'source-manifest.json').read_text()))
    if prior:
        validate_resume_plan(prior,plan)
        plan=prior
        write(root/'resumption.json',dict(utc=datetime.now(timezone.utc).isoformat(),collector_reset=True))
    write(root/'plan.json',plan)
    source=Path(__file__).resolve().parents[1]
    archive=source.parent/(source.name+'.zip')
    if archive.exists():shutil.copyfile(archive,persistent/'source.zip')
    shutil.copyfile(source/'source-manifest.json',persistent/'source-manifest.json')
    stop=threading.Event();errors=[]
    def mirror():
        while not stop.wait(30):
            try:mirror_once(root,persistent)
            except BaseException as e:errors.append(repr(e));stop.set();_thread.interrupt_main();return
    thread=threading.Thread(target=mirror,daemon=True);thread.start()
    try:
        # All references see the same calibration worlds; no validation tuning.
        controls={n:dict(rate=rate,scheduler='edf',feedback='budgeted',initial_rate=512000.,batch=batch,delay=delay)
                  for n,rate,batch,delay in [('delay-fast','delay-fast',16,.025),('immediate','delay-fast',1,0.),('bdp','bdp',16,.025),('aimd','aimd',16,.025)]}
        calibration={}
        for name,b in controls.items():
            rows=evaluate(seeds=plan['calibration_seeds'],baseline=b,environments=2,production=6.,drain=2.,profile='goodput-v4')
            calibration[name]=dict(summary=summarize(rows),records=rows)
            write(root/'calibration.json',calibration)
        minimum=min(v['summary']['interactive_miss_rate'] for v in calibration.values())
        eligible=[n for n,v in calibration.items() if v['summary']['interactive_miss_rate']<=minimum+.005]
        name=max(eligible,key=lambda n:calibration[n]['summary']['verified_goodput']);baseline=controls[name]
        write(root/'reference.json',dict(name=name,baseline=baseline))
        reference=evaluate(seeds=plan['validation_seeds'],baseline=baseline,environments=2,production=12.,drain=3.,profile='goodput-v4')
        write(root/'reference-validation.json',reference)
        for seed in seed_list:
            if errors:raise RuntimeError(errors)
            if (root/f'seed{seed}-selection.json').exists():continue
            cfg=config(seed,decisions,envs,baseline);output=root/f'seed{seed}'
            checkpoint=output/'latest.pt'
            if resume and checkpoint.exists():
                _,_,_,meta=load_checkpoint(checkpoint)
                if meta['decisions']<decisions:train(output,cfg,model_config(),resume=checkpoint)
                elif not (output/'final.pt').exists():shutil.copyfile(checkpoint,output/'final.pt')
            else:train(output,cfg,model_config())
            candidates=[root/f'seed{seed}/initial.pt',*sorted((root/f'seed{seed}').glob('d*.pt')),root/f'seed{seed}/final.pt']
            results={};seen=set()
            for checkpoint in candidates:
                _,_,_,meta=load_checkpoint(checkpoint);count=meta['decisions']
                if count in seen:continue
                seen.add(count)
                write(root/'progress.json',dict(phase='validation',seed=seed,checkpoint=checkpoint.name))
                rows=evaluate(seeds=plan['validation_seeds'],checkpoint=checkpoint,baseline=baseline,environments=2,
                              production=12.,drain=3.,profile='goodput-v4')
                results[checkpoint.name]=dict(decisions=count,summary=summarize(rows),comparison=compare(rows,reference),records=rows)
                write(root/f'seed{seed}-validation.json',results)
            selected=max(results,key=lambda n:(all(results[n]['comparison']['guards'].values()),results[n]['summary']['verified_goodput']))
            write(root/f'seed{seed}-selection.json',dict(checkpoint=selected,result=results[selected],selection_data_only=True))
            mirror_once(root,persistent)
        write(root/'progress.json',dict(phase='complete',completed_utc=datetime.now(timezone.utc).isoformat(),reserved_tests='unopened'))
    except BaseException as e:
        write(root/'study-failure.json',dict(error=repr(e),utc=datetime.now(timezone.utc).isoformat()));raise
    finally:
        stop.set();thread.join(timeout=5)
        if not errors:mirror_once(root,persistent)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['benchmark','study'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--persistent',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--decisions',type=int);p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    args=p.parse_args()
    if args.stage=='benchmark':benchmark(args.root,args.decisions or 2000)
    else:
        if args.persistent is None:p.error('--persistent required')
        study(args.root,args.persistent,args.decisions or 5000000,args.seeds,resume=args.resume)


if __name__=='__main__':main()
