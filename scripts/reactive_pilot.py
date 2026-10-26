"""Controlled reactive v6 pilot matched to the completed v5 shaped arm; validation only."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
import platform
import numpy as np
from pathlib import Path
import torch

from trace_transport.experiment_io import write
from trace_transport.goodput_reactive import DOMAIN,sample_case
from trace_transport.goodput_evaluation import summarize,compare
from trace_transport.learning_diagnostics import stratify
from trace_transport.joint_models import JointModelConfig
from trace_transport.joint_training import JointTrainConfig,train
from trace_transport.joint_evaluation import evaluate


def validation_bank():
    families={f:[] for f in DOMAIN['families']}
    for seed in range(134000,140000):
        f=sample_case(seed)[2]['family']
        if len(families[f])<2:families[f].append(seed)
        if all(len(s)==2 for s in families.values()):break
    missing={f:len(rows) for f,rows in families.items() if len(rows)!=2}
    if missing:raise ValueError(f"incomplete validation strata: {missing}")
    return sorted(s for rows in families.values() for s in rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--decisions',type=int,default=200000);args=p.parse_args();out=args.output
    if (out/'plan.json').exists():raise ValueError('use a new pilot output directory')
    seeds=validation_bank()
    baseline=dict(rate='delay-fast',scheduler='edf',feedback='budgeted',initial_rate=512000.,batch=1,delay=0.)
    model=JointModelConfig(history=64,width=32,layers=1)
    cfg=JointTrainConfig(profile='goodput-v6',time_aware_gae=True,mask_receiver_alias=True,progress_shaping=True,
        receiver_teacher_prior=.5,teacher_rate='delay-fast',teacher_initial_rate=512000.,teacher_batch=1,teacher_delay=0.,
        decisions=args.decisions,environments=2,rollout=512,epochs=3,checkpoint_every=100000,
        production_seconds=12.,drain_seconds=3.,easy_fraction=0.,gae_lambda=.98,device='cpu',seed=42)
    plan=dict(version='reactive-v6-pilot-v1',started_utc=datetime.now(timezone.utc).isoformat(),domain=DOMAIN,
        base_training=asdict(cfg),model=asdict(model),baseline=baseline,validation_seeds=seeds,
        source=json.loads(Path('source-manifest.json').read_text()),runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu'),
        scope='one-seed controlled pilot; final checkpoints fixed, no snapshot selection or reserved acceptance tests',
        difference='v6 pending ACK semantics; configuration matches v5 shaped pilot, no reserved acceptance claim')
    write(out/'plan.json',plan)
    write(out/'progress.json',dict(phase='training',variant='reactive'))
    train(out/'reactive',cfg,model)
    result=dict(plan=plan,acceptance_claim=False,modes={})
    for name,checkpoint in [('adaptive',None),('reactive',out/'reactive/final.pt')]:
        partial=[]
        def record(row):
            partial.append(row);write(out/'progress.json',dict(phase='validation',mode=name,completed=len(partial),total=len(seeds)))
            write(out/(name+'-records.json'),partial)
        rows=evaluate(seeds=seeds,checkpoint=checkpoint,baseline=baseline,environments=2,production=12.,drain=3.,profile='goodput-v6',on_record=record)
        result['modes'][name]=dict(summary=summarize(rows),strata=stratify(rows),records=rows,
            comparison=compare(rows,result['modes']['adaptive']['records']) if name!='adaptive' else None)
        write(out/'results.json',result)
    result['completed_utc']=datetime.now(timezone.utc).isoformat();write(out/'results.json',result)
    write(out/'progress.json',dict(phase='complete',reserved_tests='unopened'))


if __name__=='__main__':main()
