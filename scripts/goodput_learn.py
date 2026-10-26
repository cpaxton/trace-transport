"""Bounded single-run joint learning plus validation; never opens test seeds."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

from trace_transport.joint_training import JointTrainConfig,train
from trace_transport.joint_models import JointModelConfig
from trace_transport.experiment_io import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('runs/goodput-v21'))
    p.add_argument('--run',default='learn42');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--decisions',type=int,default=1000000);args=p.parse_args()
    reference=json.loads((args.root/'reference.json').read_text());baseline=reference['baseline']
    cfg=JointTrainConfig(profile='goodput-v2',teacher_rate=baseline['rate'],teacher_initial_rate=baseline['initial_rate'],
        teacher_batch=baseline.get('batch',8),teacher_delay=baseline.get('delay',.25 if baseline['feedback']=='sparse' else .025),
        paired_teacher=True,decisions=args.decisions,environments=2,rollout=64,epochs=3,seed=args.seed,
        checkpoint_every=100000,production_seconds=6,drain_seconds=2,easy_fraction=.1,gae_lambda=.98,device='cpu')
    model=JointModelConfig(history=16,width=32,layers=1)
    output=args.root/args.run
    write(output/'experiment.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),
        kind='joint learning and validation only',training=asdict(cfg),model=asdict(model),reference=reference['name']))
    train(output,cfg,model)
    subprocess.run([sys.executable,'-m','scripts.goodput_sweep','--root',str(args.root),'--run',args.run],check=True)
    write(output/'completed.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        status='training and validation finished; reserved tests unopened'))


if __name__=='__main__':main()
