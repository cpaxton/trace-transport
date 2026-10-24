"""Check case-by-case repeatability across evaluation worker counts; no selection."""
import argparse
import json
from pathlib import Path
from trace_transport.joint_evaluation import evaluate
from scripts.goodput_study import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    kwargs=dict(seeds=list(range(73000,73016)),checkpoint=args.checkpoint,profile='goodput-v2',production=6.,drain=2.)
    a=evaluate(**kwargs,environments=1);b=evaluate(**kwargs,environments=2)
    result=dict(kind='evaluation repeatability diagnostic; not a policy win',checkpoint=str(args.checkpoint),
        seeds=kwargs['seeds'],worker_counts=[1,2],identical=a==b,different_cases=[x['seed'] for x,y in zip(a,b) if x!=y],records_a=a,records_b=b)
    write(args.report,result)
    if a!=b:raise SystemExit('case records differ; do not claim a repeatable win')


if __name__=='__main__':main()
