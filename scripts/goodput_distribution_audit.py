"""Coverage and legal-control sensitivity audit; no learned acceptance claims."""
from dataclasses import asdict
from datetime import datetime,timezone
import argparse
import hashlib
from pathlib import Path

from scripts.goodput_study import write
from trace_transport.goodput_distribution import DOMAIN,FAMILIES,sample_case
from trace_transport.goodput_evaluation import summarize
from trace_transport.joint_workers import EnvironmentPool

CONTROLS={
    'delay-fast':dict(rate='delay-fast',scheduler='edf',feedback='budgeted',batch=16,delay=.025),
    'delay-ack1':dict(rate='delay',scheduler='edf',feedback='budgeted',batch=1,delay=0.),
    'delay-ack4':dict(rate='delay',scheduler='edf',feedback='budgeted',batch=4,delay=.005),
    'delay-ack16':dict(rate='delay',scheduler='edf',feedback='budgeted',batch=16,delay=.025),
    'delay-ack32':dict(rate='delay',scheduler='edf',feedback='budgeted',batch=32,delay=.1),
    'delay-drr':dict(rate='delay',scheduler='drr',feedback='budgeted',batch=16,delay=.025),
    'aimd':dict(rate='aimd',scheduler='edf',feedback='budgeted',batch=16,delay=.025),
    'bdp':dict(rate='bdp',scheduler='edf',feedback='budgeted',batch=16,delay=.025),
    'fixed':dict(rate='fixed',scheduler='edf',feedback='budgeted',batch=16,delay=.025)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/distribution-v3-audit'))
    p.add_argument('--cases-per-family',type=int,default=2)
    args=p.parse_args();out=args.output
    seeds={f:[] for f in FAMILIES};coverage=[]
    for seed in range(90000,91000):
        cfg,w,label=sample_case(seed)
        coverage.append(dict(seed=seed,network=asdict(cfg),workload=asdict(w),label=label))
        if len(seeds[label['family']])<args.cases_per_family:seeds[label['family']].append(seed)
    sources={n:hashlib.sha256((Path('src/trace_transport')/n).read_bytes()).hexdigest()
             for n in ['goodput_distribution.py','goodput.py','joint_session.py','joint_workers.py']}
    plan=dict(version='distribution-v3-audit-v1',started_utc=datetime.now(timezone.utc).isoformat(),
              domain=DOMAIN,seeds=seeds,controls=CONTROLS,production=6.,drain=2.,source_sha256=sources,
              purpose='synthetic coverage and legal-control sensitivity, not training or reserved tests')
    write(out/'plan.json',plan);write(out/'coverage.json',coverage)
    results=dict(plan=plan,families={})
    pool=EnvironmentPool(1)
    try:
        for family,bank in seeds.items():
            records={}
            for name,control in CONTROLS.items():
                rows=[]
                for seed in bank:
                    baseline=dict(control,initial_rate=512000.)
                    _,_,label=pool.reset(0,seed,6.,2.,'full',False,baseline,None,'goodput-v3')
                    pool.pipes[0].send(('finish-baseline',None));row=pool.receive(0)
                    row.update(seed=seed,domain=label);rows.append(row)
                    write(out/'progress.json',dict(family=family,control=name,seed=seed))
                records[name]=dict(summary=summarize(rows),records=rows)
            results['families'][family]=records;write(out/'results.json',results)
    finally:pool.close()
    results['completed_utc']=datetime.now(timezone.utc).isoformat();write(out/'results.json',results)
    write(out/'progress.json',dict(phase='complete'))


if __name__=='__main__':main()
