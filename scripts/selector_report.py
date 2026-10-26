"""Summarize all fixed selector rollouts, refusing partial studies."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from trace_transport.experiment_io import write


KINDS=('current-mlp','history-mlp','gru','transformer')


def report(data):
    required={'adaptive','low-rtt-bdp-rule'}|{f'{kind}-{seed}' for kind in KINDS for seed in [1,2,3]}
    if 'completed_utc' not in data or set(data['modes'])!=required:raise ValueError('all predeclared modes must finish')
    seeds=sorted(data['plan']['seeds']);reference=data['modes']['adaptive']['summary'];modes={}
    for name,mode in data['modes'].items():
        if sorted(a['record']['seed'] for a in mode['audits'])!=seeds:raise ValueError('mode case coverage mismatch')
        s=mode['summary'];comparison=mode['comparison'];times=[a['inference_ms'] for a in mode['audits'] if a['inference_ms'] is not None]
        modes[name]=dict(summary=s,relative_goodput_gain=s['verified_goodput']/reference['verified_goodput']-1,
            miss_difference=s['interactive_miss_rate']-reference['interactive_miss_rate'],
            failed_guards=[k for k,v in comparison['guards'].items() if not v] if comparison else [],
            comparison=comparison,inference_ms_p50_p95=np.quantile(times,[.5,.95]).tolist() if times else None)
    groups={}
    for kind in KINDS:
        rows=[modes[f'{kind}-{seed}'] for seed in [1,2,3]]
        groups[kind]=dict(goodput_gain_range=[min(r['relative_goodput_gain'] for r in rows),max(r['relative_goodput_gain'] for r in rows)],
            miss_rate_range=[min(r['summary']['interactive_miss_rate'] for r in rows),max(r['summary']['interactive_miss_rate'] for r in rows)],
            all_quality_guards_pass_count=sum(not r['failed_guards'] for r in rows),
            scope='three fixed training seeds; ranges are descriptive, not confidence intervals')
    return dict(version='selector-executed-report-v1',started_utc=data['plan']['started_utc'],completed_utc=data['completed_utc'],
        accepted_win=False,scope='exploratory validation; all fixed models reported; reserved tests unopened; endpoint-0 training versus all-role deployment confounds attribution; inference cost measured but not simulated',
        modes=modes,groups=groups)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();data=json.loads(args.results.read_text());out=report(data)
    out['input_sha256']=hashlib.sha256(args.results.read_bytes()).hexdigest();out['report_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();write(args.output,out)


if __name__=='__main__':main()
