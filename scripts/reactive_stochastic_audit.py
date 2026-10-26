"""Predeclared four-case, three-sampling-seed receiver deployment diagnostic."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import torch
from trace_transport.experiment_io import write
from trace_transport.goodput_evaluation import summarize
from scripts.reactive_diagnostics import action_case


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output
    if (out/'plan.json').exists():raise ValueError('use new output')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    original=json.loads((args.root/'results.json').read_text());checkpoint=args.root/'reactive/final.pt'
    seeds=original['plan']['validation_seeds'][:4];sampling_seeds=[1,2,3]
    plan=dict(version='reactive-stochastic-audit-v1',started_utc=datetime.now(timezone.utc).isoformat(),
        seeds=seeds,sampling_seeds=sampling_seeds,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source=json.loads(Path('source-manifest.json').read_text()),
        scope='reused four validation cases; sampled receiver with greedy sender; diagnostic not policy selection, training or held-out acceptance; fixed sample seeds; all outcomes reported')
    write(out/'plan.json',plan);result=dict(plan=plan,modes={})
    for name in ['adaptive','reactive']:
        records=[r for r in original['modes'][name]['records'] if r['seed'] in seeds]
        result['modes'][name]=dict(summary=summarize(records),records=records)
    for sampling_seed in sampling_seeds:
        audits=[]
        for seed in seeds:
            write(out/'progress.json',dict(phase='sampled-receiver',seed=seed,sampling_seed=sampling_seed))
            audits.append(action_case(seed,checkpoint,original['plan']['baseline'],receiver_mode='sampled',sampling_seed=sampling_seed))
        result['modes'][f'sampled-{sampling_seed}']=dict(summary=summarize([a['record'] for a in audits]),audits=audits)
        write(out/'results.json',result)
    result['completed_utc']=datetime.now(timezone.utc).isoformat();write(out/'results.json',result)
    write(out/'progress.json',dict(phase='complete'))


if __name__=='__main__':main()
