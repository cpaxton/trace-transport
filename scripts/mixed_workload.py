"""Probe congestion-control headroom with concurrent bulk and short messages.

Two 128-KiB bulk transfers share queues with 16 staggered 1-KiB messages. Frozen
policies only; no selection or retraining on this challenge set. Report short
message latency separately from bulk completion and total wire traffic.
"""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import numpy as np
import torch

from trace_transport.benchmark import benchmark_text
from trace_transport.network_models import wifi_cases
from trace_transport.simulator import NetworkConfig
from trace_transport.sender_residual import episode,load


def cases():
    wifi={c['name']:NetworkConfig(**c['network']) for c in wifi_cases()}
    return {'clean-shared':NetworkConfig(loss=0,ack_loss=0,jitter=.001,deadline=30),
            'small-queue':NetworkConfig(bandwidth_bps=500000,queue_bytes=4096,deadline=30),
            'wifi-mobile-fade':replace(wifi['wifi-mobile-fade'],deadline=30),
            'wifi-burst-loss':replace(wifi['wifi-burst-loss'],deadline=30),
            'reverse-constrained':NetworkConfig(feedback_bandwidth_bps=32000,deadline=30)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials',type=int,default=5)
    p.add_argument('--output',type=Path,default=Path('runs/mixed-workload-v1'))
    args=p.parse_args()
    if not 1<=args.trials<=1000:p.error('invalid trials')
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    texts=[benchmark_text(131072)]*2+[benchmark_text(1024)]*16
    starts=[0.,0.]+[.1+i*.1 for i in range(16)]
    plan=dict(version='mixed-workload-v1',seed_start=15000,trials=args.trials,
              payload_bytes=[len(t.encode()) for t in texts],starts=starts,
              cases={k:asdict(v) for k,v in cases().items()},
              selection='none; frozen policies; exploratory challenge set')
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    model=load('runs/sender-residual-v1/sender-residual-best.pt')
    final={}
    for label in ['selected','aimd','bdp-aimd','fixed']:
        records=[]
        for name,cfg in cases().items():
            for seed in range(15000,15000+args.trials):
                result,_,_=episode(cfg,seed=seed,texts=texts,start_times=starts,
                    model=model if label=='selected' else None,baseline='bdp-aimd' if label=='selected' else label)
                bulk,short=result[:2],result[2:]
                records.append(dict(scenario=name,seed=seed,completed=sum(r.completed for r in result),
                    transfers=len(result),wrong=0,bulk_completed=sum(r.completed for r in bulk),
                    bulk_elapsed=[r.elapsed_seconds for r in bulk],short_completed=sum(r.completed for r in short),
                    short_elapsed=[r.elapsed_seconds for r in short],
                    data_bytes=sum(r.data_wire_bytes for r in result),feedback_bytes=sum(r.feedback_wire_bytes for r in result),
                    total_wire_bytes=sum(r.data_wire_bytes+r.feedback_wire_bytes for r in result)))
        short=np.array([t for r in records for t in r['short_elapsed']]);bulk=np.array([t for r in records for t in r['bulk_elapsed']])
        final[label]=dict(records=records,summary=dict(completed=sum(r['completed'] for r in records),
            transfers=sum(r['transfers'] for r in records),wrong=0,short_p50=float(np.quantile(short,.5)),
            short_p95=float(np.quantile(short,.95)),bulk_mean=float(bulk.mean()),
            total_wire_bytes=sum(r['total_wire_bytes'] for r in records)))
        print(label,final[label]['summary'],flush=True)
    (args.output/'results.json').write_text(json.dumps(dict(plan=plan,final_test=final),indent=2)+'\n')


if __name__=='__main__':main()
