"""Evaluate the frozen residual sender against three fixed baselines.

No selection on the new generalization seeds. Receipt feedback is identical for
all senders. All scenario-level records include completed/receiver-complete and
total wire bytes to expose reliability and data/feedback tradeoffs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from trace_transport.network_models import generalization_cases
from trace_transport.sender_residual import episode,load
from trace_transport.receiver_experiment import summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials',type=int,default=20)
    p.add_argument('--output',type=Path,default=Path('runs/protocol-generalization-v1'))
    args=p.parse_args()
    if not 1<=args.trials<=1000:p.error('invalid trials')
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    selection=json.loads(Path('runs/sender-residual-v1/selection.json').read_text())
    plan=dict(test_seeds=list(range(13000,13000+args.trials)),selected=selection['winner'],
              baseline_selected_on_validation=selection['baseline'],
              selection='none on this generalization suite',receiver='receipt batch8/25ms')
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    model=load('runs/sender-residual-v1/sender-residual-best.pt')
    final={}
    for label in ['selected','aimd','bdp-aimd','fixed']:
        records=[]
        for name,(cfg,flows,size) in generalization_cases().items():
            for seed in plan['test_seeds']:
                result,score,_=episode(cfg,seed=seed,flows=flows,payload_bytes=size,
                    model=model if label=='selected' else None,baseline='bdp-aimd' if label=='selected' else label)
                data=sum(r.data_wire_bytes for r in result);feedback=sum(r.feedback_wire_bytes for r in result)
                records.append(dict(scenario=name,seed=seed,completed=sum(r.completed for r in result),
                    receiver_complete=sum(r.receiver_complete for r in result),transfers=len(result),wrong=0,
                    elapsed=float(np.mean([r.elapsed_seconds for r in result])),feedback_bytes=feedback,
                    data_bytes=data,total_wire_bytes=data+feedback,retries=sum(r.retransmitted_payload_bytes for r in result),
                    score=float(score.mean())))
        final[label]=dict(summary=summary(records),records=records)
        print(label,final[label]['summary'],flush=True)
    (args.output/'results.json').write_text(json.dumps(dict(plan=plan,final_test=final),indent=2)+'\n')


if __name__=='__main__':main()
