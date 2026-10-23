"""Frozen wide-delay receiver on untouched generalization seeds."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch

from trace_transport.network_models import generalization_cases
from trace_transport.receiver_learning import episode,load
from trace_transport.receiver_experiment import summary,paired_comparison


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials',type=int,default=20)
    p.add_argument('--output',type=Path,default=Path('runs/receiver-wide-generalization-v1'))
    args=p.parse_args()
    if not 1<=args.trials<=1000:p.error('invalid trials')
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    selection=json.loads(Path('runs/receiver-wide-v1/selection.json').read_text())
    plan=dict(test_seeds=list(range(23000,23000+args.trials)),winner=selection['winner'],
              baseline=selection['baseline'],feedback_delay_budget=.25,selection='none on these generalization seeds')
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    model=load('runs/receiver-wide-v1/receiver-v3-best.pt')
    final={}
    for label,policy,action in [('selected',model,None),('best-constant',None,selection['baseline']),
                               ('latency',None,(8,.025)),('sparse',None,(8,.25))]:
        records=[]
        for name,(cfg,flows,size) in generalization_cases().items():
            for seed in plan['test_seeds']:
                result,score,_=episode(replace(cfg,feedback_delay_budget=.25),seed=seed,flows=flows,payload_bytes=size,
                                       model=policy,baseline=action,receipt_feedback=True)
                records.append(dict(scenario=name,seed=seed,completed=sum(r.completed for r in result),
                    receiver_complete=sum(r.receiver_complete for r in result),transfers=len(result),wrong=0,
                    elapsed=float(np.mean([r.elapsed_seconds for r in result])),
                    feedback_bytes=sum(r.feedback_wire_bytes for r in result),data_bytes=sum(r.data_wire_bytes for r in result),
                    retries=sum(r.retransmitted_payload_bytes for r in result),score=float(score.mean())))
        final[label]=dict(summary=summary(records),records=records)
        print(label,final[label]['summary'],flush=True)
    (args.output/'results.json').write_text(json.dumps(dict(plan=plan,final_test=final,
        comparison=paired_comparison(final['selected']['records'],final['best-constant']['records'])),indent=2)+'\n')


if __name__=='__main__':main()
