"""Factor feedback delay budget from receiver controls, no learning claims.

Tune constant settings on validation, freeze them, then test fresh diagnostic
and generalization seeds. Budget is configured identically at both endpoints;
this experiment does not implement on-wire delay-budget negotiation.
"""
import argparse
from dataclasses import replace,asdict
import json
from pathlib import Path
import numpy as np
import torch

from trace_transport.receiver_experiment import scenarios,summary,paired_comparison
from trace_transport.receiver_learning import episode
from trace_transport.network_models import generalization_cases

ACTIONS=[(8,.025),(16,.025),(32,.025),(8,.1),(16,.1),(32,.1),(8,.25),(16,.25),(32,.25)]


def evaluate(case_map,action,seeds,budget):
    rows=[]
    for name,(cfg,flows,size) in case_map.items():
        for seed in seeds:
            result,score,_=episode(replace(cfg,feedback_delay_budget=budget),seed=seed,flows=flows,
                                   payload_bytes=size,baseline=action,receipt_feedback=True)
            rows.append(dict(scenario=name,seed=seed,completed=sum(r.completed for r in result),
                receiver_complete=sum(r.receiver_complete for r in result),transfers=len(result),wrong=0,
                elapsed=float(np.mean([r.elapsed_seconds for r in result])),
                feedback_bytes=sum(r.feedback_wire_bytes for r in result),data_bytes=sum(r.data_wire_bytes for r in result),
                retries=sum(r.retransmitted_payload_bytes for r in result),score=float(score.mean())))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation-trials',type=int,default=5)
    p.add_argument('--trials',type=int,default=20)
    p.add_argument('--output',type=Path,default=Path('runs/feedback-budget-v1'))
    args=p.parse_args()
    if not 1<=args.validation_trials<=1000 or not 1<=args.trials<=1000:p.error('invalid budgets')
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    diagnostic={name:(cfg,2,32768) for name,cfg in scenarios().items()}
    plan=dict(version='feedback-budget-v1',actions=ACTIONS,budget=.25,
        validation_seeds=list(range(17000,17000+args.validation_trials)),
        test_seeds=list(range(18000,18000+args.trials)),generalization_seeds=list(range(19000,19000+args.trials)),
        selection='global constant action: validation completion first, then transport-v2 return',
        domains={name:dict(network=asdict(cfg),flows=flows,payload_bytes=size) for name,(cfg,flows,size) in diagnostic.items()},
        protocol='same configured .25s delay budget at both endpoints; not negotiated')
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    validation=[]
    for action in ACTIONS:
        rows=evaluate(diagnostic,action,plan['validation_seeds'],.25)
        validation.append(dict(action=action,summary=summary(rows),records=rows))
        print('validation',action,summary(rows),flush=True)
    selected=max(validation,key=lambda r:(r['summary']['completed'],r['summary']['mean_score']))['action']
    (args.output/'selection.json').write_text(json.dumps(dict(action=selected,validation=validation),indent=2)+'\n')
    final={}
    for suite,case_map,seeds in [('diagnostic',diagnostic,plan['test_seeds']),('generalization',generalization_cases(),plan['generalization_seeds'])]:
        final[suite]={}
        for label,action,budget in [('old',(8,.025),0.),('timers-only',(8,.025),.25),('wide-control',selected,.25)]:
            rows=evaluate(case_map,action,seeds,budget)
            final[suite][label]=dict(summary=summary(rows),records=rows)
            print(suite,label,summary(rows),flush=True)
    report=dict(plan=plan,selected=selected,validation=validation,final_test=final,
        comparisons={suite:paired_comparison(rows['wide-control']['records'],rows['old']['records']) for suite,rows in final.items()})
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
