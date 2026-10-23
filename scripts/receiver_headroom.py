"""Measure receiver action-space headroom before increasing training budgets.

Select constant controls globally/per scenario on validation, then freeze both
before testing. Per-scenario selection is an oracle diagnostic: endpoint policies
cannot see a scenario label. This does not train or select on test outcomes.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch

from trace_transport.receiver_experiment import scenarios,summary,paired_comparison
from trace_transport.receiver_learning import episode
from trace_transport.network_models import wifi_cases
from trace_transport.simulator import NetworkConfig

from trace_transport.receiver_discrete import ACTIONS


def cases():
    result={name:(cfg,32768) for name,cfg in scenarios().items()}
    result.update({c['name']:(NetworkConfig(**c['network']),c['message_bytes']) for c in wifi_cases()})
    return result


def run_case(name,action,seeds):
    cfg,size=cases()[name]
    rows=[]
    for seed in seeds:
        result,score,_=episode(cfg,seed=seed,payload_bytes=size,baseline=action,receipt_feedback=True)
        rows.append(dict(scenario=name,seed=seed,completed=sum(r.completed for r in result),
            transfers=len(result),wrong=0,elapsed=float(np.mean([r.elapsed_seconds for r in result])),
            feedback_bytes=sum(r.feedback_wire_bytes for r in result),data_bytes=sum(r.data_wire_bytes for r in result),
            retries=sum(r.retransmitted_payload_bytes for r in result),score=float(score.mean())))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials',type=int,default=5)
    p.add_argument('--test-trials',type=int,default=20)
    p.add_argument('--output',type=Path,default=Path('runs/receiver-headroom'))
    args=p.parse_args()
    if not 1<=args.trials<=1000 or not 1<=args.test_trials<=1000:
        p.error('trials must lie in [1,1000]')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    plan=dict(actions=ACTIONS,validation_seeds=list(range(6000,6000+args.trials)),
              test_seeds=list(range(7000,7000+args.test_trials)),
              cases={k:dict(network=asdict(cfg),payload_bytes=size) for k,(cfg,size) in cases().items()},
              selection='completed transfers first, then mean transport-v2 return; validation only',
              oracle='per-scenario constant control, uses hidden scenario label; not deployable')
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    validation={}
    for name in cases():
        validation[name]=[run_case(name,a,plan['validation_seeds']) for a in ACTIONS]
        print('swept',name,flush=True)
    rank=lambda rows:(summary(rows)['completed'],summary(rows)['mean_score'])
    per_case={name:max(range(len(ACTIONS)),key=lambda i:rank(rows[i])) for name,rows in validation.items()}
    global_index=max(range(len(ACTIONS)),key=lambda i:rank([r for rows in validation.values() for r in rows[i]]))
    selected=dict(global_index=global_index,global_action=ACTIONS[global_index],
                  per_case_indices=per_case,per_case_actions={k:ACTIONS[i] for k,i in per_case.items()})
    (args.output/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    final={label:[] for label in ['global','scenario_oracle','old_baseline']}
    for name in cases():
        for label,action in [('global',ACTIONS[global_index]),('scenario_oracle',ACTIONS[per_case[name]]),('old_baseline',(8,.025))]:
            final[label].extend(run_case(name,action,plan['test_seeds']))
    report=dict(plan=plan,selection=selected,validation=validation,final_test=final,
                summaries={k:summary(v) for k,v in final.items()},
                comparison=paired_comparison(final['scenario_oracle'],final['global']))
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(selection=selected,summaries=report['summaries'],comparison=report['comparison']),indent=2),flush=True)


if __name__=='__main__':
    main()
