"""Frozen receiver-v2 generalization and protocol ablation, no model selection.

Run after both receiver-v2 preset/randomized experiments have completed.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from trace_transport.receiver_experiment import evaluate, summary, paired_comparison
from trace_transport.receiver_learning import episode, load
from trace_transport.simulator import NetworkConfig
from trace_transport.network_models import wifi_cases

BASELINES={'immediate':(1,0.),'delayed-2':(2,.01),'delayed-4':(4,.01),
           'delayed-8':(8,.01),'delayed-8-25ms':(8,.025)}


def domains():
    cases = {
        'loss-bursts':(NetworkConfig(loss=.01,ack_loss=.01,feedback_bandwidth_bps=64000,
            loss_changes=((.2,.4,.5),(.55,.01,.01),(1.2,.5,.4),(1.6,.01,.01)),deadline=25),2,32768),
        'delay-drift':(NetworkConfig(feedback_bandwidth_bps=64000,rtt_changes=((.2,.1),(.4,.2),(.8,.35),(1.5,.1),(2.,.05)),deadline=25),2,32768),
        'long-rtt':(NetworkConfig(base_rtt=.3,feedback_bandwidth_bps=64000,deadline=25),2,32768),
        'reverse-12k':(NetworkConfig(feedback_bandwidth_bps=12000,deadline=25),2,32768),
        'six-flows':(NetworkConfig(feedback_bandwidth_bps=192000,deadline=25),6,32768),
        'large-message':(NetworkConfig(bandwidth_bps=2000000,feedback_bandwidth_bps=128000,deadline=25),2,131072),
        'small-queue':(NetworkConfig(feedback_bandwidth_bps=64000,feedback_queue_bytes=2048,
                                    queue_bytes=2048,deadline=25),2,32768),
        'reverse-changes':(NetworkConfig(feedback_bandwidth_bps=32000,
             feedback_capacity_changes=((.3,512000),(.9,16000),(2.,64000)),deadline=25),2,32768),
    }
    cases.update({case['name']:(NetworkConfig(**case['network']),case['flows'],case['message_bytes'])
                  for case in wifi_cases()})
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials',type=int,default=20)
    p.add_argument('--output',type=Path,default=Path('runs/receiver-v2-generalization'))
    args=p.parse_args()
    if not 1<=args.trials<=1000:
        p.error('trials must be in [1,1000]')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    paths={name:Path(f'runs/receiver-v2-{name}/receiver-v2-best.pt') for name in ['presets','randomized','memoryless']}
    selections={name:json.loads(Path(f'runs/receiver-v2-{name}/selection.json').read_text()) for name in paths}
    assert selections['presets']['baseline']==selections['randomized']['baseline']
    baseline_name=selections['presets']['baseline']
    models={name:load(path) for name,path in paths.items()}
    plan=dict(version='receiver-v2-generalization',seed_start=5000,trials=args.trials,
              selected_on_validation={name:value['learned'] for name,value in selections.items()},
              deterministic_baseline=baseline_name,
              checkpoint_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()},
              domains={name:dict(network=asdict(cfg),flows=flows,payload_bytes=size) for name,(cfg,flows,size) in domains().items()},
              selection='None on these test domains; models and baseline frozen using prior validation only')
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    output={}
    for label in ['presets','randomized','memoryless',baseline_name,'immediate']:
        if label in output:
            continue
        records=[]
        for name,(cfg,flows,size) in domains().items():
            for seed in range(5000,5000+args.trials):
                result,score,_=episode(cfg,seed=seed,payload_bytes=size,flows=flows,
                    model=models.get(label),baseline=BASELINES.get(label),receipt_feedback=True)
                records.append(dict(scenario=name,seed=seed,completed=sum(r.completed for r in result),
                    receiver_complete=sum(r.receiver_complete for r in result),transfers=len(result),
                    wrong=sum(r.received_text is not None and r.received_text!=r.sent_text for r in result),
                    elapsed=float(np.mean([r.elapsed_seconds for r in result])),
                    feedback_bytes=sum(r.feedback_wire_bytes for r in result),
                    data_bytes=sum(r.data_wire_bytes for r in result),retries=sum(r.retransmitted_payload_bytes for r in result),
                    score=float(score.mean())))
        output[label]=dict(summary=summary(records),records=records)
        print(label,output[label]['summary'],flush=True)
    old=load('artifacts/receiver-v1-candidate.pt')
    ablation={}
    for receipts in [False,True]:
        records=evaluate(old,None,range(4000,4000+args.trials),receipt_feedback=receipts)
        ablation[str(receipts)]=dict(summary=summary(records),records=records)
        print('frozen-v1 receiver, receipts=',receipts,ablation[str(receipts)]['summary'],flush=True)
    report=dict(plan=plan,final_test=output,protocol_ablation=ablation,
        comparisons={name:paired_comparison(output[name]['records'],output[baseline_name]['records']) for name in models})
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
