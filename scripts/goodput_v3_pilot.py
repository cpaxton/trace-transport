"""Joint v3 retraining pilot, receiver telemetry and validation-only role comparison."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path

from scripts.goodput_study import write
from scripts.goodput_distribution_audit import CONTROLS
from trace_transport.goodput_distribution import DOMAIN,FAMILIES,sample_case
from trace_transport.goodput_evaluation import summarize,compare
from trace_transport.joint_evaluation import evaluate
from trace_transport.joint_models import JointModelConfig
from trace_transport.joint_training import JointTrainConfig,train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/goodput-v3-pilot'))
    p.add_argument('--decisions',type=int,default=300000)
    args=p.parse_args();out=args.output
    audit=json.loads(Path('runs/distribution-v3-audit/results.json').read_text())
    if 'completed_utc' not in audit:raise ValueError('distribution audit must finish first')
    grouped={name:[r for f in audit['families'].values() for r in f[name]['records']] for name in CONTROLS}
    minimum=min(summarize(r)['interactive_miss_rate'] for r in grouped.values())
    feasible=[n for n,r in grouped.items() if summarize(r)['interactive_miss_rate']<=minimum+.005]
    name=max(feasible,key=lambda n:summarize(grouped[n])['verified_goodput'])
    baseline=dict(CONTROLS[name],initial_rate=512000.)
    seeds={f:[] for f in FAMILIES}
    for seed in range(93000,94000):
        family=sample_case(seed)[2]['family']
        if len(seeds[family])<2:seeds[family].append(seed)
    bank=sorted(s for rows in seeds.values() for s in rows)
    cfg=JointTrainConfig(profile='goodput-v3',teacher_scheduler=baseline['scheduler'],teacher_rate=baseline['rate'],teacher_initial_rate=512000.,
        teacher_batch=baseline['batch'],teacher_delay=baseline['delay'],receiver_teacher_prior=.5,
        paired_teacher=True,decisions=args.decisions,environments=2,rollout=64,epochs=3,seed=42,
        checkpoint_every=100000,production_seconds=12.,drain_seconds=3.,easy_fraction=.1,gae_lambda=.98,device='cpu')
    model=JointModelConfig(history=32,width=32,layers=1)
    plan=dict(version='goodput-v3-pilot-v1',started_utc=datetime.now(timezone.utc).isoformat(),domain=DOMAIN,
        training=asdict(cfg),model=asdict(model),reference=name,baseline=baseline,validation_seeds=bank,
        selection='initial and final fixed before validation; diagnostic pilot, no reserved test seeds',
        reserved_test_seed_range=[95000,95999],reserved_ood_seed_range=[96000,96999])
    write(out/'plan.json',plan)
    final=train(out/'seed42',cfg,model)
    counts=[0]*22;weighted=[0.]*22
    for line in (out/'seed42/training.jsonl').read_text().splitlines():
        rx=json.loads(line)['receiver']
        for i,(n,value) in enumerate(zip(rx.get('choice_counts',[]),rx.get('choice_mean_normalized_advantage',[]))):
            counts[i]+=n
            if value is not None:weighted[i]+=n*value
    write(out/'receiver-telemetry.json',dict(scope='sampled active receiver training actions, not validation performance',
        choices=counts,mean_normalized_advantage=[v/n if n else None for v,n in zip(weighted,counts)]))
    result=dict(plan=plan,modes={})
    for mode,sender,receiver,checkpoint in [('adaptive',False,False,None),('initial',True,True,out/'seed42/initial.pt'),
                                          ('sender-only',True,False,final),('receiver-only',False,True,final),('joint',True,True,final)]:
        partial=[]
        def record(row):
            partial.append(row);write(out/'progress.json',dict(phase='validation',mode=mode,completed=len(partial),total=len(bank)))
        rows=evaluate(seeds=bank,checkpoint=checkpoint,baseline=baseline,sender_enabled=sender,
            receiver_enabled=receiver,environments=2,production=12.,drain=3.,profile='goodput-v3',on_record=record)
        result['modes'][mode]=dict(summary=summarize(rows),records=rows,
            comparison=compare(rows,result['modes']['adaptive']['records']) if mode!='adaptive' else None)
        write(out/'validation.json',result)
    result['completed_utc']=datetime.now(timezone.utc).isoformat();write(out/'validation.json',result)
    write(out/'progress.json',dict(phase='complete',reserved_tests='unopened'))


if __name__=='__main__':main()
