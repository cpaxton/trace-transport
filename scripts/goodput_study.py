"""Versioned goodput-v2 calibration, validation and frozen tests. Never auto-merges."""
import argparse
import hashlib
import json
from pathlib import Path

from trace_transport.goodput import DOMAIN
from trace_transport.goodput_evaluation import summarize,compare
from trace_transport.joint_evaluation import evaluate
from trace_transport.joint_models import load_checkpoint,export_actors


from trace_transport.experiment_io import write


def cases(root,key,seeds,*,baseline,checkpoint=None,device='cpu',ood=False):
    path=root/'cases'/f'{key}.json';identity=dict(seeds=seeds,baseline=baseline,ood=ood,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,
        domain=DOMAIN,production=6.,drain=2.)
    saved={}
    if path.exists():
        data=json.loads(path.read_text())
        if data['identity']!=identity:raise ValueError('cached case identity changed')
        saved={r['seed']:r for r in data['records']}
        if len(saved)!=len(data['records']) or set(saved)-set(seeds):raise ValueError('invalid cached seeds')
    def record(row):
        saved[row['seed']]=row;write(path,dict(identity=identity,records=[saved[s] for s in sorted(saved)]))
        write(root/'progress.json',dict(phase=key,completed=len(saved),total=len(seeds)))
    missing=[seed for seed in seeds if seed not in saved]
    if missing:evaluate(seeds=missing,checkpoint=checkpoint,baseline=baseline,device=device,environments=2,
        production=6.,drain=2.,ood=ood,profile='goodput-v2',on_record=record)
    return [saved[s] for s in seeds]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['calibrate','validate','test'])
    p.add_argument('--root',type=Path,default=Path('runs/goodput-v21'));p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',default='cpu');args=p.parse_args();root=args.root
    package=Path(__file__).resolve().parents[1]/'src/trace_transport'
    source_hashes={name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in ['goodput.py','goodput_evaluation.py','joint_session.py','joint_models.py','joint_workers.py','joint_evaluation.py','wire.py','protocol.py','session_receiver.py']}
    plan=dict(version='goodput-study-v2.1',domain=DOMAIN,source_sha256=source_hashes,calibration=list(range(71000,71008)),
        pilot_validation=list(range(72000,72008)),validation=list(range(73000,73016)),
        test=list(range(74000,74040)),ood=list(range(75000,75040)),
        criteria='10% verified goodput gain and positive paired CI; miss CI upper <=0.005; wire <=1.05; no corruption/starvation; class/confirmation/completion guards',
        baseline_search='24 initial controls then all 21 receiver actions for three distinct pacing configurations',
        resources='one isolated service, CPU quota 400%, RAM max 8GiB, 2 workers; CPU pilot')
    if (root/'plan.json').exists() and json.loads((root/'plan.json').read_text())!=plan:raise ValueError('study plan changed')
    write(root/'plan.json',plan)
    if args.stage=='calibrate':
        records={};configs={}
        for rate in ['fixed','aimd','bdp','delay']:
            for start in [128000.,512000.,2000000.]:
                for feedback in ['budgeted','sparse']:
                    name=f'{rate}-{int(start)}-{feedback}';baseline=dict(rate=rate,initial_rate=start,scheduler='edf',feedback=feedback)
                    configs[name]=baseline;records[name]=cases(root,'cal-'+name,plan['calibration'],baseline=baseline)
                    write(root/'calibration.json',dict(configs=configs,records=records))
        # Prefer latency noninferiority, then the highest useful throughput among
        # eligible controls. This calibration never reads validation/test data.
        # Hierarchical calibration: retain three distinct pacing configurations,
        # then test every constant receiver action, including those available to ML.
        minimum=min(summarize(rows)['interactive_miss_rate'] for rows in records.values())
        ranked=sorted(records,key=lambda name:(summarize(records[name])['interactive_miss_rate']>minimum+.005,-summarize(records[name])['verified_goodput']))
        pacing=[]
        for name in ranked:
            pair=(configs[name]['rate'],configs[name]['initial_rate'])
            if pair not in pacing:pacing.append(pair)
            if len(pacing)==3:break
        from trace_transport.joint_session import ACK_ACTIONS
        for rate,start in pacing:
            for i,(batch,delay) in enumerate(ACK_ACTIONS):
                name=f'{rate}-{int(start)}-ack{i}';baseline=dict(rate=rate,initial_rate=start,scheduler='edf',feedback='budgeted',batch=batch,delay=delay)
                configs[name]=baseline;records[name]=cases(root,'cal-'+name,plan['calibration'],baseline=baseline)
                write(root/'calibration.json',dict(configs=configs,records=records))
        minimum=min(summarize(rows)['interactive_miss_rate'] for rows in records.values())
        feasible=[name for name,rows in records.items() if summarize(rows)['interactive_miss_rate']<=minimum+.005]
        selected=max(feasible,key=lambda name:summarize(records[name])['verified_goodput'])
        baseline=configs[selected]
        # Freeze a full-validation reference before any trained policy is tested.
        ref=cases(root,'reference-validation',plan['validation'],baseline=baseline)
        write(root/'reference.json',dict(name=selected,baseline=baseline,validation=ref))
        write(root/'progress.json',dict(phase='calibration-complete',reference=selected));return
    reference=json.loads((root/'reference.json').read_text());baseline=reference['baseline']
    if args.checkpoint is None:p.error('--checkpoint required')
    _,_,_,metadata=load_checkpoint(args.checkpoint)
    if metadata.get('profile')!='goodput-v2':raise ValueError('expected goodput-v2 checkpoint')
    if args.stage=='validate':
        key='validation-'+hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()[:16]
        rows=cases(root,key,plan['validation'],baseline=baseline,checkpoint=args.checkpoint,device=args.device)
        result=dict(checkpoint=str(args.checkpoint),summary=summarize(rows),comparison=compare(rows,reference['validation']),records=rows)
        write(root/(key+'.json'),result);print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2));return
    # Test selection is frozen once; changing a checkpoint requires a new study.
    freeze=dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),baseline=baseline)
    if (root/'frozen.json').exists() and json.loads((root/'frozen.json').read_text())!=freeze:raise ValueError('test selection already frozen')
    write(root/'frozen.json',freeze);export_actors(args.checkpoint,root/'selected-actors.pt')
    results=dict(plan=plan,selection=freeze,training_decisions=metadata.get('decisions',0),suites={})
    for suite,ood in [('test',False),('ood',True)]:
        ref=cases(root,suite+'-reference',plan[suite],baseline=baseline,ood=ood)
        rows=cases(root,suite+'-selected',plan[suite],baseline=baseline,checkpoint=args.checkpoint,device=args.device,ood=ood)
        results['suites'][suite]=dict(selected=summarize(rows),reference=summarize(ref),comparison=compare(rows,ref),records=rows,reference_records=ref)
        write(root/'results.json',results)
    write(root/'progress.json',dict(phase='complete'))


if __name__=='__main__':main()
