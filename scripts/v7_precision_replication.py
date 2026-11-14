"""Fresh-future replication on all existing test snapshots; no predictor training."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import torch
from trace_transport.experiment_io import write
from trace_transport.reactive_sender import MacroWorld,FEATURES
from scripts.v7_local_predictability import (Predictor,teacher_actions,continuation,
    rows,tensors,bank_values,case_mean,summarize_choices)
from scripts.v7_policy_diagnostics import seed_value


def identity(state):
    return {k:state[k] for k in ['seed','snapshot','producer','seconds','history','valid']}


def fingerprint(state):
    return hashlib.sha256(json.dumps(identity(state),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate(config,profile):
    spec=config['profiles'][profile]
    if not spec['cases'] or len(spec['cases'])!=len(set(spec['cases'])) or any(type(s) is not int or s<0 for s in spec['cases']):raise ValueError('unique cases required')
    if not 1<=config['workers']<=4 or not 2<=spec['replicas']<=64 or spec['replica_offset']<8:raise ValueError('invalid bounded fresh futures')
    engineer=config['profiles']['engineering'];pilot=config['profiles']['pilot']
    if max(engineer['replica_offset'],pilot['replica_offset'])<min(engineer['replica_offset']+engineer['replicas'],pilot['replica_offset']+pilot['replicas']):raise ValueError('engineering/pilot future overlap')
    if config['model_seeds']!=[1,2,3] or set(config['frozen_model_sha256'])!={f'{k}-{s}' for k in ['current','history','transformer'] for s in [1,2,3]}:raise ValueError('all frozen models required')
    return spec


def check_reference(config,data,result,source):
    if hashlib.sha256(json.dumps(data,indent=2).encode()+b'\n').hexdigest()!=config['reference_data_sha256']:raise ValueError('reference dataset mismatch')
    reference_cases=sorted(c['seed'] for c in data['test'])
    if sorted(config['profiles']['pilot']['cases'])!=reference_cases or not set(config['profiles']['engineering']['cases'])<=set(reference_cases):raise ValueError('replicate all prior test cases')
    if result['constant_action']!=config['constant_action']:raise ValueError('frozen constant mismatch')
    expected={k:v for k,v in result['plan']['source'].items() if k.startswith('src/') or k in ['scripts/v7_local_predictability.py','scripts/v7_policy_diagnostics.py']}
    if any(source.get(k)!=v for k,v in expected.items()):raise ValueError('frozen simulator/predictor source mismatch')
    if len(rows(data['test']))!=24:raise ValueError('24 complete reference snapshots required')


def collect_case(task):
    torch.set_num_threads(1);seed=task['seed'];world=MacroWorld(seed,production=task.get('production',12.),drain=task.get('drain',3.));states=[]
    for expected,after in zip(task['states'],task['times']):
        j=expected['snapshot'];deadline=after+.75
        while not world.env.done() and world.env.now<deadline:
            o=world.observation();eligible=np.flatnonzero(o['eligible'])
            if world.env.now+1e-9>=after and len(eligible):break
            world.step(teacher_actions(world))
        else:raise ValueError('previous snapshot no longer reachable')
        index=int(eligible[j%len(eligible)])
        actual=dict(seed=seed,snapshot=j,producer=index,seconds=world.env.now,history=o['history'][index].tolist(),valid=o['valid'][index].tolist())
        if fingerprint(actual)!=fingerprint(expected):raise ValueError('snapshot identity changed')
        row=dict(**actual,status='complete',banks={},reference_snapshot_sha256=fingerprint(expected))
        for bank in ['fit','score']:
            b=0 if bank=='fit' else 1
            row['banks'][bank]=[]
            for a in range(3):
                arm=[];row['banks'][bank].append(arm)
                for r in range(task['replica_offset'],task['replica_offset']+task['replicas']):
                    arm.append(continuation(world,index,a,seed_value(seed,j,b,a,r,0x5052)))
                    if task.get('output'):
                        done=((j*2+b)*3+a)*task['replicas']+len(arm)
                        write(Path(task['output'])/f'case-{seed}-progress.json',dict(seed=seed,phase='futures',
                            completed=done,total=len(task['states'])*6*task['replicas'],updated_utc=datetime.now(timezone.utc).isoformat()))
                        if len(arm)%8==0:
                            write(Path(task['output'])/f'case-{seed}.json',dict(seed=seed,states=states+[row],complete=False))
        states.append(row)
        if task.get('output'):write(Path(task['output'])/f'case-{seed}.json',dict(seed=seed,states=states,complete=False))
    result=dict(seed=seed,states=states,complete=True)
    if task.get('output'):
        write(Path(task['output'])/f'case-{seed}.json',result)
        total=len(task['states'])*6*task['replicas']
        write(Path(task['output'])/f'case-{seed}-progress.json',dict(seed=seed,phase='complete',completed=total,total=total,updated_utc=datetime.now(timezone.utc).isoformat()))
    return result


def fixed_choices(reference_root,states,model_seeds):
    x,v=tensors(states);selections={}
    for kind in ['current','history','transformer']:
        for seed in model_seeds:
            tag=f'{kind}-{seed}';ck=torch.load(reference_root/(tag+'.pt'),map_location='cpu',weights_only=True)
            if ck['kind']!=kind or ck['features']!=list(FEATURES):raise ValueError('predictor schema mismatch')
            model=Predictor(kind);model.load_state_dict(ck['weights']);model.eval();norm=ck['normalization']
            with torch.inference_mode():prediction=model((x-norm['mean'])/norm['scale'],v)
            if not torch.isfinite(prediction).all():raise ValueError('nonfinite frozen predictor')
            selections[tag]=prediction.argmax(-1).tolist()
    return selections


def report(cases,reference_root,config):
    states=rows(cases);constant=config['constant_action']
    selections=dict(teacher=[1]*len(states),**{'train-selected-constant':[constant]*len(states),
        'snapshot-fit-table':bank_values(states,'fit').argmax(1).tolist()})
    selections.update(fixed_choices(reference_root,states,config['model_seeds']))
    result=dict(accepted_win=False,modes={},reference_snapshot_sha256=[s['reference_snapshot_sha256'] for s in states])
    for tag,choices in selections.items():
        entry=summarize_choices(states,choices)
        entry['comparison_to_train_constant']=summarize_choices(states,choices,reference_choices=selections['train-selected-constant'],reference_label='train_constant')
        if tag.startswith(('history-','transformer-')):
            entry['comparison_to_current']=summarize_choices(states,choices,reference_choices=selections['current-'+tag.rsplit('-',1)[1]],reference_label='current')
        result['modes'][tag]=entry
    fit=bank_values(states,'fit');score=bank_values(states,'score')
    result['fit_score_action_agreement']=float(np.mean(fit.argmax(1)==score.argmax(1)))
    result['optimistic_same_score_bank_max']=case_mean(states,score.max(1)-score[:,1])
    result['per_snapshot']=[dict(seed=s['seed'],snapshot=s['snapshot'],fit_means=f.tolist(),score_means=q.tolist(),
        score_return_standard_errors=[float(np.std([r['return_value'] for r in arm],ddof=1)/np.sqrt(len(arm))) for arm in s['banks']['score']]) for s,f,q in zip(states,fit,score)]
    result['scope']='fresh future-noise replication on the same diagnostic test states; all predictors and constant frozen; not a new case holdout, policy rollout, or oracle upper bound; no multiple-comparison adjustment'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--profile',choices=['engineering','pilot'],default='engineering');args=p.parse_args()
    config=json.loads(args.config.read_text());spec=validate(config,args.profile);out=args.output
    if out.exists() and any(out.iterdir()):raise ValueError('fresh output required')
    torch.set_num_threads(1);source=json.loads(Path('source-manifest.json').read_text())
    for path,digest in source.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('source manifest mismatch')
    old=json.loads((args.reference_root/'data.json').read_text());old_result=json.loads((args.reference_root/'results.json').read_text())
    check_reference(config,old,old_result,source)
    if hashlib.sha256((args.reference_root/'results.json').read_bytes()).hexdigest()!=config['reference_result_sha256']:raise ValueError('reference result mismatch')
    hashes={f'{kind}-{seed}':hashlib.sha256((args.reference_root/f'{kind}-{seed}.pt').read_bytes()).hexdigest() for kind in ['current','history','transformer'] for seed in config['model_seeds']}
    if hashes!=config['frozen_model_sha256']:raise ValueError('frozen predictor weights changed')
    plan=dict(version='v7-precision-replication-v1',started_utc=datetime.now(timezone.utc).isoformat(),config=config,profile=args.profile,source=source,
        frozen_model_sha256=hashes,reference_result_sha256=config['reference_result_sha256'],reference_data_sha256=config['reference_data_sha256'],runtime=dict(torch=str(torch.__version__),numpy=str(np.__version__)),accepted_win=False)
    write(out/'plan.json',plan)
    by_seed={c['seed']:c for c in old['test']};tasks=[dict(seed=s,states=by_seed[s]['states'],times=old_result['plan']['config']['snapshot_after_seconds'],
        replicas=spec['replicas'],replica_offset=spec['replica_offset'],output=str(out/'cases')) for s in spec['cases']]
    from scripts.v7_precision_replication import collect_case as execute
    try:
        cases=[]
        with ProcessPoolExecutor(max_workers=config['workers'],mp_context=get_context('spawn')) as pool:
            pending={pool.submit(execute,t):t for t in tasks}
            for f in as_completed(pending):
                cases.append(f.result());write(out/'progress.json',dict(phase='collection',completed=len(cases),total=len(tasks)))
        cases.sort(key=lambda c:c['seed'])
        if [c['seed'] for c in cases]!=sorted(spec['cases']):raise ValueError('incomplete replication')
        ids=[r['replica_seed'] for c in cases for s in c['states'] for bank in s['banks'].values() for arm in bank for r in arm]
        old_ids={r['replica_seed'] for c in old['test'] for s in c['states'] for bank in s['banks'].values() for arm in bank for r in arm}
        if len(ids)!=len(set(ids)) or set(ids)&old_ids:raise ValueError('future seed reuse')
        write(out/'data.json',cases);result=report(cases,args.reference_root,config)
        result.update(plan=plan,completed_utc=datetime.now(timezone.utc).isoformat(),terminal_continuations=len(ids),all_futures_new=True)
        write(out/'results.json',result);write(out/'progress.json',dict(phase='complete',accepted_win=False))
    except BaseException as exc:
        write(out/'failure.json',dict(error=repr(exc),utc=datetime.now(timezone.utc).isoformat()));raise


if __name__=='__main__':main()
