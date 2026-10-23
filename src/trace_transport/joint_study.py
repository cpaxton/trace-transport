"""Calibration, validation-only paired selection, frozen tests and role ablations."""
import json
from pathlib import Path
import shutil
import time

import torch

from .joint_evaluation import BASELINES,evaluate,summarize,select_baseline,compare,headroom
from .joint_models import export_actors,load_checkpoint


def _write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def _rank(rows,reference):
    a,b=summarize(rows),summarize(reference)
    feasible=(a['image_goodput']>=.95*b['image_goodput'] and a['map_goodput']>=.95*b['map_goodput'] and
              a['image_bytes']>=.95*b['image_bytes'] and a['map_bytes']>=.95*b['map_bytes'] and
              a['wire_per_delivered_byte']<=1.05*b['wire_per_delivered_byte'] and
              a['completed']/max(1,a['admitted'])>=b['completed']/max(1,b['admitted']))
    return feasible,-a['interactive_miss_rate'],-a['wire_per_delivered_byte']


def _ready(root,directories,progress,output,wait):
    for directory in directories:
        while not (root/directory/'final.pt').exists():
            progress.update(phase='waiting-for-training',run=directory);_write(output/'progress.json',progress)
            if not wait:raise RuntimeError(f'training incomplete: {directory}; rerun with --wait')
            print('waiting for',directory,flush=True);time.sleep(30)
    for directory in directories:
        _,_,model_cfg,meta=load_checkpoint(root/directory/'final.pt')
        config=meta['training']
        if meta['decisions']<config['decisions']:raise ValueError('incomplete training checkpoint')
        if directory.startswith('seed') and config['seed']!=int(directory[4:]):raise ValueError('seed/directory mismatch')
        if directory=='history1' and model_cfg.history!=1:raise ValueError('history ablation is not one-step')
        if directory=='sender-only' and config.get('learn_receiver',True):raise ValueError('receiver was not frozen')
        if directory=='receiver-only' and config.get('learn_sender',True):raise ValueError('sender was not frozen')


def run(root,*,device='cpu',environments=4,wait=False,quick=False):
    root=Path(root);output=root/'study';output.mkdir(parents=True,exist_ok=True)
    plan=dict(version='joint-study-v1',calibration_seeds=list(range(30000,30000+(2 if quick else 4))),
        screen_seeds=list(range(40000,40000+(2 if quick else 4))),
        validation_seeds=list(range(41000,41000+(2 if quick else 12))),
        test_seeds=list(range(50000,50000+(2 if quick else 40))),
        ood_seeds=list(range(60000,60000+(2 if quick else 40))),quick=quick,
        win='20% fewer desired interactive deadline misses with paired CI; bulk rate/volume >=95%, completion noninferior, wire/delivered-byte <=105%',
        selection='calibrate deterministic pairs; screen snapshots, then full validation; freeze before tests')
    if (output/'plan.json').exists() and json.loads((output/'plan.json').read_text())!=plan:
        raise ValueError('study configuration changed; use a fresh study root')
    _write(output/'plan.json',plan)
    progress=dict(phase='calibration',completed=0,total=len(BASELINES))
    _write(output/'progress.json',progress)
    calibration_path=output/'calibration.json'
    calibration=json.loads(calibration_path.read_text()) if calibration_path.exists() else {}
    configs={f"{b['rate']}/{b['scheduler']}/{b['feedback']}":b for b in BASELINES
             if not quick or (b['rate'] in ('fixed','bdp') and b['scheduler']=='edf' and b['feedback']=='budgeted')}
    for name,b in configs.items():
        if name not in calibration:
            rows=evaluate(seeds=plan['calibration_seeds'],baseline=b,environments=environments,production=2 if quick else 6,drain=2 if quick else 4)
            calibration[name]=rows;_write(calibration_path,calibration)
        progress['completed']=len(calibration);_write(output/'progress.json',progress)
        print('calibration',name,summarize(calibration[name]),flush=True)
    winner=select_baseline(calibration)
    # Include the highest-throughput and lowest-miss controls as additional contenders.
    contenders=list(dict.fromkeys([winner,
        max(calibration,key=lambda n:summarize(calibration[n])['image_goodput']+summarize(calibration[n])['map_goodput']),
        min(calibration,key=lambda n:summarize(calibration[n])['interactive_miss_rate'])]))
    full_prod,full_drain=(2.,2.) if quick else (20.,20.)
    if (output/'baseline.json').exists():
        cached=json.loads((output/'baseline.json').read_text())
        baseline_validation=cached['validation'];reference_name=cached['selected'];reference=cached['config']
    else:
        baseline_validation={}
        for name in contenders:
            baseline_validation[name]=evaluate(seeds=plan['validation_seeds'],baseline=configs[name],environments=environments,production=full_prod,drain=full_drain)
        reference_name=select_baseline(baseline_validation);reference=configs[reference_name]
        _write(output/'baseline.json',dict(selected=reference_name,config=reference,validation=baseline_validation))
    _ready(root,['seed42','seed43','seed44'],progress,output,wait)
    screen_reference=evaluate(seeds=plan['screen_seeds'],baseline=reference,environments=environments,
        production=2 if quick else 6,drain=2 if quick else 4)
    _write(output/'screen-reference.json',screen_reference)
    progress.update(phase='checkpoint-screening');_write(output/'progress.json',progress)
    screen_path=output/'screening.json';screen=json.loads(screen_path.read_text()) if screen_path.exists() else {}
    candidates=[]
    for directory in ['seed42','seed43','seed44']:
        paths=[root/directory/'initial.pt',*sorted((root/directory).glob('d*.pt')),root/directory/'final.pt']
        if quick:paths=[paths[0],paths[-1]]
        for path in paths:
            key=str(path.relative_to(root))
            if key not in screen:
                rows=evaluate(seeds=plan['screen_seeds'],checkpoint=path,device=device,environments=environments,
                    production=2 if quick else 6,drain=2 if quick else 4)
                screen[key]=rows;_write(screen_path,screen)
                print('screen',key,summarize(rows),flush=True)
        initial=str(paths[0].relative_to(root));candidates.append(initial)
        ranked=sorted([str(p.relative_to(root)) for p in paths[1:]],key=lambda key:_rank(screen[key],screen_reference),reverse=True)
        candidates.extend(ranked[:2])
    validation_path=output/'validation.json';validation=json.loads(validation_path.read_text()) if validation_path.exists() else {}
    progress.update(phase='full-validation');_write(output/'progress.json',progress)
    for key in dict.fromkeys(candidates):
        if key not in validation:
            validation[key]=evaluate(seeds=plan['validation_seeds'],checkpoint=root/key,device=device,environments=environments,production=full_prod,drain=full_drain)
            _write(validation_path,validation)
        print('validation',key,summarize(validation[key]),flush=True)
    selected=max(validation,key=lambda key:_rank(validation[key],baseline_validation[reference_name]))
    selection=dict(checkpoint=selected,baseline=reference_name,validation_summaries={k:summarize(v) for k,v in validation.items()})
    _write(output/'selection.json',selection)
    export_actors(root/selected,output/'joint-v1-actors.pt')
    result=dict(plan=plan,selection=selection,baseline=reference,final_test={},ood={})
    if (output/'results.json').exists():
        cached=json.loads((output/'results.json').read_text())
        if any(cached.get(k)!=v for k,v in [('plan',plan),('selection',selection),('baseline',reference)]):
            raise ValueError('cached held-out results disagree with frozen selection or plan')
        result=cached
    modes=[('joint',root/selected,True,True),('baseline',None,False,False),
           ('sender-crossplay',root/selected,True,False),('receiver-crossplay',root/selected,False,True),
           ('history1',root/'history1/final.pt',True,True),('sender-only-training',root/'sender-only/final.pt',True,True),
           ('receiver-only-training',root/'receiver-only/final.pt',True,True)]
    # Ablations are fixed final-budget models, not selected using test outcomes.
    for suite,seeds,ood in [('final_test',plan['test_seeds'],False),('ood',plan['ood_seeds'],True)]:
        for label,path,send,receive in modes:
            if path is not None and path.parent.name in ('history1','sender-only','receiver-only'):
                _ready(root,[path.parent.name],progress,output,wait)
            progress.update(phase=suite,run=label);_write(output/'progress.json',progress)
            if label in result[suite]:
                if [r['seed'] for r in result[suite][label]['records']]!=seeds:
                    raise ValueError('cached held-out seed list mismatch')
                continue
            rows=evaluate(seeds=seeds,checkpoint=path,baseline=reference,device=device,environments=environments,
                production=full_prod,drain=full_drain,ood=ood,sender_enabled=send,receiver_enabled=receive)
            result[suite][label]=dict(summary=summarize(rows),records=rows)
            _write(output/'results.json',result);print(suite,label,summarize(rows),flush=True)
        result[suite]['comparison']=compare(result[suite]['joint']['records'],result[suite]['baseline']['records'])
        _write(output/'results.json',result)
    progress.update(phase='complete');_write(output/'progress.json',progress)
    return result
