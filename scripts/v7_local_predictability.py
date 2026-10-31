"""Case-split local-action predictability, with independent fitting/scoring futures.

No RL, actor checkpoint selection, or transport acceptance. Only local histories
enter predictors. Full-state clones and future returns are diagnostic labels.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from trace_transport.experiment_io import write
from trace_transport.reactive_sender import MacroWorld,BIASES,FEATURES,KEEP
from scripts.v7_policy_diagnostics import seed_value,GAMMA,REFERENCE_SECONDS

KINDS=('current','history','transformer')

def validate(config,profile):
    spec=config['profiles'][profile];splits=spec['splits']
    all_seeds=[]
    for split in ['train','test']:
        seeds=splits[split]
        if not seeds or len(seeds)!=len(set(seeds)) or any(type(s) is not int or s<0 for s in seeds):raise ValueError('invalid case split')
        all_seeds+=seeds
    if len(all_seeds)!=len(set(all_seeds)):raise ValueError('case leakage between splits')
    if spec['replicas']<2 or not 1<=config['workers']<=4:raise ValueError('invalid resources/replicas')
    times=config['snapshot_after_seconds']
    if not times or times!=sorted(times) or len(times)!=len(set(times)) or any(not np.isfinite(t) or not 0<=t<10 for t in times):raise ValueError('invalid snapshot times')
    if sorted(config['model_seeds'])!=[1,2,3] or not 1<=config['fit_steps']<=1000:raise ValueError('fixed model coverage required')
    return spec


def teacher_actions(world):
    o=world.observation();a=np.full(world.env.count,KEEP,int);a[o['eligible']]=1
    return a


def continuation(snapshot,index,choice,seed):
    if not snapshot.observation()['eligible'][index]:raise ValueError('ineligible producer')
    trial=snapshot.clone();trial.env.streams=[random.Random(seed_value(seed,d,0x4E4554)) for d in range(2)]
    total=0.;discount=1.;first=True
    before=dict(verified=sum(trial.env.verified[c] for c in ['image','map']),miss=trial.env.reward_counts['missed'],wire=trial.env.reward_counts['wire'])
    while not trial.env.done():
        a=teacher_actions(trial)
        if first:a[index]=choice;first=False
        _,reward,_,_,dt,_,_=trial.step(a);total+=discount*reward;discount*=GAMMA**(dt/REFERENCE_SECONDS)
    return dict(replica_seed=seed,return_value=total,verified=sum(trial.env.verified[c] for c in ['image','map'])-before['verified'],
        misses=trial.env.reward_counts['missed']-before['miss'],wire=trial.env.reward_counts['wire']-before['wire'])


def collect_case(task):
    torch.set_num_threads(1)
    seed=task['seed'];world=MacroWorld(seed,production=task.get('production',12.),drain=task.get('drain',3.));states=[]
    for j,after in enumerate(task['times']):
        deadline=after+.75
        while not world.env.done() and world.env.now<deadline:
            o=world.observation();eligible=np.flatnonzero(o['eligible'])
            if world.env.now+1e-9>=after and len(eligible):break
            world.step(teacher_actions(world))
        else:
            states.append(dict(seed=seed,snapshot=j,status='skipped',reason='no eligible producer before deadline'));continue
        index=int(eligible[j%len(eligible)]);h=o['history'][index];valid=o['valid'][index]
        row=dict(seed=seed,snapshot=j,status='complete',producer=index,seconds=world.env.now,history=h.tolist(),valid=valid.tolist(),banks={})
        for bank in ['fit','score']:
            bank_id=0 if bank=='fit' else 1
            row['banks'][bank]=[[continuation(world,index,a,seed_value(seed,j,bank_id,a,r,0x5052)) for r in range(task['replicas'])] for a in range(3)]
        states.append(row)
        if task.get('output'):write(Path(task['output'])/f'case-{seed}.json',dict(seed=seed,states=states,complete=False))
    out=dict(seed=seed,states=states,complete=True)
    if task.get('output'):write(Path(task['output'])/f'case-{seed}.json',out)
    return out


def rows(cases):return [s for c in cases for s in c['states'] if s['status']=='complete']

def tensors(states):
    x=torch.tensor(np.asarray([s['history'] for s in states]),dtype=torch.float32)
    v=torch.tensor(np.asarray([s['valid'] for s in states]),dtype=torch.bool)
    if x.shape!=(len(states),64,len(FEATURES)) or v.shape!=(len(states),64) or not v[:,-1].all() or not torch.isfinite(x).all():raise ValueError('invalid local tensors')
    return x,v


def bank_values(states,bank,key='return_value'):
    values=np.asarray([[np.mean([r[key] for r in arm]) for arm in s['banks'][bank]] for s in states])
    if values.shape!=(len(states),3) or not np.isfinite(values).all():raise ValueError('invalid action table')
    return values


class Predictor(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;n=len(FEATURES)
        if kind=='current':self.encoder=nn.Sequential(nn.Linear(n,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh());width=32
        elif kind=='history':self.encoder=nn.Sequential(nn.Linear(64*(n+1),32),nn.Tanh(),nn.Linear(32,32),nn.Tanh());width=32
        elif kind=='transformer':
            self.input=nn.Linear(n,32);self.position=nn.Parameter(torch.zeros(1,64,32))
            self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(32,4,64,dropout=0.,batch_first=True),1,enable_nested_tensor=False);width=32
        else:raise ValueError('unknown predictor')
        self.head=nn.Linear(width,2)
    def forward(self,x,v):
        x=x.masked_fill(~v[:,:,None],0.)
        if self.kind=='current':z=self.encoder(x[:,-1])
        elif self.kind=='history':z=self.encoder(torch.cat([x,v[:,:,None].float()],-1).flatten(1))
        else:z=self.encoder(self.input(x)+self.position,src_key_padding_mask=~v)[:,-1]
        out=self.head(z);return torch.stack([out[:,0],torch.zeros_like(out[:,0]),out[:,1]],-1)


def fit_choices(train_states,test_states,kind,seed,steps):
    # Score futures are deliberately never read by this function.
    torch.set_num_threads(1);torch.manual_seed(seed)
    x,v=tensors(train_states);test,tv=tensors(test_states)
    valid=x[v];mean=valid.mean(0);scale=valid.std(0,unbiased=False).clamp_min(.05)
    x=(x-mean)/scale;test=(test-mean)/scale
    q=bank_values(train_states,'fit');target=torch.tensor(q-q[:,1,None],dtype=torch.float32)
    norm=target[:,[0,2]].square().mean().sqrt().clamp_min(1e-4);target/=norm
    counts={s['seed']:sum(t['seed']==s['seed'] for t in train_states) for s in train_states}
    weights=torch.tensor([1/counts[s['seed']] for s in train_states]);weights/=weights.sum()
    model=Predictor(kind);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    model.train()
    for _ in range(steps):
        error=(model(x,v)[:,[0,2]]-target[:,[0,2]]).square().mean(1)
        loss=(error*weights).sum();optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    model.eval()
    with torch.inference_mode():
        tr=model(x,v);te=model(test,tv)
    return dict(train_choices=tr.argmax(-1).tolist(),test_choices=te.argmax(-1).tolist(),
        training_normalized_mse=float(((tr[:,[0,2]]-target[:,[0,2]])**2).mean()),
        weights=model.state_dict(),normalization=dict(mean=mean,scale=scale,target_scale=norm))


def case_mean(states,values):
    return float(np.mean([np.mean([v for s,v in zip(states,values) if s['seed']==seed]) for seed in sorted({s['seed'] for s in states})]))


def summarize_choices(states,choices,*,reference_choices=None,reference_label='teacher',bootstrap_seed=871,draws=2000):
    if len(choices)!=len(states) or any(a not in [0,1,2] for a in choices):raise ValueError('invalid choices')
    reference_choices=[1]*len(states) if reference_choices is None else reference_choices
    if len(reference_choices)!=len(states) or any(a not in [0,1,2] for a in reference_choices):raise ValueError('invalid reference choices')
    metrics={}
    for key in ['return_value','verified','misses','wire']:
        q=bank_values(states,'score',key);delta=q[np.arange(len(states)),choices]-q[np.arange(len(states)),reference_choices]
        metrics[key+'_difference_from_'+reference_label]=case_mean(states,delta)
    # Hierarchical bootstrap: cases, then independent future replicas within each action arm.
    # States in a case share the prefix and remain clustered. Teacher selections have exact zero delta.
    rng=np.random.default_rng(bootstrap_seed);cases=sorted({s['seed'] for s in states});case_draws=[]
    for seed in cases:
        differences=[]
        for s,a,b in zip(states,choices,reference_choices):
            if s['seed']!=seed:continue
            arms=s['banks']['score'];x=np.asarray([r['return_value'] for r in arms[a]]);y=np.asarray([r['return_value'] for r in arms[b]])
            differences.append(np.zeros(draws) if a==b else x[rng.integers(len(x),size=(draws,len(x)))].mean(1)-y[rng.integers(len(y),size=(draws,len(y)))].mean(1))
        case_draws.append(np.mean(differences,axis=0))
    samples=np.stack(case_draws)[rng.integers(len(cases),size=(draws,len(cases))),np.arange(draws)[:,None]].mean(1)
    metrics.update(reference=reference_label,return_difference_ci95=np.quantile(samples,[.025,.975]).tolist(),choices=np.bincount(choices,minlength=3).tolist(),cases=len(cases),snapshots=len(states),
        scope='conditional single-decision value under teacher prefix/continuation; independent future banks; case-cluster bootstrap with replica resampling; not an executed adaptive policy or acceptance')
    return metrics


def analyze(data,config,output):
    train_states=rows(data['train']);test_states=rows(data['test'])
    if {c['seed'] for c in data['train']}&{c['seed'] for c in data['test']}:raise ValueError('case leakage between splits')
    if not train_states or not test_states:raise ValueError('no complete train/test states')
    train_q=bank_values(train_states,'fit');constant=int(np.argmax([case_mean(train_states,train_q[:,a]) for a in range(3)]))
    report=dict(status='complete',accepted_win=False,constant_action=constant,skipped={k:sum(s['status']!='complete' for c in v for s in c['states']) for k,v in data.items()},modes={})
    selections={'teacher':[1]*len(test_states),'train-selected-constant':[constant]*len(test_states),
        'snapshot-fit-table':bank_values(test_states,'fit').argmax(1).tolist()}
    for kind in KINDS:
        for seed in config['model_seeds']:
            fitted=fit_choices(train_states,test_states,kind,seed,config['fit_steps'])
            tag=f'{kind}-{seed}';selections[tag]=fitted['test_choices']
            report.setdefault('training',{})[tag]=dict(normalized_mse=fitted['training_normalized_mse'],choices=fitted['train_choices'])
            torch.save(dict(kind=kind,features=list(FEATURES),weights=fitted['weights'],normalization=fitted['normalization']),output/f'{tag}.pt')
    for tag,choices in selections.items():
        report['modes'][tag]=summarize_choices(test_states,choices)
        report['modes'][tag]['comparison_to_train_constant']=summarize_choices(test_states,choices,reference_choices=selections['train-selected-constant'],reference_label='train_constant')
        if tag.startswith(('history-','transformer-')):
            current='current-'+tag.rsplit('-',1)[1]
            report['modes'][tag]['comparison_to_current']=summarize_choices(test_states,choices,reference_choices=selections[current],reference_label='current')
    q=bank_values(test_states,'score')
    report['optimistic_same_score_bank_max']=case_mean(test_states,q.max(1)-q[:,1])
    report['fit_score_action_agreement']=float(np.mean(bank_values(test_states,'fit').argmax(1)==q.argmax(1)))
    report['limitations']='snapshot-fit-table uses test-state full-rollout labels, evaluated on independent score futures; not deployable or an upper bound. Same-score maximum is selection-biased. Predictors use train fit bank only; no score bank model selection. Reused diagnostic case banks, two snapshots/case, finite futures and fixed teacher continuation restrict inference.'
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=['engineering','pilot'],default='engineering');args=p.parse_args()
    config=json.loads(args.config.read_text());spec=validate(config,args.profile);out=args.output
    if out.exists() and any(out.iterdir()):raise ValueError('fresh output required')
    source=json.loads(Path('source-manifest.json').read_text())
    for rel,digest in source.items():
        if hashlib.sha256(Path(rel).read_bytes()).hexdigest()!=digest:raise ValueError('source mismatch')
    plan=dict(version='v7-local-predictability-v1',started_utc=datetime.now(timezone.utc).isoformat(),config=config,profile=args.profile,source=source,
        runtime=dict(torch=str(torch.__version__),numpy=str(np.__version__)),schema=list(FEATURES),contract='teacher prefix and continuation; eligible even producers; one forced action with existing hold/inference delay; true-terminal return',accepted_win=False)
    write(out/'plan.json',plan);data={k:[] for k in ['train','test']}
    tasks=[dict(seed=s,split=k,times=config['snapshot_after_seconds'],replicas=spec['replicas'],output=str(out/k)) for k,seeds in spec['splits'].items() for s in seeds]
    from scripts.v7_local_predictability import collect_case as execute
    try:
        with ProcessPoolExecutor(max_workers=config['workers'],mp_context=get_context('spawn')) as pool:
            pending={pool.submit(execute,t):t for t in tasks}
            for f in as_completed(pending):
                t=pending[f];data[t['split']].append(f.result());write(out/'progress.json',dict(phase='collection',completed=sum(map(len,data.values())),total=len(tasks)))
        for k,entries in data.items():
            if sorted(c['seed'] for c in entries)!=sorted(spec['splits'][k]):raise ValueError('incomplete case coverage')
            entries.sort(key=lambda c:c['seed'])
        write(out/'data.json',data);write(out/'progress.json',dict(phase='fitting'))
        report=analyze(data,config,out);report.update(plan=plan,completed_utc=datetime.now(timezone.utc).isoformat());write(out/'results.json',report)
        write(out/'progress.json',dict(phase='complete',accepted_win=False))
    except BaseException as exc:
        write(out/'failure.json',dict(error=repr(exc),utc=datetime.now(timezone.utc).isoformat()));raise


if __name__=='__main__':main()
