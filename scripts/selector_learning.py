"""Fixed-budget offline controller-selection learnability diagnostic.

Family labels, future outcomes and seeds never enter model tensors. Cost-table
lookups are diagnostic: they do not execute the probe-then-switch policy.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from trace_transport.experiment_io import write
from trace_transport.goodput_evaluation import summarize
from trace_transport.joint_models import LocalEncoder,JointModelConfig
from trace_transport.joint_session import SENDER_FEATURES
from scripts.reactive_diagnostics_report import admissible


KINDS=('current-mlp','history-mlp','gru','transformer')


def dataset(data,split):
    part=data['splits'][split];controls=data['plan']['controls'];names=[c['name'] for c in controls]
    seeds=data['plan']['splits'][split]
    if len(set(seeds))!=len(seeds):raise ValueError('duplicate split seeds')
    if set(part['controls'])!=set(names):raise ValueError('incomplete candidate coverage')
    probes={p['seed']:p for p in part['probes']}
    if set(probes)!=set(seeds) or len(probes)!=len(part['probes']):raise ValueError('probe membership mismatch')
    indexed={}
    for name in names:
        records=part['controls'][name]['records'];indexed[name]={r['seed']:r for r in records}
        if set(indexed[name])!=set(seeds) or len(records)!=len(seeds):raise ValueError('control membership mismatch')
    histories=[];valid=[];targets=[];rows=[];feasible=[]
    for seed in seeds:
        p=probes[seed]
        if p['schema']!=list(SENDER_FEATURES):raise ValueError('probe schema mismatch')
        x=np.asarray(p['history'],np.float32);v=np.asarray(p['valid'],bool)
        if x.shape!=(data['plan']['history'],len(SENDER_FEATURES)) or v.shape!=(len(x),) or not v[-1] or not np.isfinite(x).all():raise ValueError('invalid local history')
        candidates=[indexed[n][seed] for n in names];reference=candidates[0]
        allowed=[admissible(row,reference) for row in candidates]
        choices=[i for i,ok in enumerate(allowed) if ok]
        target=max(choices,key=lambda i:candidates[i]['verified_bulk_bytes']) if choices else 0
        histories.append(x);valid.append(v);targets.append(target);rows.append(candidates);feasible.append(allowed)
    return dict(x=torch.tensor(np.stack(histories)),valid=torch.tensor(np.stack(valid)),
        y=torch.tensor(targets),records=rows,admissible=np.asarray(feasible,bool),seeds=seeds,names=names)


class Selector(nn.Module):
    def __init__(self,kind,history=64,choices=9):
        super().__init__();self.kind=kind;n=len(SENDER_FEATURES)
        if kind=='current-mlp':self.encoder=nn.Sequential(nn.Linear(n,96),nn.GELU(),nn.Linear(96,96),nn.GELU());width=96
        elif kind=='history-mlp':self.encoder=nn.Sequential(nn.Linear(history*(n+1),8),nn.GELU(),nn.Linear(8,32),nn.GELU());width=32
        elif kind=='gru':self.encoder=nn.GRU(n,48,batch_first=True);width=48
        elif kind=='transformer':self.encoder=LocalEncoder(n,JointModelConfig(history=history,width=32,layers=1,heads=4));width=32
        else:raise ValueError('unknown selector')
        self.head=nn.Linear(width,choices)
    def forward(self,x,valid):
        x=x.masked_fill(~valid[:,:,None],0)
        if self.kind=='current-mlp':h=self.encoder(x[:,-1])
        elif self.kind=='history-mlp':h=self.encoder(torch.cat([x,valid[:,:,None].float()],-1).flatten(1))
        elif self.kind=='gru':
            # Valid observations form a suffix. Pack without inserting pad steps.
            lengths=valid.sum(-1);shifted=torch.zeros_like(x)
            for i,length in enumerate(lengths.tolist()):shifted[i,:length]=x[i,valid[i]]
            packed=nn.utils.rnn.pack_padded_sequence(shifted,lengths.cpu(),batch_first=True,enforce_sorted=False)
            _,hidden=self.encoder(packed);h=hidden[-1]
        else:h=self.encoder(x,valid)
        return self.head(h)


def normalizer(part):
    values=part['x'][part['valid']];return values.mean(0),values.std(0,unbiased=False).clamp_min(.1)


def normalized(part,mean,scale):return ((part['x']-mean)/scale).clamp(-10,10).masked_fill(~part['valid'][:,:,None],0)


def fit(model,x,valid,y,epochs):
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad();loss=nn.functional.cross_entropy(model(x,valid),y);loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    model.eval()
    with torch.inference_mode():pred=model(x,valid).argmax(-1)
    return dict(training_accuracy=float((pred==y).float().mean()),final_training_loss=float(loss.detach()))


def lookup(part,pred):
    pred=[int(i) for i in pred];chosen=[rows[i] for rows,i in zip(part['records'],pred)]
    reference=[rows[0] for rows in part['records']];target=[rows[int(i)] for rows,i in zip(part['records'],part['y'])]
    violations=sum(not part['admissible'][j,i] for j,i in enumerate(pred))
    gaps=[b['verified_bulk_bytes']-a['verified_bulk_bytes'] for a,b in zip(chosen,target)]
    return dict(scope='same-case candidate lookup from start; probe/switch not executed; not online performance or a win',
        target_accuracy=float(np.mean(np.asarray(pred)==part['y'].numpy())),inadmissible_case_count=violations,
        mean_unconstrained_verified_byte_gap_to_admissible_target=float(np.mean(gaps)),summary=summarize(chosen),reference_summary=summarize(reference),
        predictions=[dict(seed=s,choice=i,control=part['names'][i]) for s,i in zip(part['seeds'],pred)])


def save_model(path,model,history,controls,mean,scale,plan):
    # JSON roundtrip keeps metadata primitive, including torch's str subclass.
    metadata=json.loads(json.dumps(plan))
    torch.save(dict(kind=model.kind,history=history,choices=len(controls),controls=controls,
        sender_features=list(SENDER_FEATURES),state=model.state_dict(),mean=mean,scale=scale,plan=metadata),path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=400)
    p.add_argument('--tiny-only',action='store_true');args=p.parse_args();out=args.output
    if out.exists() and any(f.name!='service.log' for f in out.iterdir()):raise ValueError('new output directory required')
    if args.epochs<1:raise ValueError('positive training budget required')
    data=json.loads(args.dataset.read_text());dev=dataset(data,'development');torch.set_num_threads(1)
    if not args.tiny_only:
        if 'completed_utc' not in data:raise ValueError('collection must finish before full comparison')
        val=dataset(data,'validation')
        if set(dev['seeds'])&set(val['seeds']):raise ValueError('overlapping splits')
    mean,scale=normalizer(dev);x=normalized(dev,mean,scale)
    plan=dict(version='selector-learning-v1',started_utc=datetime.now(timezone.utc).isoformat(),dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        kinds=list(KINDS),seeds=[1,2,3],epochs=args.epochs,lr=.001,weight_decay=.0001,history=int(x.shape[1]),
        source=json.loads(Path('source-manifest.json').read_text()),runtime=dict(torch=str(torch.__version__),numpy=str(np.__version__)),
        objective='full-batch unweighted cross entropy of per-case admissible finite-control hindsight targets',
        scope='offline diagnostic; final epochs only, no validation selection or tuning; not joint RL or executed transport performance',
        development_labels=dict(Counter(dev['y'].tolist())),tiny_only=args.tiny_only)
    write(out/'plan.json',plan);result=dict(plan=plan,tiny_fit={},models={})
    # At most four distinct development targets. No validation information used.
    ids=[]
    for target in sorted(set(dev['y'].tolist())):
        ids.append(int(torch.nonzero(dev['y']==target)[0]))
        if len(ids)==4:break
    for kind in KINDS:
        torch.manual_seed(1);model=Selector(kind,history=x.shape[1],choices=len(dev['names']))
        write(out/'progress.json',dict(phase='tiny-fit',model=kind))
        result['tiny_fit'][kind]=dict(cases=[dev['seeds'][i] for i in ids],
            **fit(model,x[ids],dev['valid'][ids],dev['y'][ids],args.epochs))
        write(out/'results.json',result)
    if not args.tiny_only:
        xv=normalized(val,mean,scale)
        result['baselines']={name:lookup(val,[i]*len(val['seeds'])) for i,name in enumerate(dev['names'])}
        rtt=SENDER_FEATURES.index('min_rtt');ack=SENDER_FEATURES.index('ack_rate')
        heuristic=[6 if row[-1,rtt]<np.log1p(5.) and row[-1,ack]>0 else 0 for row in val['x'].numpy()]
        result['baselines']['low-rtt-bdp-rule']=lookup(val,heuristic)
        for kind in KINDS:
            result['models'][kind]={}
            for seed in [1,2,3]:
                write(out/'progress.json',dict(phase='training',model=kind,seed=seed))
                torch.manual_seed(seed);model=Selector(kind,history=x.shape[1],choices=len(dev['names']))
                training=fit(model,x,dev['valid'],dev['y'],args.epochs)
                with torch.inference_mode():pred=model(xv,val['valid']).argmax(-1).tolist()
                filename=f'{kind}-{seed}.pt'
                save_model(out/filename,model,int(x.shape[1]),data['plan']['controls'],mean,scale,plan)
                result['models'][kind][str(seed)]=dict(training=training,parameters=sum(p.numel() for p in model.parameters()),lookup=lookup(val,pred))
                write(out/'results.json',result)
    result['completed_utc']=datetime.now(timezone.utc).isoformat();write(out/'results.json',result)
    write(out/'progress.json',dict(phase='complete',reserved_tests='unopened'))


if __name__=='__main__':main()
