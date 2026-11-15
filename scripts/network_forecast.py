"""Causal delivery/RTT/ACK-silence forecasting under a fixed transport teacher."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from trace_transport.experiment_io import write
from trace_transport.reactive_sender import MacroWorld,PhysicalHistory,FEATURES
from trace_transport.goodput_reactive import sample_case
from scripts.v7_local_predictability import teacher_actions

SCHEMA=tuple(f for f in FEATURES if f!='age')
COLUMNS=[FEATURES.index(f) for f in SCHEMA]
HORIZONS=(.5,1.,2.)
QUANTILES=np.array([.1,.5,.9])
KINDS=('current','history','transformer')
BASELINES=('constant','persistence','moving-average','ewma')


def validate(config,profile):
    spec=config['profiles'][profile];all_seeds=[]
    for p in config['profiles'].values():
        for split in ['train','calibration','test']:
            seeds=p['splits'][split]
            if not seeds or len(seeds)!=len(set(seeds)) or any(type(s) is not int or s<0 for s in seeds):raise ValueError('invalid case bank')
            all_seeds+=seeds
    if len(all_seeds)!=len(set(all_seeds)):raise ValueError('case leakage across profiles/splits')
    if config['horizons']!=list(HORIZONS) or config['history_seconds']!=2. or config['bins']!=64:raise ValueError('forecast schema mismatch')
    times=config['anchors']
    if not times or sorted(set(times))!=times or min(times)<2 or max(times)+max(HORIZONS)>12:raise ValueError('invalid causal anchor windows')
    if not 1<=config['workers']<=4 or config['model_seeds']!=[1,2,3] or not 1<=spec['fit_steps']<=2000:raise ValueError('invalid bounded fit')
    return spec


def observe_counters(world):
    obs=world.obs['sender'];bulk=[FEATURES.index(f) for f in ['image_bytes','map_bytes','bulk_bytes']]
    return dict(time=world.env.now,acked=np.array([s.acked for s in world.env.sessions]),
        samples=np.array([sum(len(m.sender.rtt_samples) for m in s.messages.values()) for s in world.env.sessions]),
        srtt=np.array([s.srtt for s in world.env.sessions]),backlog=(obs[:,bulk]>0).any(1))


def future_targets(trace,start,endpoint,horizons):
    times=np.array([r['time'] for r in trace]);base=trace[start];ys=[];masks=[];durations=[];fractions=[];raw=[]
    for horizon in horizons:
        end=int(np.searchsorted(times,base['time']+horizon-1e-8))
        if end>=len(trace):raise ValueError('future horizon incomplete')
        last=trace[end];dt=last['time']-base['time'];acked=int(last['acked'][endpoint]-base['acked'][endpoint])
        fresh=int(last['samples'][endpoint]-base['samples'][endpoint])
        if dt<=0 or abs(dt-horizon)>1e-6 or acked<0 or fresh<0:raise ValueError('nonmonotonic endpoint counters')
        rtt=float(last['srtt'][endpoint]);rtt_valid=fresh>0 and rtt>0
        ys.append([np.log1p(acked/(128000.*horizon)),np.log1p(rtt/.1) if rtt_valid else 0.,float(acked==0)])
        masks.append([True,rtt_valid,True]);durations.append(dt);raw.append([acked,rtt])
        fractions.append(float(np.dot(np.diff(times[start:end+1]),[r['backlog'][endpoint] for r in trace[start:end]])/dt))
    return np.asarray(ys,np.float32),np.asarray(masks,bool),durations,fractions,np.asarray(raw)


def collect_case(task):
    world=MacroWorld(task['seed'],production=task.get('production',12.),drain=task.get('drain',3.))
    world.history=PhysicalHistory(world.features(),seconds=task['history_seconds'],bins=64)
    trace=[];anchors=[];next_anchor=0
    while True:
        frame=observe_counters(world);trace.append(frame)
        if next_anchor<len(task['anchors']) and world.env.now+1e-8>=task['anchors'][next_anchor]:
            observation=world.observation()
            for endpoint in range(0,world.env.count,2):
                if frame['backlog'][endpoint]:
                    anchors.append(dict(start=len(trace)-1,endpoint=endpoint,history=observation['history'][endpoint][:,COLUMNS].copy(),valid=observation['valid'][endpoint].copy()))
            next_anchor+=1
        if world.env.done():break
        world.step(teacher_actions(world))
        if task.get('output') and len(trace)%20==0:write(Path(task['output'])/f"case-{task['seed']}-progress.json",dict(phase='collection',simulated_seconds=world.env.now,anchors=len(anchors)))
    n=len(anchors);h=len(task['horizons']);arrays=dict(history=np.empty((n,64,len(SCHEMA)),np.float32),valid=np.empty((n,64),bool),
        y=np.empty((n,h,3),np.float32),mask=np.empty((n,h,3),bool),duration=np.empty((n,h)),backlog_fraction=np.empty((n,h)),
        ack_bytes=np.empty((n,h),np.int64),srtt_seconds=np.empty((n,h)),anchor_time=np.empty(n),endpoint=np.empty(n,int),seed=np.full(n,task['seed'],dtype=np.int64))
    for i,a in enumerate(anchors):
        y,mask,dt,fraction,raw=future_targets(trace,a['start'],a['endpoint'],task['horizons'])
        arrays['history'][i]=a['history'];arrays['valid'][i]=a['valid'];arrays['y'][i]=y;arrays['mask'][i]=mask
        arrays['duration'][i]=dt;arrays['backlog_fraction'][i]=fraction;arrays['ack_bytes'][i]=raw[:,0];arrays['srtt_seconds'][i]=raw[:,1];arrays['anchor_time'][i]=trace[a['start']]['time'];arrays['endpoint'][i]=a['endpoint']
    if not task.get('output'):return arrays
    root=Path(task['output']);root.mkdir(parents=True,exist_ok=True);path=root/f"case-{task['seed']}.npz";temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(path)
    meta=dict(seed=task['seed'],family=world.label['family'],samples=n,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write(root/f"case-{task['seed']}.json",meta);write(root/f"case-{task['seed']}-progress.json",dict(phase='complete',simulated_seconds=world.env.now,samples=n))
    return meta


def load_split(root,metas):
    batches=[]
    for meta in sorted(metas,key=lambda m:m['seed']):
        path=root/meta['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('case data hash mismatch')
        with np.load(path,allow_pickle=False) as z:batch={k:z[k] for k in z.files}
        if not (batch['seed']==meta['seed']).all() or len(batch['seed'])!=meta['samples']:raise ValueError('case identity mismatch')
        batch['family']=np.full(len(batch['seed']),meta['family'],dtype='U32')
        batches.append(batch)
    data={k:np.concatenate([b[k] for b in batches]) for k in batches[0]}
    n=len(data['seed'])
    if not n or data['history'].shape!=(n,64,len(SCHEMA)) or data['y'].shape!=(n,3,3) or data['mask'].shape!=(n,3,3) or data['ack_bytes'].shape!=(n,3) or not data['valid'][:,-1].all():raise ValueError('invalid forecast dataset')
    if not all(np.isfinite(data[k]).all() for k in ['history','y','duration','backlog_fraction']):raise ValueError('nonfinite forecast data')
    if (data['duration']<=0).any():raise ValueError('invalid forecast durations')
    return data


def case_weights(seeds,mask):
    weights=np.zeros(mask.shape,float)
    for seed in np.unique(seeds):
        selected=(seeds==seed).reshape((-1,)+(1,)*(mask.ndim-1))&mask
        counts=selected.sum(0);weights+=selected/np.maximum(1,counts)
    totals=weights.sum(0);return weights/np.maximum(totals,1)


def weighted_quantile(values,weights,qs):
    order=np.argsort(values);x=np.asarray(values)[order];w=np.asarray(weights)[order]
    positive=w>0;x=x[positive];w=w[positive]
    if not len(x):return np.zeros(len(qs))
    cumulative=np.cumsum(w)/w.sum();return np.interp(qs,cumulative,x)


def decode_srtt(encoded):
    """v6 inherits v4's log1p(SRTT / 1 ms) observation encoding."""
    return np.expm1(np.asarray(encoded,dtype=float))*.001


def baseline_points(data,kind,history_seconds=2.):
    x=data['history'];v=data['valid'];n=len(x);h=len(HORIZONS);width=history_seconds/x.shape[1]
    ack=np.expm1(x[:,:,SCHEMA.index('ack_rate')])*128000.
    rtt=decode_srtt(x[:,:,SCHEMA.index('rtt')])
    if kind=='persistence':w=np.zeros_like(ack);w[:,-1]=1.
    elif kind=='moving-average':w=v.astype(float)
    elif kind=='ewma':w=np.broadcast_to(2.**(-np.arange(x.shape[1]-1,-1,-1)*width/.25),ack.shape)*v
    elif kind=='constant':w=np.zeros_like(ack)
    else:raise ValueError('unknown baseline')
    rate=(ack*w).sum(1)/np.maximum(w.sum(1),1e-12);rrt=rtt[:,-1]
    if kind=='constant':rrt=np.zeros(n)
    point=np.stack([np.log1p(rate/128000.),np.log1p(rrt/.1)],-1)
    return np.broadcast_to(point[:,None,:],(n,h,2)).copy(),(rate==0).astype(float)


def fit_baseline(train,kind):
    point,_=baseline_points(train,kind);weights=case_weights(train['seed'],train['mask']);residuals=np.zeros((3,2,3))
    gap=[]
    for horizon in range(3):
        for target in range(2):residuals[horizon,target]=weighted_quantile(train['y'][:,horizon,target]-point[:,horizon,target],weights[:,horizon,target],QUANTILES)
        gap.append(float(np.dot(weights[:,horizon,2],train['y'][:,horizon,2])))
    return dict(kind=kind,residual_quantiles=residuals,constant_gap=np.array(gap))


def predict_baseline(fit,data):
    point,recent_silence=baseline_points(data,fit['kind']);q=np.maximum(0,point[:,:,:,None]+fit['residual_quantiles'][None])
    gap=np.broadcast_to(fit['constant_gap'],(len(point),3)).copy() if fit['kind']=='constant' else np.broadcast_to(recent_silence[:,None],(len(point),3))
    return np.concatenate([q[:,:,0],q[:,:,1],gap[:,:,None]],-1)


class ForecastNet(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;n=len(SCHEMA)
        if kind=='current':self.encoder=nn.Sequential(nn.Linear(n,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh())
        elif kind=='history':self.encoder=nn.Sequential(nn.Linear(64*(n+1),32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        elif kind=='transformer':
            self.input=nn.Linear(n,32);self.position=nn.Parameter(torch.zeros(1,64,32))
            self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(32,4,64,dropout=0.,batch_first=True),1,enable_nested_tensor=False)
        else:raise ValueError('unknown forecast kind')
        self.head=nn.Linear(32,21)
    def forward(self,x,valid):
        x=x.masked_fill(~valid[:,:,None],0.)
        if self.kind=='current':z=self.encoder(x[:,-1])
        elif self.kind=='history':z=self.encoder(torch.cat([x,valid[:,:,None].float()],-1).flatten(1))
        else:z=self.encoder(self.input(x)+self.position,src_key_padding_mask=~valid)[:,-1]
        raw=self.head(z).reshape(-1,3,7)
        return torch.cat([raw[:,:,:3].sort(-1).values,raw[:,:,3:6].sort(-1).values,raw[:,:,6:]],-1)


def train_model(train,kind,seed,steps,output):
    torch.set_num_threads(1);torch.manual_seed(seed)
    x=torch.tensor(train['history']);valid=torch.tensor(train['valid']);past=x[valid]
    mean=past.mean(0);scale=past.std(0,unbiased=False).clamp_min(.05);x=(x-mean)/scale
    y=torch.tensor(train['y']);weights=torch.tensor(case_weights(train['seed'],train['mask']),dtype=torch.float32)*len(x)
    model=ForecastNet(kind);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    rng=torch.Generator().manual_seed(seed+10000);quantiles=torch.tensor(QUANTILES,dtype=torch.float32)
    model.train()
    for step in range(steps):
        ids=torch.randint(len(x),(min(128,len(x)),),generator=rng);p=model(x[ids],valid[ids]);losses=[]
        for target,start in [(0,0),(1,3)]:
            error=y[ids,:,target,None]-p[:,:,start:start+3]
            pinball=torch.maximum(quantiles*error,(quantiles-1)*error).mean(-1)
            losses.append(pinball*weights[ids,:,target])
        losses.append(F.binary_cross_entropy_with_logits(p[:,:,6],y[ids,:,2],reduction='none')*weights[ids,:,2])
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite forecast loss')
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if (step+1)%100==0:write(output/f'{kind}-{seed}-progress.json',dict(phase='fitting',steps=step+1,total=steps,training_loss=float(loss.detach())))
    model.eval();state=dict(version='network-forecast-v1',kind=kind,schema=list(SCHEMA),horizons=list(HORIZONS),history_seconds=2.,
        weights=model.state_dict(),mean=mean,scale=scale,seed=seed,fit_steps=steps)
    torch.save(state,output/f'{kind}-{seed}.pt')
    write(output/f'{kind}-{seed}-progress.json',dict(phase='complete',steps=steps,training_loss=float(loss.detach())))
    return model,mean,scale


def predict_model(fitted,data):
    model,mean,scale=fitted;parts=[]
    with torch.inference_mode():
        for start in range(0,len(data['seed']),128):
            x=(torch.tensor(data['history'][start:start+128])-mean)/scale;valid=torch.tensor(data['valid'][start:start+128])
            p=model(x,valid);p[:,:,:6]=p[:,:,:6].clamp_min(0.);p[:,:,6]=p[:,:,6].sigmoid();parts.append(p.numpy())
    out=np.concatenate(parts)
    if not np.isfinite(out).all():raise ValueError('nonfinite forecasts')
    return out


def summarize(data,pred):
    summaries={};cases={};w=case_weights(data['seed'],data['mask'])
    for h,seconds in enumerate(HORIZONS):
        y=data['y'][:,h];p=pred[:,h];dt=seconds
        actual_bytes=data['ack_bytes'][:,h];estimated_bytes=np.expm1(p[:,1])*128000.*dt
        err=y[:,0,None]-p[:,:3];pin=np.maximum(QUANTILES*err,(QUANTILES-1)*err).mean(1)
        actual_rtt=data['srtt_seconds'][:,h];estimated_rtt=np.expm1(p[:,4])*.1
        terms=dict(ack_mae_bytes=np.abs(actual_bytes-estimated_bytes),ack_pinball_log=pin,
            ack_coverage80=((y[:,0]>=p[:,0])&(y[:,0]<=p[:,2])).astype(float),
            ack_interval_width_bytes=(np.expm1(p[:,2])-np.expm1(p[:,0]))*128000.*dt,
            srtt_mae_seconds=np.abs(actual_rtt-estimated_rtt),srtt_coverage80=((y[:,1]>=p[:,3])&(y[:,1]<=p[:,5])).astype(float),
            ack_silence_brier=(p[:,6]-y[:,2])**2,ack_silence_prevalence=y[:,2])
        summary={};case_table={}
        for name,values in terms.items():
            target=1 if name.startswith('srtt') else (2 if name.startswith('ack_silence') else 0)
            mask=data['mask'][:,h,target];summary[name]=float(np.dot(w[:,h,target],values)) if mask.any() else None
            case_table[name]={str(int(seed)):float(values[(data['seed']==seed)&mask].mean()) for seed in np.unique(data['seed']) if ((data['seed']==seed)&mask).any()}
        summary.update(samples=len(y),windows_with_fresh_rtt=int(data['mask'][:,h,1].sum()),future_backlog_fraction=float(np.dot(w[:,h,0],data['backlog_fraction'][:,h])))
        bins=[]
        for low,high in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
            selected=(p[:,6]>=low)&((p[:,6]<high) if high<1 else (p[:,6]<=high))
            if selected.any():
                bw=w[:,h,2]*selected;bw/=bw.sum()
                bins.append(dict(low=float(low),high=float(high),samples=int(selected.sum()),mean_probability=float(np.dot(bw,p[:,6])),observed_frequency=float(np.dot(bw,y[:,2]))))
        saturated=data['backlog_fraction'][:,h]>=.9
        sw=case_weights(data['seed'],saturated)
        summary['backlogged90_windows']=int(saturated.sum())
        summary['ack_mae_bytes_backlogged90']=float(np.dot(sw,terms['ack_mae_bytes'])) if saturated.any() else None
        summary['ack_silence_reliability_bins']=bins # Descriptive row counts; rows are not independent cases.
        summaries[str(seconds)]=summary;cases[str(seconds)]=case_table
    return dict(summary=summaries,per_case=cases,case_family={str(int(seed)):str(data['family'][np.flatnonzero(data['seed']==seed)[0]]) for seed in np.unique(data['seed'])})


def comparison(mode,reference,horizon='1.0'):
    a=mode['per_case'][horizon]['ack_mae_bytes'];b=reference['per_case'][horizon]['ack_mae_bytes']
    if set(a)!=set(b):raise ValueError('paired cases required')
    delta=np.array([a[k]-b[k] for k in sorted(a)]);rng=np.random.default_rng(604);estimates=np.zeros(2000)
    family=mode['case_family']
    for f in sorted(set(family.values())):
        selected=np.array([a[k]-b[k] for k in sorted(a) if family[k]==f])
        estimates+=selected[rng.integers(len(selected),size=(2000,len(selected)))].mean(1)*len(selected)/len(delta)
    x=mode['summary'][horizon]['ack_mae_bytes'];y=reference['summary'][horizon]['ack_mae_bytes']
    return dict(mae_difference=float(delta.mean()),difference_ci95=np.quantile(estimates,[.025,.975]).tolist(),relative_mae_reduction=1-x/y if y>0 else None,
        scope='family-stratified case bootstrap; endpoints and overlapping windows clustered, fixed fitted predictor; not a transport acceptance test')


def analyze(datasets,config,steps,output):
    results={split:{} for split in ['calibration','test']};predictions={split:{} for split in results}
    train=datasets['train'];baseline_fits={}
    for name in BASELINES:
        baseline=fit_baseline(train,name)
        baseline_fits[name]={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in baseline.items()}
        for split in results:predictions[split][name]=predict_baseline(baseline,datasets[split])
    write(output/'baseline-fits.json',baseline_fits)
    for kind in KINDS:
        for seed in config['model_seeds']:
            write(output/'progress.json',dict(phase='fitting',model=f'{kind}-{seed}'))
            fitted=train_model(train,kind,seed,steps,output)
            for split in results:predictions[split][f'{kind}-{seed}']=predict_model(fitted,datasets[split])
        for split in results:predictions[split][kind+'-ensemble']=np.mean([predictions[split][f'{kind}-{s}'] for s in config['model_seeds']],axis=0)
    for split in results:
        for name,p in predictions[split].items():
            results[split][name]=summarize(datasets[split],p)
            np.savez_compressed(output/f'predictions-{split}-{name}.npz',prediction=p)
    baseline=min(BASELINES,key=lambda n:results['calibration'][n]['summary']['1.0']['ack_mae_bytes'])
    comparisons={name:comparison(row,results['test'][baseline]) for name,row in results['test'].items()}
    history=comparison(results['test']['transformer-ensemble'],results['test']['current-ensemble'])
    primary=comparisons['transformer-ensemble'];coverage=results['test']['transformer-ensemble']['summary']['1.0']['ack_coverage80']
    positive=primary['relative_mae_reduction'] is not None and primary['relative_mae_reduction']>=.1 and primary['difference_ci95'][1]<0 and .7<=coverage<=.9
    return dict(accepted_win=False,forecast_evidence=positive,selected_baseline=baseline,primary='transformer-ensemble: 1-second ACK bytes',
        primary_comparison=primary,history_vs_current=history,comparisons_to_baseline=comparisons,modes=results,
        scope='fresh synthetic case forecasting under fixed teacher; calibration selects baseline only; all fixed models reported; no scheduler/control deployment or real-WiFi claim')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=['engineering','pilot'],default='engineering');args=p.parse_args()
    config=json.loads(args.config.read_text());spec=validate(config,args.profile);out=args.output
    if out.exists() and any(out.iterdir()):raise ValueError('fresh output required')
    torch.set_num_threads(1);source=json.loads(Path('source-manifest.json').read_text())
    for path,digest in source.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('source mismatch')
    plan=dict(version='network-forecast-v1',started_utc=datetime.now(timezone.utc).isoformat(),config=config,profile=args.profile,source=source,
        schema=list(SCHEMA),targets=['log_ack_payload_rate','log_future_srtt_if_fresh','ack_payload_silence'],runtime=dict(torch=str(torch.__version__),numpy=str(np.__version__)),accepted_win=False)
    write(out/'plan.json',plan);metas={split:[] for split in spec['splits']}
    tasks=[dict(seed=s,split=split,history_seconds=2.,anchors=config['anchors'],horizons=config['horizons'],output=str(out/'cases')) for split,seeds in spec['splits'].items() for s in seeds]
    from scripts.network_forecast import collect_case as execute
    try:
        with ProcessPoolExecutor(max_workers=config['workers'],mp_context=get_context('spawn')) as pool:
            pending={pool.submit(execute,t):t for t in tasks}
            for f in as_completed(pending):
                t=pending[f];metas[t['split']].append(f.result());write(out/'progress.json',dict(phase='collection',completed=sum(map(len,metas.values())),total=len(tasks)))
                write(out/'collection.json',metas)
        for split,entries in metas.items():
            if sorted(x['seed'] for x in entries)!=sorted(spec['splits'][split]):raise ValueError('incomplete split coverage')
        datasets={k:load_split(out/'cases',v) for k,v in metas.items()}
        result=analyze(datasets,config,spec['fit_steps'],out)
        files=[*out.glob('*.pt'),*out.glob('predictions-*.npz'),out/'baseline-fits.json']
        result['artifact_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        result.update(plan=plan,completed_utc=datetime.now(timezone.utc).isoformat(),collection=metas)
        write(out/'results.json',result);write(out/'progress.json',dict(phase='complete',accepted_win=False,forecast_evidence=result['forecast_evidence']))
    except BaseException as exc:
        write(out/'failure.json',dict(error=repr(exc),utc=datetime.now(timezone.utc).isoformat()));raise


if __name__=='__main__':main()
