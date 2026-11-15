"""Frozen-current-model RTT replication with prospective cases and stronger baselines."""
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
from scripts.network_forecast import (SCHEMA,HORIZONS,ForecastNet,collect_case,load_split,
    case_weights,weighted_quantile,predict_model,predict_baseline,summarize)

BASELINES=('raw-persistence','adjusted-persistence','rtt-ewma','bounded-trend','ridge-current')


def validate(config,profile):
    all_cases=[]
    for p in config['profiles'].values():
        seeds=p['cases']
        if not seeds or len(seeds)!=len(set(seeds)) or any(type(s) is not int or s<0 for s in seeds):raise ValueError('invalid case bank')
        all_cases+=seeds
    if len(all_cases)!=len(set(all_cases)):raise ValueError('engineering/test overlap')
    if not 1<=config['workers']<=4 or config['ridge_penalty']!=.1 or config['model_seeds']!=[1,2,3]:raise ValueError('fixed fit/resources required')
    names={f'{k}-{s}.pt' for k in ['current','history','transformer'] for s in [1,2,3]}
    if set(config['frozen_model_sha256'])!=names:raise ValueError('all frozen models required')
    return config['profiles'][profile]


def current_design(data,normalization=None):
    x=data['history'][:,-1].astype(float)
    if normalization is None:
        w=case_weights(data['seed'],np.ones(len(x),bool));mean=np.sum(x*w[:,None],0)
        scale=np.sqrt(np.sum((x-mean)**2*w[:,None],0)).clip(.05);normalization=dict(mean=mean,scale=scale)
    z=(x-normalization['mean'])/normalization['scale']
    return np.concatenate([z,np.ones((len(z),1))],1),normalization


def rtt_points(data,kind,ridge=None):
    rtt=data['history'][:,:,SCHEMA.index('rtt')].astype(float)*.1;valid=data['valid']&(rtt>0)
    latest=rtt[:,-1];n=len(rtt)
    if kind in ['raw-persistence','adjusted-persistence']:point=np.repeat(latest[:,None],3,1)
    elif kind=='rtt-ewma':
        w=valid*2.**(-np.arange(63,-1,-1)*(2/64)/.25)
        smooth=(rtt*w).sum(1)/np.maximum(w.sum(1),1e-12);point=np.repeat(smooth[:,None],3,1)
    elif kind=='bounded-trend':
        times=-np.arange(63,-1,-1)*(2/64);count=valid.sum(1).clip(1)
        tmean=(valid*times).sum(1)/count;ymean=(valid*rtt).sum(1)/count
        numerator=(valid*(times-tmean[:,None])*(rtt-ymean[:,None])).sum(1)
        denominator=(valid*(times-tmean[:,None])**2).sum(1)
        slope=np.divide(numerator,denominator,out=np.zeros(n),where=denominator>1e-12)
        change=slope[:,None]*np.asarray(HORIZONS)[None]
        limit=np.maximum(latest,.01)[:,None];point=latest[:,None]+np.clip(change,-limit,limit)
    elif kind=='ridge-current':
        design,_=current_design(data,ridge['normalization']);return (design@ridge['coefficients']).clip(0.)
    else:raise ValueError('unknown RTT baseline')
    return np.log1p(point.clip(0.)/.1)


def fit_baselines(train,config,original_ewma):
    design,norm=current_design(train);w=case_weights(train['seed'],train['mask'])[:,:,1]
    coeff=[];penalty=np.eye(design.shape[1])*config['ridge_penalty'];penalty[-1,-1]=0
    for h in range(3):
        if not w[:,h].any():raise ValueError('training RTT labels missing')
        coeff.append(np.linalg.solve(design.T@(w[:,h,None]*design)+penalty,design.T@(w[:,h]*train['y'][:,h,1])))
    ridge=dict(normalization=norm,coefficients=np.stack(coeff,1));fits={}
    for kind in BASELINES:
        point=rtt_points(train,kind,ridge);residual=np.zeros((3,3))
        for h in range(3):residual[h]=weighted_quantile(train['y'][:,h,1]-point[:,h],w[:,h],[.1,.5,.9])
        if kind=='raw-persistence':residual[:,1]=0. # Preserve the raw point; force ordered intervals around it below.
        if kind=='adjusted-persistence':residual=np.asarray(original_ewma['residual_quantiles'])[:,1,:].copy()
        fits[kind]=dict(residual=residual,ridge=ridge if kind=='ridge-current' else None)
    return fits


def predict_rule(data,kind,fit,ewma):
    prediction=predict_baseline(ewma,data);point=rtt_points(data,kind,fit['ridge'])
    interval=point[:,:,None]+fit['residual'][None]
    if kind=='raw-persistence':
        interval[:,:,0]=np.minimum(interval[:,:,0],point);interval[:,:,1]=point;interval[:,:,2]=np.maximum(interval[:,:,2],point)
    prediction[:,:,3:6]=interval.clip(0.)
    return prediction


def rtt_comparison(mode,reference,horizon='1.0'):
    a=mode['per_case'][horizon]['srtt_mae_seconds'];b=reference['per_case'][horizon]['srtt_mae_seconds']
    if not a or set(a)!=set(b):raise ValueError('paired fresh-RTT cases required')
    family={k:mode['case_family'][k] for k in a}
    if any(reference['case_family'][k]!=v for k,v in family.items()):raise ValueError('family pairing mismatch')
    rng=np.random.default_rng(718);draws=np.zeros(2000)
    for f in sorted(set(family.values())):
        delta=np.array([a[k]-b[k] for k in sorted(a) if family[k]==f])
        draws+=delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(1)*len(delta)/len(a)
    x=np.mean(list(a.values()));y=np.mean(list(b.values()))
    return dict(mae_difference_seconds=float(x-y),difference_ci95_seconds=np.quantile(draws,[.025,.975]).tolist(),
        relative_mae_reduction=float(1-x/y) if y>0 else None,cases_with_fresh_rtt=len(a),
        scope='paired family-stratified case bootstrap, overlapping windows/endpoints clustered; fixed fitted models; conditional on fresh RTT labels')


def load_frozen(root):
    models={}
    for kind in ['current','history','transformer']:
        for seed in [1,2,3]:
            name=f'{kind}-{seed}';ck=torch.load(root/(name+'.pt'),map_location='cpu',weights_only=True)
            if ck['schema']!=list(SCHEMA) or ck['horizons']!=list(HORIZONS) or ck['version']!='network-forecast-v1':raise ValueError('frozen forecast schema mismatch')
            model=ForecastNet(kind);model.load_state_dict(ck['weights']);model.eval();models[name]=(model,ck['mean'],ck['scale'])
    return models


def evaluate(data,fits,ewma,models):
    predictions={k:predict_rule(data,k,fits[k],ewma) for k in BASELINES}
    for name,model in models.items():predictions[name]=predict_model(model,data)
    for kind in ['current','history','transformer']:predictions[kind+'-ensemble']=np.mean([predictions[f'{kind}-{s}'] for s in [1,2,3]],0)
    modes={name:summarize(data,p) for name,p in predictions.items()}
    weights=case_weights(data['seed'],data['mask'])[:,:,1]
    for name,p in predictions.items():
        for h,seconds in enumerate(HORIZONS):
            error=np.abs(data['srtt_seconds'][:,h]-np.expm1(p[:,h,4])*.1)
            modes[name]['summary'][str(seconds)]['srtt_error_p95_seconds']=float(weighted_quantile(error,weights[:,h],[.95])[0]) if weights[:,h].any() else None
            modes[name]['summary'][str(seconds)]['srtt_interval_width_seconds']=float(np.dot(weights[:,h],(np.expm1(p[:,h,5])-np.expm1(p[:,h,3]))*.1)) if weights[:,h].any() else None
    return modes,predictions


def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,dict):return {k:jsonable(v) for k,v in x.items()}
    return x


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--profile',choices=['engineering','replication'],default='engineering');args=p.parse_args()
    config=json.loads(args.config.read_text());spec=validate(config,args.profile);root=args.reference_root;out=args.output
    if out.exists() and any(out.iterdir()):raise ValueError('fresh output required')
    torch.set_num_threads(1);source=json.loads(Path('source-manifest.json').read_text())
    for rel,h in source.items():
        if hashlib.sha256(Path(rel).read_bytes()).hexdigest()!=h:raise ValueError('source mismatch')
    reference=json.loads((root/'results.json').read_text())
    if hashlib.sha256((root/'results.json').read_bytes()).hexdigest()!=config['reference_result_sha256']:raise ValueError('reference result mismatch')
    if source.get('scripts/network_forecast.py')!=reference['plan']['source']['scripts/network_forecast.py']:raise ValueError('collector/predictor implementation changed')
    core={k:v for k,v in source.items() if k.startswith('src/')}
    if core!={k:v for k,v in reference['plan']['source'].items() if k.startswith('src/')}:raise ValueError('simulator changed')
    for name,h in config['frozen_model_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('frozen model changed')
    if hashlib.sha256((root/'baseline-fits.json').read_bytes()).hexdigest()!=config['reference_baselines_sha256']:raise ValueError('original baseline changed')
    old_cases={m['seed'] for metas in reference['collection'].values() for m in metas}
    if old_cases&set(s for prof in config['profiles'].values() for s in prof['cases']):raise ValueError('prior/new case overlap')
    train=load_split(root/'cases',reference['collection']['train']);cal=load_split(root/'cases',reference['collection']['calibration'])
    old=json.loads((root/'baseline-fits.json').read_text())['ewma'];ewma={k:np.asarray(v) if isinstance(v,list) else v for k,v in old.items()}
    fits=fit_baselines(train,config,ewma);models=load_frozen(root);cal_modes,_=evaluate(cal,fits,ewma,models)
    chosen=min(BASELINES,key=lambda k:cal_modes[k]['summary']['1.0']['srtt_mae_seconds'])
    plan=dict(version='rtt-replication-v1',started_utc=datetime.now(timezone.utc).isoformat(),config=config,profile=args.profile,source=source,
        selected_baseline=chosen,selection_scope='original calibration cases only; no new cases or original test labels used for fitting/selection',accepted_win=False)
    write(out/'plan.json',plan);write(out/'baseline-fits.json',jsonable(fits));write(out/'calibration.json',cal_modes)
    del train,cal
    tasks=[dict(seed=s,history_seconds=2.,anchors=reference['plan']['config']['anchors'],horizons=list(HORIZONS),output=str(out/'cases')) for s in spec['cases']]
    try:
        metas=[]
        with ProcessPoolExecutor(max_workers=config['workers'],mp_context=get_context('spawn')) as pool:
            pending={pool.submit(collect_case,t):t for t in tasks}
            for f in as_completed(pending):
                metas.append(f.result());write(out/'collection.json',metas);write(out/'progress.json',dict(phase='collection',completed=len(metas),total=len(tasks)))
        if sorted(m['seed'] for m in metas)!=sorted(spec['cases']):raise ValueError('incomplete case coverage')
        data=load_split(out/'cases',metas);modes,predictions=evaluate(data,fits,ewma,models)
        for name,prediction in predictions.items():np.savez_compressed(out/f'predictions-{name}.npz',prediction=prediction)
        comparisons={name:rtt_comparison(m, modes[chosen]) for name,m in modes.items()}
        primary=comparisons['current-ensemble'];raw=rtt_comparison(modes['current-ensemble'],modes['raw-persistence'])
        positive=primary['relative_mae_reduction'] is not None and primary['relative_mae_reduction']>=.1 and primary['difference_ci95_seconds'][1]<0 and raw['mae_difference_seconds']<0
        result=dict(plan=plan,completed_utc=datetime.now(timezone.utc).isoformat(),collection=metas,modes=modes,comparisons=comparisons,
            primary_comparison=primary,raw_persistence_comparison=raw,rtt_evidence=positive,accepted_win=False,
            scope='fresh-case confirmation of a previously selected SRTT lead; frozen neural models, train-only baseline fitting, original calibration selects comparator; no physical RTT/outage/deadline or transport performance claim')
        files=[*out.glob('predictions-*.npz'),out/'baseline-fits.json'];result['artifact_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        write(out/'results.json',result);write(out/'progress.json',dict(phase='complete',rtt_evidence=positive,accepted_win=False))
    except BaseException as exc:
        write(out/'failure.json',dict(error=repr(exc),utc=datetime.now(timezone.utc).isoformat()));raise


if __name__=='__main__':main()
