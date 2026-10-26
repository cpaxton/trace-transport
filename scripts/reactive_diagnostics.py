"""V6 role, applied-action and separate-development control diagnostics.

No training, held-out acceptance or hindsight deployment claim. Outputs retain
all records, configuration and timestamps; source is frozen by the launcher.
"""
import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from trace_transport.experiment_io import write
from trace_transport.goodput_reactive import DOMAIN, GoodputReactiveEnv, sample_case
from trace_transport.goodput_evaluation import summarize, compare
from trace_transport.joint_evaluation import evaluate, MAX_GOODPUT_ENDPOINTS
from trace_transport.joint_models import load_checkpoint
from trace_transport.joint_training import Histories


def bank(start, per_family=1):
    families={f:[] for f in DOMAIN['families']}
    for seed in range(start,start+10000):
        f=sample_case(seed)[2]['family']
        if len(families[f])<per_family:families[f].append(seed)
        if all(len(x)==per_family for x in families.values()):return sorted(s for x in families.values() for s in x)
    raise ValueError('development bank incomplete')


class InstrumentedEnv(GoodputReactiveEnv):
    def reset(self, **kwargs):
        self.control_stats=Counter();self.in_control=False
        return super().reset(**kwargs)
    def on_receiver_control(self,index):
        s=self.sessions[index];old=self._feedback_settings[index]
        changed=old!=(s.batch,s.delay,s.ack_mode)
        pending=len(s.pending);generations=dict(s.ack_generations)
        self.control_stats['setting_changes']+=int(changed)
        self.control_stats['changes_with_pending']+=int(changed and bool(pending))
        self.in_control=True
        try:super().on_receiver_control(index)
        finally:self.in_control=False
        self.control_stats['pending_timer_replacements']+=sum(
            s.ack_generations.get(mid,0)!=generations.get(mid,0) for mid in s.pending) if changed else 0
    def _flush(self,index,mid,generation=None):
        before=self.sessions[index].feedback_wire
        super()._flush(index,mid,generation)
        delta=self.sessions[index].feedback_wire-before
        if self.in_control and delta:
            self.control_stats['control_flushes']+=1;self.control_stats['control_flush_wire_bytes']+=delta


def action_case(seed,checkpoint,baseline,production=12.,drain=3.,*,receiver_mode="greedy",sampling_seed=0):
    if receiver_mode not in ("greedy","sampled"):raise ValueError("unknown receiver mode")
    generator=torch.Generator().manual_seed(sampling_seed)
    sender,receiver,model,metadata=load_checkpoint(checkpoint)
    if metadata['profile']!='goodput-v6':raise ValueError('expected v6 checkpoint')
    cfg,w,label=sample_case(seed,production=production,drain=drain)
    env=InstrumentedEnv(replace(cfg,initial_rate=baseline['initial_rate']),w,
        churn=label.get('churn',False),churn_times=label.get('churn_times'))
    obs,_=env.reset(seed=seed)
    for s in env.sessions:
        s.rate_control=baseline['rate'];s.scheduler=baseline['scheduler']
        s.batch=s.baseline_batch=baseline['batch'];s.delay=s.baseline_delay=baseline['delay']
        s.ack_mode=s.baseline_ack_mode='constant'
    obs=env.observe();h=Histories(obs,model);counts=Counter();prob_sum=None;entropy=0.;rows=0
    while not env.done():
        h.append(obs);n=env.count;size=MAX_GOODPUT_ENDPOINTS
        if n>size:raise ValueError('endpoint bound exceeded')
        valid=np.zeros((size,model.history),bool);valid[:n]=h.valid
        tx=np.zeros((n,3),np.float32);rx=np.zeros(n,np.int64)
        def padded(x):
            a=np.zeros((size,*x.shape[1:]),x.dtype);a[:n]=x;return torch.as_tensor(a)
        with torch.inference_mode():
            if obs['sender_active'].any():
                a=sender.act(padded(h.sender),torch.as_tensor(valid),padded(obs['queue']),padded(obs['queue_mask'])).numpy()
                tx[obs['sender_active']]=a[:n][obs['sender_active']]
            active=obs['receiver_active']
            if active.any():
                logits=receiver(padded(h.receiver),torch.as_tensor(valid));probs=logits.softmax(-1)[:n][active]
                if receiver_mode=='sampled':rx[active]=torch.multinomial(probs,1,generator=generator).flatten().numpy()
                else:
                    choices=logits.argmax(-1).numpy()[:n];rx[active]=choices[active]
                counts.update(rx[active].tolist());rows+=int(active.sum())
                p=probs.sum(0).numpy();prob_sum=p if prob_sum is None else prob_sum+p
                entropy+=float(torch.distributions.Categorical(probs=probs).entropy().sum())
        obs,_,_,_,_=env.step(tx,rx)
    record=env.metrics();record.pop('messages_detail',None);record.update(seed=seed,domain=label)
    return dict(seed=seed,receiver_mode=receiver_mode,sampling_seed=sampling_seed if receiver_mode=='sampled' else None,
        active_receiver_decisions=rows,applied_receiver_choices=dict(counts),
        mean_receiver_probabilities=(prob_sum/max(1,rows)).tolist() if prob_sum is not None else [],
        mean_receiver_entropy=entropy/max(1,rows),control_effects=dict(env.control_stats),record=record)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output
    if out.exists() and any(p.name!='service.log' for p in out.iterdir()):raise ValueError('new diagnostic output directory required')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    data=json.loads((args.root/'results.json').read_text());plan=data['plan'];checkpoint=args.root/'reactive/final.pt'
    seeds=bank(150000)
    controls=[dict(name=f'{rate}-b{batch}-d{delay}',baseline={**plan['baseline'],'rate':rate,'batch':batch,'delay':delay})
        for rate in ['delay-fast','aimd','bdp'] for batch,delay in [(1,0.),(8,.005),(32,.025)]]
    audit_plan=dict(version='reactive-diagnostics-v1',started_utc=datetime.now(timezone.utc).isoformat(),
        profile='goodput-v6',validation_seeds=plan['validation_seeds'],development_seeds=seeds,controls=controls,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source=json.loads(Path('source-manifest.json').read_text()),
        scope='diagnosis; validation reused only for roles; separate development sweep; no training or reserved tests')
    assert not set(seeds)&set(plan['validation_seeds']);write(out/'plan.json',audit_plan)
    result=dict(plan=audit_plan,roles={},action_audit=[],development={})
    reference=data['modes']['adaptive']['records'];joint=data['modes']['reactive']['records']
    for name,tx,rx in [('sender-only',True,False),('receiver-only',False,True)]:
        write(out/'progress.json',dict(phase='roles',mode=name))
        rows=evaluate(seeds=plan['validation_seeds'],checkpoint=checkpoint,baseline=plan['baseline'],sender_enabled=tx,
            receiver_enabled=rx,environments=2,production=12.,drain=3.,profile='goodput-v6')
        result['roles'][name]=dict(records=rows,summary=summarize(rows),comparison=compare(rows,reference),
            exactly_matches_reference_records=rows==reference,exactly_matches_joint_records=rows==joint)
        write(out/'results.json',result)
    for seed in plan['validation_seeds'][:4]:
        write(out/'progress.json',dict(phase='action-audit',seed=seed))
        audit=action_case(seed,checkpoint,plan['baseline'])
        expected=next(r for r in joint if r['seed']==seed)
        audit['exactly_matches_standard_joint_record']=audit['record']==expected
        if not audit['exactly_matches_standard_joint_record']:raise AssertionError('instrumentation altered evaluation')
        result['action_audit'].append(audit);write(out/'results.json',result)
    for control in controls:
        write(out/'progress.json',dict(phase='development-sweep',control=control['name']))
        rows=evaluate(seeds=seeds,baseline=control['baseline'],environments=2,production=12.,drain=3.,profile='goodput-v6')
        result['development'][control['name']]=dict(baseline=control['baseline'],records=rows,summary=summarize(rows))
        write(out/'results.json',result)
    ref=result['development'][controls[0]['name']]['records']
    for mode in result['development'].values():mode['comparison']=compare(mode['records'],ref)
    result['completed_utc']=datetime.now(timezone.utc).isoformat();write(out/'results.json',result)
    write(out/'progress.json',dict(phase='complete',reserved_tests='unopened'))


if __name__=='__main__':main()
