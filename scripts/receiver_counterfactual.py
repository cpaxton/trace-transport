"""Compare legal ACK interventions from identical cloned v4 environment states.

Uses hidden simulation state to generate diagnostic labels. This is not a
runtime oracle or held-out learned-policy evaluation. Policy inputs remain local.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from trace_transport.experiment_io import write
from trace_transport.goodput_deployment import sample_case,GoodputDeploymentEnv
from trace_transport.joint_session import ACK_ACTIONS


def compare_actions(env,index,*,horizon=1.,choices=None):
    choices=list(range(len(ACK_ACTIONS)+1)) if choices is None else list(choices)
    before=env.reward_counts.copy();verified=sum(env.verified[c] for c in ('image','map'))
    output=[]
    for choice in choices:
        trial=env.clone();end=min(trial.horizon,trial.now+horizon);reward=0.
        tx=np.zeros((env.count,3));rx=np.zeros(env.count,dtype=int);rx[index]=choice
        while trial.now<end-1e-9:
            _,r,_,_,_=trial.step(tx,rx);reward+=r
        output.append(dict(choice=choice,ack=list(ACK_ACTIONS[choice-1]) if choice else 'teacher',
                           reward=reward,verified_bytes=sum(trial.verified[c] for c in ('image','map'))-verified,
                           on_time=trial.reward_counts['on_time']-before['on_time'],
                           missed=trial.reward_counts['missed']-before['missed'],
                           wire=trial.reward_counts['wire']-before['wire'],elapsed=trial.now-env.now))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/receiver-counterfactual-v4/results.json'))
    args=p.parse_args();result=dict(version='receiver-counterfactual-v4-v1',started_utc=datetime.now(timezone.utc).isoformat(),
        caveat='finite-horizon simulator counterfactual; cloned RNG streams, action-dependent packet draws; not causal real-path evidence or deployable oracle',cases=[])
    for family in ['reverse-congestion','capacity-opportunities','wifi-contention','propagation-step']:
        for seed in [131000,131001]:
            cfg,w,label=sample_case(seed,production=3.,drain=1.,family=family)
            env=GoodputDeploymentEnv(cfg,w);obs,_=env.reset(seed=seed)
            for s in env.sessions:
                s.rate_control='delay-fast';s.baseline_batch=s.batch=16;s.baseline_delay=s.delay=.025
            tx=np.zeros((env.count,3));rx=np.zeros(env.count,int)
            while env.now<1.8:
                obs,_,_,_,_=env.step(tx,rx)
                if env.now>=.5 and any(s.pending for s in env.sessions):break
            pending=[i for i,s in enumerate(env.sessions) if s.pending]
            if not pending:
                result['cases'].append(dict(seed=seed,family=family,status='no pending ACK state found'));continue
            index=pending[0];rows=compare_actions(env,index)
            teacher=rows[0];best=max(rows,key=lambda r:r['reward'])
            result['cases'].append(dict(seed=seed,family=family,status='complete',time=env.now,index=index,
                control_interval=cfg.control_interval,receiver_observation=obs['receiver'][index].tolist(),
                actions=rows,best_choice=best['choice'],best_reward_gain=best['reward']-teacher['reward']))
            write(args.output,result)
    result['completed_utc']=datetime.now(timezone.utc).isoformat();write(args.output,result)


if __name__=='__main__':main()
