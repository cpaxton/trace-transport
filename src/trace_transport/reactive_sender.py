"""Versioned producer-only target-rate bias and physical-time observations.

The v6 task and verifier are unchanged. This adapter is not a v6 checkpoint
format: reverse-role control, receiver ACKs and window control remain teachers.
"""
from collections import deque
from dataclasses import replace
import copy
import math
from multiprocessing import get_context
import numpy as np
from .goodput_reactive import GoodputReactiveEnv,sample_case
from .joint_session import SENDER_FEATURES

BIASES=(.8,1.,1.2)
KEEP=-1
FEATURES=(*SENDER_FEATURES,'log_target_bias','decision_hold')
CRITIC_EXTRA=('mean_log_bias','mean_hold_remaining','mean_pending_bias','eligible_fraction')
RATE_SCALES={'ack_rate':128000.,'wire_rate':128000.,'loss_rate':16384.}
CONTRACT=dict(version='reactive-sender-v7',task='goodput-v6',roles='even-index bulk producers only',
    biases=list(BIASES),window_teacher=True,receiver_teacher=True,macro_seconds=.1,
    history_seconds=1.,bins=64,inference_delay_seconds=.02,initial_teacher_prior=.6,max_hold_seconds=.5,
    note='new policy schema, not a retagged joint checkpoint; raw whole-object task reward')


class PhysicalHistory:
    def __init__(self,initial,*,seconds=1.,bins=64):
        if seconds<=0 or bins<1:raise ValueError('invalid history window')
        initial=np.asarray(initial,np.float32)
        self.seconds=seconds;self.bins=bins;self.count=len(initial)
        self.frames=deque([(0.,0.,initial.copy(),np.zeros((self.count,len(RATE_SCALES))))])
    def add(self,left,right,state,*,rate_window):
        state=np.asarray(state,np.float32)
        if left!=self.frames[-1][1] or right<=left or rate_window<=0 or not np.isfinite(state).all():raise ValueError('invalid history transition')
        rates=np.stack([np.expm1(state[:,SENDER_FEATURES.index(name)])*scale*rate_window/(right-left)
            for name,scale in RATE_SCALES.items()],-1)
        self.frames.append((left,right,state.copy(),rates))
        while len(self.frames)>1 and self.frames[1][1]<=right-self.seconds:self.frames.popleft()
    def snapshot(self):
        frames=list(self.frames);now=frames[-1][1];width=self.seconds/self.bins
        right=now-self.seconds+np.arange(1,self.bins+1)*width;left=right-width
        ends=np.asarray([f[1] for f in frames]);starts=np.asarray([f[0] for f in frames])
        ids=np.searchsorted(ends,right+1e-10,side='right')-1;valid=(ids>=0)&(right>0)
        states=np.stack([f[2] for f in frames]);out=states[np.maximum(ids,0)].transpose(1,0,2).copy()
        weights=np.maximum(0,np.minimum(right[:,None],ends)-np.maximum(left[:,None],starts))
        rates=np.einsum('bt,tnf->nbf',weights,np.stack([f[3] for f in frames]))/width
        for i,(name,scale) in enumerate(RATE_SCALES.items()):out[:,:,SENDER_FEATURES.index(name)]=np.log1p(rates[:,:,i]/scale)
        if now==0:valid[-1]=True;out[:,-1]=states[-1]
        out[:,~valid]=0
        return out.astype(np.float32),np.broadcast_to(valid,(self.count,self.bins)).copy()


class TargetBiasEnv(GoodputReactiveEnv):
    def reset(self,**kwargs):
        result=super().reset(**kwargs);self.target_bias=np.ones(self.count);return result
    def baseline(self,index,rate_control='fixed',scheduler='edf',feedback='budgeted'):
        a=super().baseline(index,rate_control,scheduler,feedback)
        bias=self.target_bias[index];s=self.sessions[index]
        if index%2==0 and bias!=1 and rate_control=='delay-fast' and s.last_feedback is not None:
            target=min(self.rate_limit,max(2000.,self.network.payload_size/(.5*max(.000001,s.srtt-s.min_rtt))))
            target=min(self.rate_limit,max(2000.,target*bias))
            a[0]=np.clip(math.log(target/s.rate)/.2,-1,1)
        return a


class MacroWorld:
    def __init__(self,seed,*,production=12.,drain=3.,inference_delay=.02):
        if not math.isfinite(inference_delay) or inference_delay<0:raise ValueError('invalid inference delay')
        cfg,w,self.label=sample_case(seed,production=production,drain=drain)
        self.env=TargetBiasEnv(replace(cfg,initial_rate=512000.),w,churn=self.label.get('churn',False),churn_times=self.label.get('churn_times'))
        self.env.reset(seed=seed);self.delay=inference_delay;self.hold=np.full(self.env.count,CONTRACT['macro_seconds'])
        self.next_decision=np.zeros(self.env.count);self.pending=[];self.kernel_steps=0
        for s in self.env.sessions:
            s.rate_control='delay-fast';s.scheduler='edf';s.batch=s.baseline_batch=1;s.delay=s.baseline_delay=0.
            s.ack_mode=s.baseline_ack_mode='constant'
        self.obs=self.env.observe();self.history=PhysicalHistory(self.features(),seconds=CONTRACT['history_seconds'],bins=CONTRACT['bins'])
    def clone(self):return copy.deepcopy(self)
    def features(self):
        return np.concatenate([self.obs['sender'],np.log(self.env.target_bias)[:,None],np.log1p(self.hold/CONTRACT['macro_seconds'])[:,None]],-1)
    def observation(self):
        h,v=self.history.snapshot();mask=(np.arange(self.env.count)%2==0)&self.obs['sender_active']&(self.env.now+1e-9>=self.next_decision)
        producer=np.arange(self.env.count)%2==0
        extra=[float(np.log(self.env.target_bias[producer]).mean()),float(np.maximum(0,self.next_decision[producer]-self.env.now).mean()),
            float(np.mean([math.log(BIASES[c]) for _,_,c in self.pending])) if self.pending else 0.,float(mask[producer].mean())]
        return dict(history=h,valid=v,eligible=mask,critic=np.concatenate([self.env.critic_state(self.obs),extra]).astype(np.float32))
    def step(self,choices):
        choices=np.asarray(choices)
        if choices.shape!=(self.env.count,) or not np.isfinite(choices).all() or np.any(choices!=choices.astype(int)) or np.any((choices<KEEP)|(choices>=len(BIASES))):raise ValueError('invalid macro actions')
        eligible=self.observation()['eligible'];selected=choices!=KEEP
        if np.any(selected&~eligible):raise ValueError('action on ineligible/reverse role')
        for i in np.flatnonzero(selected):
            self.hold[i]=min(CONTRACT['max_hold_seconds'],max(CONTRACT['macro_seconds'],2*self.env.sessions[i].srtt))
            self.next_decision[i]=self.env.now+self.hold[i];self.pending.append((self.env.now+self.delay,int(i),int(choices[i])))
        start=self.env.now;end=min(self.env.horizon,start+CONTRACT['macro_seconds']);reward=raw=0.;steps=0
        zero=(np.zeros((self.env.count,3)),np.zeros(self.env.count,int))
        while self.env.now<end-1e-9 and not self.env.done():
            waiting=[]
            for due,i,choice in self.pending:
                if due<=self.env.now+1e-9:self.env.target_bias[i]=BIASES[choice]
                else:waiting.append((due,i,choice))
            self.pending=waiting
            before=self.env.now
            self.obs,r,_,_,_=self.env.step(*zero)
            reward+=.999**((before-start)/.02)*r;raw+=r;steps+=1
            self.history.add(before,self.env.now,self.features(),rate_window=self.env.network.control_interval)
        self.kernel_steps+=steps;done=self.env.done();metrics=None
        if done:
            metrics=self.env.metrics();metrics.pop('messages_detail',None);metrics.update(seed=self.label['seed'],domain=self.label)
        return self.observation(),float(reward),done,metrics,self.env.now-start,steps,float(raw)


def _worker(pipe):
    try:
        world=None
        while True:
            command,args=pipe.recv()
            if command=='close':break
            if command=='reset':world=MacroWorld(*args);pipe.send(world.observation())
            elif command=='step':pipe.send(world.step(args))
    except BaseException as exc:
        import traceback
        try:pipe.send(('ERROR',repr(exc),traceback.format_exc()))
        except (BrokenPipeError,EOFError):pass
    finally:pipe.close()


class MacroPool:
    def __init__(self,count):
        self.pipes=[];self.processes=[];ctx=get_context('spawn')
        for _ in range(count):
            parent,child=ctx.Pipe();process=ctx.Process(target=_worker,args=(child,),daemon=True);process.start();child.close()
            self.pipes.append(parent);self.processes.append(process)
    def receive(self,i):
        if not self.pipes[i].poll(300):raise TimeoutError('macro worker stalled')
        result=self.pipes[i].recv()
        if isinstance(result,tuple) and isinstance(result[0],str) and result[0]=='ERROR':raise RuntimeError(result[1]+'\n'+result[2])
        return result
    def reset(self,i,seed):self.pipes[i].send(('reset',(int(seed),)));return self.receive(i)
    def close(self):
        for pipe in self.pipes:
            try:pipe.send(('close',None))
            except (BrokenPipeError,EOFError):pass
        for p in self.processes:
            p.join(timeout=.5)
            if p.is_alive():p.terminate();p.join(timeout=2)
        for pipe in self.pipes:pipe.close()
