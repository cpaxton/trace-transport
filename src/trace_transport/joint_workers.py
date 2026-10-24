"""CPU event workers; no CUDA or neural-policy imports."""
from multiprocessing import get_context

from .joint_curriculum import sample_case
from .joint_session import JointTransportEnv


def _worker(pipe):
    import faulthandler
    faulthandler.enable()
    env=None;teacher_env=None;raw_return=teacher_return=0.
    try:
        while True:
            command,args=pipe.recv()
            if command=='close':break
            if command=='reset':
                seed,production,drain,stage,*options=args
                task=options[3] if len(options)>3 else 'deadline-v1'
                if task=='goodput-v2':
                    from .goodput import sample_case as sampler,GoodputEnv
                elif task=='deadline-v1':sampler=sample_case
                else:raise ValueError('unknown experiment profile')
                cfg,workload,label=sampler(seed,production=production,drain=drain,stage=stage,ood=options[0] if options else False)
                if len(options)>2 and options[2] is not None:
                    from .simulator import NetworkConfig
                    cfg=NetworkConfig(**options[2]);label={'family':'heldout-recording','trace_replay':True}
                if len(options)>1 and 'initial_rate' in options[1]:
                    from dataclasses import replace
                    cfg=replace(cfg,initial_rate=options[1]['initial_rate'])
                env=GoodputEnv(cfg,workload,churn=label.get('churn',False),churn_times=label.get('churn_times')) if task=='goodput-v2' else JointTransportEnv(cfg,workload)
                obs,_=env.reset(seed=seed)
                if len(options)>1:
                    baseline=options[1]
                    for s in env.sessions:
                        s.rate_control=baseline.get('rate','fixed');s.scheduler=baseline.get('scheduler','edf')
                        profile=baseline.get('feedback','budgeted')
                        s.batch,s.delay=(8,.25) if profile=='sparse' else (8,.025)
                        s.batch=baseline.get('batch',s.batch);s.delay=baseline.get('delay',s.delay)
                        s.ack_mode='deadline' if profile=='deadline' else 'constant'
                        s.baseline_batch,s.baseline_delay,s.baseline_ack_mode=s.batch,s.delay,s.ack_mode
                    obs=env.observe()
                paired=bool(options[4]) if len(options)>4 else False
                if paired and task!='goodput-v2':raise ValueError('paired teacher requires fixed-horizon goodput task')
                teacher_env=env.clone() if paired else None;raw_return=teacher_return=0.
                pipe.send((obs,env.critic_state(obs),label))
            elif command=='finish-baseline':
                import numpy as np
                actions=(np.zeros((env.count,3),np.float32),np.zeros(env.count,np.int64))
                while not env.done():env.step(*actions)
                metrics=env.metrics();metrics.pop('messages_detail',None)
                pipe.send(metrics)
            elif command=='step':
                obs,reward,done,_,info=env.step(*args)
                raw_return+=reward
                if teacher_env is not None:
                    import numpy as np
                    _,reference_reward,reference_done,_,_=teacher_env.step(np.zeros((env.count,3)),np.zeros(env.count,dtype=int))
                    if reference_done!=done:raise RuntimeError('teacher/learner horizons diverged')
                    teacher_return+=reference_reward;reward-=reference_reward
                metrics=None
                if done:
                    metrics=env.metrics();metrics.pop('messages_detail',None)
                    if teacher_env is not None:metrics.update(raw_training_return=raw_return,teacher_training_return=teacher_return)
                pipe.send((obs,env.critic_state(obs),reward,done,metrics,env.processed_events))
    except EOFError:pass
    except BaseException as exc:
        import traceback
        pipe.send(('ERROR',repr(exc),traceback.format_exc()))
    finally:pipe.close()


class EnvironmentPool:
    def __init__(self,count):
        self.count=count;self.pipes=[];self.processes=[];self.context=[None]*count
        context=get_context('spawn')
        for _ in range(count):
            parent,child=context.Pipe()
            process=context.Process(target=_worker,args=(child,),daemon=True)
            process.start();child.close();self.pipes.append(parent);self.processes.append(process)
    def receive(self,index):
        if not self.pipes[index].poll(300):raise TimeoutError('CPU environment worker stalled')
        try:result=self.pipes[index].recv()
        except EOFError as exc:
            self.processes[index].join(timeout=.1)
            raise RuntimeError(f'CPU worker {index} disconnected; exitcode={self.processes[index].exitcode}; context={self.context[index]}') from exc
        if isinstance(result,(tuple,list)) and result and isinstance(result[0],str) and result[0]=='ERROR':raise RuntimeError(result[1]+'\n'+result[2])
        return result
    def reset(self,index,*args):
        self.context[index]={'seed':args[0],'steps':0}
        self.pipes[index].send(('reset',args));return self.receive(index)
    def send_step(self,index,action):
        self.context[index]['steps']+=1
        self.pipes[index].send(('step',action))
    def send_steps(self,actions):
        for i,action in enumerate(actions):self.send_step(i,action)
    def collect(self):return [self.receive(i) for i in range(self.count)]
    def close(self):
        for pipe in self.pipes:
            try:pipe.send(('close',None))
            except (BrokenPipeError,EOFError,OSError):pass
        for process in self.processes:
            process.join(timeout=.25)
            if process.is_alive():process.terminate();process.join(timeout=2)
        for pipe in self.pipes:pipe.close()
