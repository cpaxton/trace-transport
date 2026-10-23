"""CPU event workers; no CUDA or neural-policy imports."""
from multiprocessing import get_context

from .joint_curriculum import sample_case
from .joint_session import JointTransportEnv


def _worker(pipe):
    env=None
    try:
        while True:
            command,args=pipe.recv()
            if command=='close':break
            if command=='reset':
                seed,production,drain,stage,*options=args
                cfg,workload,label=sample_case(seed,production=production,drain=drain,stage=stage,ood=options[0] if options else False)
                if len(options)>2 and options[2] is not None:
                    from .simulator import NetworkConfig
                    cfg=NetworkConfig(**options[2]);label={'family':'heldout-recording','trace_replay':True}
                env=JointTransportEnv(cfg,workload);obs,_=env.reset(seed=seed)
                if len(options)>1:
                    baseline=options[1]
                    for s in env.sessions:
                        s.rate_control=baseline.get('rate','fixed');s.scheduler=baseline.get('scheduler','edf')
                        profile=baseline.get('feedback','budgeted')
                        s.batch,s.delay=(8,.25) if profile=='sparse' else (8,.025)
                        s.ack_mode='deadline' if profile=='deadline' else 'constant'
                        s.baseline_batch,s.baseline_delay,s.baseline_ack_mode=s.batch,s.delay,s.ack_mode
                    obs=env.observe()
                pipe.send((obs,env.critic_state(obs),label))
            elif command=='step':
                obs,reward,done,_,info=env.step(*args)
                metrics=None
                if done:
                    metrics=env.metrics();metrics.pop('messages_detail',None)
                pipe.send((obs,env.critic_state(obs),reward,done,metrics,env.processed_events))
    except EOFError:pass
    except BaseException as exc:
        import traceback
        pipe.send(('ERROR',repr(exc),traceback.format_exc()))
    finally:pipe.close()


class EnvironmentPool:
    def __init__(self,count):
        self.count=count;self.pipes=[];self.processes=[]
        context=get_context('spawn')
        for _ in range(count):
            parent,child=context.Pipe()
            process=context.Process(target=_worker,args=(child,),daemon=True)
            process.start();child.close();self.pipes.append(parent);self.processes.append(process)
    def receive(self,index):
        if not self.pipes[index].poll(300):raise TimeoutError('CPU environment worker stalled')
        result=self.pipes[index].recv()
        if isinstance(result[0],str) and result[0]=='ERROR':raise RuntimeError(result[1]+'\n'+result[2])
        return result
    def reset(self,index,*args):
        self.pipes[index].send(('reset',args));return self.receive(index)
    def send_steps(self,actions):
        for pipe,action in zip(self.pipes,actions):pipe.send(('step',action))
    def collect(self):return [self.receive(i) for i in range(self.count)]
    def close(self):
        for pipe in self.pipes:
            try:pipe.send(('close',None))
            except (BrokenPipeError,EOFError,OSError):pass
        for process in self.processes:
            process.join(timeout=.25)
            if process.is_alive():process.terminate();process.join(timeout=2)
        for pipe in self.pipes:pipe.close()
