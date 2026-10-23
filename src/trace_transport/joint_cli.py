"""Installed entry points for persistent-session experiments."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np

from .joint_session import JointTransportEnv,WorkloadConfig,ACK_ACTIONS
from .joint_curriculum import sample_case,DOMAIN
from .simulator import NetworkConfig


def main(argv=None):
    p=argparse.ArgumentParser(prog='trace-transport')
    commands=p.add_subparsers(dest='command',required=True)
    train=commands.add_parser('joint-train')
    train.add_argument('--profile',choices=['deadline-v1','goodput-v2'],default='deadline-v1')
    train.add_argument('--teacher-initial-rate',type=float,default=128000.)
    train.add_argument('--teacher-rate',choices=['fixed','aimd','bdp','delay'],default='fixed')
    train.add_argument('--gae-lambda',type=float,default=.95)
    train.add_argument('--gpu-memory-fraction',type=float,default=.15)
    train.add_argument('--resume',type=Path,help='continue matching optimizer checkpoint; reset collector worlds explicitly')
    train.add_argument('--output',type=Path,default=Path('runs/joint-v1/seed42'))
    train.add_argument('--decisions',type=int,default=5000000)
    train.add_argument('--envs',type=int,default=16)
    train.add_argument('--seed',type=int,default=42)
    train.add_argument('--device',default='cpu')
    train.add_argument('--history',type=int,default=32)
    train.add_argument('--width',type=int,default=64)
    train.add_argument('--layers',type=int,default=2)
    train.add_argument('--rollout',type=int,default=256)
    train.add_argument('--freeze-sender',action='store_true')
    train.add_argument('--freeze-receiver',action='store_true')
    train.add_argument('--easy-fraction',type=float,default=.2)
    train.add_argument('--epochs',type=int,default=4)
    train.add_argument('--checkpoint-every',type=int,default=250000)
    train.add_argument('--production-seconds',type=float,default=20)
    train.add_argument('--drain-seconds',type=float,default=20)
    demo=commands.add_parser('joint-simulate')
    demo.add_argument('--seed',type=int,default=42)
    demo.add_argument('--clients',type=int,default=1)
    demo.add_argument('--production-seconds',type=float,default=2)
    demo.add_argument('--drain-seconds',type=float,default=5)
    demo.add_argument('--report',type=Path,default=Path('runs/joint-demo.json'))
    send=commands.add_parser('send-file')
    send.add_argument('--checkpoint',type=Path,help='experimental joint sender actor; deterministic receiver, single-file distribution shift')
    send.add_argument('--device',default='cpu')
    send.add_argument('--input',type=Path,required=True);send.add_argument('--host',default='127.0.0.1')
    send.add_argument('--port',type=int,default=9001);send.add_argument('--timeout',type=float,default=30)
    send.add_argument('--class',dest='traffic_class',choices=['interactive','image','map','bulk'],default='bulk')
    receive=commands.add_parser('receive-file')
    receive.add_argument('--output',type=Path,required=True);receive.add_argument('--host',default='127.0.0.1')
    receive.add_argument('--port',type=int,default=9001);receive.add_argument('--timeout',type=float,default=30)
    receive.add_argument('--linger',type=float,default=5);receive.add_argument('--ready-json',action='store_true')
    study=commands.add_parser('joint-study')
    study.add_argument('--root',type=Path,default=Path('runs/joint-v1'))
    study.add_argument('--device',default='cpu');study.add_argument('--envs',type=int,default=4)
    study.add_argument('--wait',action='store_true');study.add_argument('--quick',action='store_true')
    probe=commands.add_parser('joint-headroom');probe.add_argument('--steps',type=int,default=5)
    probe.add_argument('--seed',type=int,default=40000);probe.add_argument('--report',type=Path,default=Path('runs/joint-headroom.json'))
    status=commands.add_parser('joint-status')
    status.add_argument('--root',type=Path,default=Path.cwd(),help='repository root containing runs/joint-v1')
    status.add_argument('--json',action='store_true')
    evaluate=commands.add_parser('joint-evaluate')
    evaluate.add_argument('--checkpoint',type=Path)
    evaluate.add_argument('--baseline',default='fixed/edf/budgeted')
    evaluate.add_argument('--seed',type=int,default=50000);evaluate.add_argument('--trials',type=int,default=10)
    evaluate.add_argument('--device',default='cpu');evaluate.add_argument('--envs',type=int,default=4)
    evaluate.add_argument('--trace',type=Path,help='a held-out whole service-trace-v1 recording')
    evaluate.add_argument('--ood',action='store_true');evaluate.add_argument('--report',type=Path,default=Path('runs/joint-evaluation.json'))
    args=p.parse_args(argv)
    if args.command=='joint-status':
        from .joint_monitor import status
        result=status(args.root)
        if args.json:print(json.dumps(result,indent=2))
        else:
            for run in result['runs']:
                progress=run['progress'];print(f"{run['name']:15s} {run['phase']:10s} {progress.get('decisions',0):>9,d} decisions  {progress.get('decisions_per_second',0):.0f}/s")
            print('Study:',result['study'].get('phase','not started'))
        return 0
    if args.command=='joint-evaluate':
        from .joint_evaluation import evaluate,BASELINES,summarize
        if args.trials<1 or args.envs<1:p.error('positive trials/envs required')
        names={f"{b['rate']}/{b['scheduler']}/{b['feedback']}":b for b in BASELINES}
        if args.baseline not in names:p.error('unknown baseline combination')
        trace_network=trace_meta=None
        if args.trace:
            from .joint_traces import load_trace
            trace_network,trace_meta=load_trace(args.trace,required_seconds=40)
        rows=evaluate(seeds=list(range(args.seed,args.seed+args.trials)),checkpoint=args.checkpoint,
            baseline=names[args.baseline],device=args.device,environments=args.envs,ood=args.ood,trace_network=trace_network)
        if rows and rows[0].get('profile')=='goodput-v2':
            from .goodput_evaluation import summarize
        result=dict(summary=summarize(rows),records=rows,trace=trace_meta);args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary'],indent=2));return 0
    if args.command=='joint-study':
        from .joint_study import run
        run(args.root,device=args.device,environments=args.envs,wait=args.wait,quick=args.quick);return 0
    if args.command=='joint-headroom':
        from .joint_evaluation import headroom
        result=headroom(seed=args.seed,steps=args.steps)
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'report':str(args.report),'kind':result['kind']}));return 0
    if args.command in ('send-file','receive-file'):
        from .binary_udp import send_bytes,receive_bytes
        if args.command=='send-file':
            from .messages import MAX_MESSAGE_BYTES
            if args.input.stat().st_size>MAX_MESSAGE_BYTES:p.error('file exceeds the 16-MiB message limit')
            result=send_bytes(args.input.read_bytes(),args.host,args.port,traffic_class=args.traffic_class,timeout=args.timeout,checkpoint=args.checkpoint,device=args.device)
        else:result=receive_bytes(args.host,args.port,args.output,timeout=args.timeout,linger=args.linger,
            ready=(lambda port:print(json.dumps({'ready':True,'port':port}),flush=True)) if args.ready_json else None)
        print(json.dumps(result,indent=2));return 0 if result['completed'] else 2
    if args.command=='joint-train':
        from .joint_training import train as run,JointTrainConfig
        from .joint_models import JointModelConfig
        run(args.output,JointTrainConfig(profile=args.profile,teacher_rate=args.teacher_rate,teacher_initial_rate=args.teacher_initial_rate,gae_lambda=args.gae_lambda,gpu_memory_fraction=args.gpu_memory_fraction,decisions=args.decisions,environments=args.envs,seed=args.seed,device=args.device,
            rollout=args.rollout,epochs=args.epochs,checkpoint_every=args.checkpoint_every,
            production_seconds=args.production_seconds,drain_seconds=args.drain_seconds,easy_fraction=args.easy_fraction,learn_sender=not args.freeze_sender,learn_receiver=not args.freeze_receiver),
            JointModelConfig(history=args.history,width=args.width,layers=args.layers),resume=args.resume)
        return 0
    env=JointTransportEnv(NetworkConfig(feedback_delay_budget=.25),WorkloadConfig(clients=args.clients,
        production_seconds=args.production_seconds,drain_seconds=args.drain_seconds))
    env.reset(seed=args.seed)
    while not env.done():env.step(np.zeros((env.count,3)),np.zeros(env.count,int))
    result=env.metrics();args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('messages_detail','network')},indent=2))
    return 0
