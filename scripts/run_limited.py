"""Launch one resource-limited experiment in a user service, outside terminal/tmux.

No shell evaluates the experiment command. Logs go to a file, not the terminal.
Stop only this service with: systemctl --user stop NAME.service
"""
import argparse
from pathlib import Path
import re
import os
import subprocess


def command(name,argv,log,*,directory,cpu=400,memory_gib=8,tasks=128):
    if not re.fullmatch(r'trace-[a-z0-9][a-z0-9-]{0,63}',name):raise ValueError('unit name must start trace- and contain lowercase letters, digits or hyphens')
    if not argv or not 100<=cpu<=800 or cpu%100!=0 or not 1<=memory_gib<=16 or not 16<=tasks<=256:raise ValueError('invalid bounded resources or empty command')
    log=Path(log).resolve();directory=Path(directory).resolve()
    available=sorted(os.sched_getaffinity(0));affinity=available[-min(len(available),cpu//100):]
    return ['systemd-run','--user',f'--unit={name}',f'--working-directory={directory}',
        f'--property=CPUQuota={cpu}%',f'--property=CPUAffinity={" ".join(map(str,affinity))}',f'--property=MemoryMax={memory_gib}G',
        f'--property=MemoryHigh={max(1,memory_gib-2)}G',f'--property=TasksMax={tasks}',
        '--property=MemorySwapMax=512M','--property=Nice=10','--property=OOMPolicy=stop','--property=KillMode=control-group',
        '--property=RuntimeMaxSec=7200','--property=LimitCORE=0',
        f'--property=StandardOutput=append:{log}',f'--property=StandardError=append:{log}',
        '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',
        '--setenv=PYTHONFAULTHANDLER=1','--setenv=PYTHONUNBUFFERED=1','--setenv=PYTHONPATH=','--',*argv]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True);p.add_argument('--log',type=Path,required=True)
    p.add_argument('--cpu',type=int,default=400);p.add_argument('--memory-gib',type=int,default=8)
    p.add_argument('--tasks',type=int,default=128);p.add_argument('--dry-run',action='store_true')
    p.add_argument('argv',nargs=argparse.REMAINDER);args=p.parse_args()
    argv=args.argv[1:] if args.argv[:1]==['--'] else args.argv
    cmd=command(args.name,argv,args.log,directory=Path.cwd(),cpu=args.cpu,memory_gib=args.memory_gib,tasks=args.tasks)
    if args.dry_run:
        import shlex
        print(shlex.join(cmd));return
    # A study gets one unit; never start a duplicate unit with the same name.
    active=subprocess.run(['systemctl','--user','is-active',args.name],capture_output=True,text=True)
    if active.stdout.strip() in ('active','activating','reloading'):raise SystemExit('experiment unit already active')
    running=subprocess.run(['systemctl','--user','list-units','--type=service','--state=running','--no-legend','--plain','--no-pager','trace-*'],capture_output=True,text=True,check=True)
    if running.stdout.strip():raise SystemExit('another trace experiment service is running; serialize jobs')
    args.log.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(cmd,check=True)
    print(f'Logs: {args.log}\nStatus: systemctl --user status {args.name}\nStop: systemctl --user stop {args.name}')


if __name__=='__main__':main()
