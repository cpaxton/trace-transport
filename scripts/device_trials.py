"""Exact UDP transfer trials to an installed Trace Transport endpoint via SSH.

Requires SSH authentication already configured (key or ControlMaster), and a
trace-transport installation on the receiver. Never moves or configures a robot.
Remote payload files are retained in /tmp for inspection, with unique names.
"""
import argparse
import hashlib
import json
from pathlib import Path
from queue import Queue, Empty
import shlex
import subprocess
import threading
import time
import uuid

from trace_transport.benchmark import benchmark_text
from trace_transport.udp import send_udp


def remote_listener_command(path,deadline):
    args=['trace-transport','listen','--host','0.0.0.0','--port','0','--output',path,
          '--timeout',str(deadline+5),'--linger','1','--receipt-feedback','--ready-json']
    return 'PATH="$HOME/.local/bin:$PATH" exec '+shlex.join(args)


def wait_ready(stream,timeout):
    queue=Queue()
    def read():
        try:
            queue.put(stream.readline())
        except Exception as exc:
            queue.put(exc)
    threading.Thread(target=read,daemon=True).start()
    try:
        line=queue.get(timeout=timeout)
    except Empty:
        raise TimeoutError('SSH listener did not become ready') from None
    if isinstance(line,Exception):
        raise line
    data=json.loads(line)
    if data.get('ready') is not True or type(data.get('port')) is not int:
        raise ValueError('unexpected listener readiness response')
    return data['port']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh',required=True,help='SSH target, e.g. jetson1@herman or caliban')
    parser.add_argument('--address',required=True,help='UDP-reachable IP/DNS name; SSH aliases may not resolve for UDP')
    parser.add_argument('--condition',required=True,choices=['ethernet','wifi-stationary','wifi-mobile'])
    parser.add_argument('--trials',type=int,default=10)
    parser.add_argument('--bytes',type=int,default=131072)
    parser.add_argument('--deadline',type=float,default=20)
    parser.add_argument('--controller',choices=['fixed','aimd','learned'],default='aimd')
    parser.add_argument('--checkpoint')
    parser.add_argument('--interface',help='optional remote Wi-Fi interface to snapshot using iw')
    parser.add_argument('--report',type=Path,default=Path('runs/devices/trials.json'))
    args=parser.parse_args()
    if args.ssh.startswith('-') or args.trials<1 or not 1<=args.bytes<=16*1024*1024 or not 0<args.deadline<=3600:
        parser.error('invalid target, trials, payload size, or deadline')
    if args.controller=='learned' and not args.checkpoint:
        parser.error('learned controller requires --checkpoint')
    text=benchmark_text(args.bytes)
    expected=hashlib.sha256(text.encode()).hexdigest()
    report=dict(target=args.ssh,address=args.address,condition=args.condition,controller=args.controller,
                checkpoint=args.checkpoint,payload_bytes=len(text.encode()),sha256=expected,trials=[])
    def telemetry():
        if not args.interface:
            return None
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2',args.ssh,
            shlex.join(['iw','dev',args.interface,'link'])],capture_output=True,text=True,timeout=10)
        return dict(returncode=result.returncode,output=result.stdout.strip(),error=result.stderr.strip())
    report['wifi_before']=telemetry()
    for trial in range(args.trials):
        path=f'/tmp/trace-transport-{uuid.uuid4().hex}.txt'
        process=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2',args.ssh,
            remote_listener_command(path,args.deadline)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        row=dict(trial=trial,remote_output=path)
        start=time.monotonic()
        try:
            port=wait_ready(process.stdout,10)
            result=send_udp(text,args.address,port,deadline=args.deadline,
                            controller=args.controller,checkpoint=args.checkpoint)
            remaining,ssh_error=process.communicate(timeout=args.deadline+10)
            row['ssh_error']=ssh_error.strip()
            receiver=json.loads(remaining)
            row.update(sender=result.summary(),receiver=receiver,
                       exact_confirmed=result.completed and receiver.get('sha256')==expected,
                       wall_seconds=time.monotonic()-start)
        except Exception as exc:
            row.update(exact_confirmed=False,error=str(exc),wall_seconds=time.monotonic()-start)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            _,ssh_error=process.communicate(timeout=5)
            row['ssh_returncode']=process.returncode
            row.setdefault('ssh_error',ssh_error.strip())
        report['trials'].append(row)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        print(f'{trial+1}/{args.trials}: exact confirmed={row["exact_confirmed"]}',flush=True)
    report['wifi_after']=telemetry()
    report['completed']=sum(row['exact_confirmed'] for row in report['trials'])
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['completed']==args.trials else 2


if __name__=='__main__':
    raise SystemExit(main())
