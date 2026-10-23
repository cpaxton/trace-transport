"""Bounded synthetic binary transfers and raw RTT observations over SSH.

Copies a source-only ZIP to a unique remote /tmp directory; requires Python 3.10+,
not a system installation. Does not move robots or change network settings.
RTT probes are observations, not estimates of available link capacity.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import shlex
import struct
import subprocess
import tempfile
import time
import uuid
import zipfile
import zlib

from trace_transport.binary_udp import send_bytes
from scripts.device_trials import wait_ready


def png_payload():
    def chunk(kind,data):
        return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
    rng=random.Random(42)
    pixels=b''.join(b'\0'+rng.randbytes(256*3) for _ in range(256))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!IIBBBBB',256,256,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(pixels))+chunk(b'IEND',b'')


def source_bundle(root,destination):
    # Explicit source-only selection excludes credentials, checkpoints and local runs.
    files=['__init__.py','messages.py','binary_udp.py','session_receiver.py','protocol.py','wire.py','simulator.py']
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(Path(root)/'src/trace_transport'/name,'trace_transport/'+name)
    return hashlib.sha256(Path(destination).read_bytes()).hexdigest()


def ping_observations(address,count,interval):
    result=subprocess.run(['ping','-n','-c',str(count),'-i',str(interval),'-W','1',address],
        capture_output=True,text=True,timeout=count*interval+count+5)
    return dict(kind='raw-ping-observations-v1',count=count,interval_seconds=interval,
        returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
        caveat='ICMP round trips and loss only; not calibrated service capacity or UDP loss.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,help='experimental sender-only joint checkpoint on the local host')
    p.add_argument('--ssh',required=True);p.add_argument('--address',required=True)
    p.add_argument('--condition',required=True,choices=['ethernet','wifi-stationary','wifi-mobile'])
    p.add_argument('--trials',type=int,default=3);p.add_argument('--probe-count',type=int,default=80)
    p.add_argument('--probe-interval',type=float,default=.5)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    if args.ssh.startswith('-') or args.address.startswith('-') or not 1<=args.trials<=20 or not 1<=args.probe_count<=240 or not .2<=args.probe_interval<=2:
        p.error('invalid target or bounded trial/probe settings')
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2',args.ssh]
    remote='/tmp/trace-binary-'+uuid.uuid4().hex
    report=dict(format_version='device-binary-observations-v1',condition=args.condition,
        target=args.ssh,started_unix=time.time(),controller='joint-sender-single-file' if args.checkpoint else 'deterministic-reliability',trials=[],
        caveat='Synthetic payloads, sequential transfers; no learned live-session deployment or available-bandwidth estimate.')
    if args.checkpoint:
        report['checkpoint_sha256']=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    def save():
        args.report.parent.mkdir(parents=True,exist_ok=True)
        temp=args.report.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.report)
    root=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='trace-device-') as tmp, ThreadPoolExecutor(max_workers=1) as pool:
        archive=Path(tmp)/'source.zip';report['source_sha256']=source_bundle(root,archive)
        subprocess.run(ssh+[shlex.join(['mkdir','-p',remote])],check=True,timeout=15)
        subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=5',str(archive),args.ssh+':'+remote+'/source.zip'],check=True,timeout=30)
        probes=pool.submit(ping_observations,args.address,args.probe_count,args.probe_interval)
        payloads={'interactive':('Trace Transport exact Unicode: 🌐\n'*32).encode(),
                  'image':png_payload(),'map':random.Random(43).randbytes(512*1024)}
        for trial in range(args.trials):
            for traffic,payload in payloads.items():
                output=remote+'/'+str(trial)+'-'+traffic+'.bin'
                code=('import json;from trace_transport.binary_udp import receive_bytes;'
                    f'r=receive_bytes("0.0.0.0",0,{output!r},timeout=25,linger=1,'
                    'ready=lambda port:print(json.dumps(dict(ready=True,port=port)),flush=True));'
                    'print(json.dumps(r),flush=True)')
                command=f'PYTHONPATH={shlex.quote(remote+"/source.zip")} python3 -u -c '+shlex.quote(code)
                process=subprocess.Popen(ssh+[command],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                row=dict(trial=trial,traffic_class=traffic,payload_bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
                try:
                    port=wait_ready(process.stdout,10)
                    row['sender']=send_bytes(payload,args.address,port,traffic_class=traffic,timeout=20,checkpoint=args.checkpoint)
                    out,err=process.communicate(timeout=30);row['receiver']=json.loads(out)
                    row['exact_confirmed']=bool(process.returncode==0 and row['sender']['receiver_confirmed'] and row['receiver']['completed'] and row['receiver']['sha256']==row['sha256'])
                    if err:row['stderr']=err
                except Exception as exc:row.update(exact_confirmed=False,error=str(exc))
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:process.communicate(timeout=5)
                        except subprocess.TimeoutExpired:process.kill();process.communicate(timeout=5)
                report['trials'].append(row);save()
                print(trial,traffic,row['exact_confirmed'],flush=True)
        report['rtt_observations']=probes.result()
        report['completed']=sum(row['exact_confirmed'] for row in report['trials'])
        report['finished_unix']=time.time();save()
    return 0 if report['completed']==len(report['trials']) else 2


if __name__=='__main__':raise SystemExit(main())
