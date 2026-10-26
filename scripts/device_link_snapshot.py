"""Bounded read-only link inventory and idle ping; never estimates capacity.

Uses existing SSH keys only. Writes local ignored run data, not credentials or
SSID/MAC/IP inventories. Does not configure devices or start saturation traffic.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import shlex
import subprocess

DEVICES={'herman':'jetson1@herman','caliban':'caliban'}
REMOTE = '''import json,pathlib,subprocess
r=subprocess.run(['ip','-j','route','get','1.1.1.1'],capture_output=True,text=True,check=True)
interface=json.loads(r.stdout)[0]['dev']
p=pathlib.Path('/sys/class/net')/interface
out={'medium':'wifi' if (p/'wireless').exists() else 'ethernet-or-other'}
try:out['reported_link_speed_mbps']=int((p/'speed').read_text())
except (OSError,ValueError):pass
if out['medium']=='wifi':
 try:
  x=subprocess.run(['iw','dev',interface,'link'],capture_output=True,text=True,timeout=3)
  out['wireless_link_summary']=[line.strip() for line in x.stdout.splitlines() if any(k in line for k in ['signal:','tx bitrate:','rx bitrate:'])]
 except (OSError,subprocess.TimeoutExpired):pass
print(json.dumps(out))'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/device-link-snapshot-v1.json'))
    args=p.parse_args();result=dict(started_utc=datetime.now(timezone.utc).isoformat(),
        purpose='idle engineering inventory, not capacity or controller performance',devices={})
    for name,target in DEVICES.items():
        row={}
        try:
            x=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',target,
                              'python3 -c '+shlex.quote(REMOTE)],capture_output=True,text=True,timeout=12)
            row['link']=json.loads(x.stdout) if x.returncode==0 else {'status':'ssh inventory unavailable','exit_code':x.returncode}
        except (OSError,ValueError,subprocess.TimeoutExpired) as e:row['link']={'status':'unavailable','reason':type(e).__name__}
        try:
            config=subprocess.run(['ssh','-G',target],capture_output=True,text=True,timeout=5)
            hostname=next((line.split(' ',1)[1] for line in config.stdout.splitlines() if line.startswith('hostname ')),name)
            x=subprocess.run(['ping','-n','-c','10','-i','0.2','-W','1',hostname],capture_output=True,text=True,timeout=15)
            loss=re.search(r'([\d.]+)% packet loss',x.stdout)
            timing=re.search(r'= ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms',x.stdout)
            row['ping']=dict(packet_loss_percent=float(loss[1]) if loss else None,
                            rtt_ms=dict(zip(['min','mean','max','mdev'],map(float,timing.groups()))) if timing else None,
                            packets=10,exit_code=x.returncode)
        except (OSError,subprocess.TimeoutExpired) as e:row['ping']={'status':'unavailable','reason':type(e).__name__}
        result['devices'][name]=row
    result['completed_utc']=datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
