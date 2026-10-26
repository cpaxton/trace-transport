"""Deploy an immutable source bundle to a pre-bootstrapped Caliban runtime.

No system installs, robot commands, credentials, or arbitrary cleanup. Bootstrap
requirements and reboot recovery are documented in docs/experiments/caliban-v4.md.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import zipfile


def remote(ssh,argv,*,cwd=None):
    command=shlex.join([str(x) for x in argv])
    if cwd:command='cd '+shlex.quote(str(cwd))+' && '+command
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',ssh,command],
                          check=True,capture_output=True,text=True,timeout=180).stdout


def package(root,archive):
    paths=[root/p for p in ['pyproject.toml','uv.lock','README.md']]
    for pattern in ['src/**/*.py','scripts/*.py','tests/*.py']:paths.extend(root.glob(pattern))
    hashes={}
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(paths)):
            rel=path.relative_to(root).as_posix();data=path.read_bytes()
            hashes[rel]=hashlib.sha256(data).hexdigest();z.writestr(rel,data)
        z.writestr('source-manifest.json',json.dumps(hashes,sort_keys=True,indent=2)+'\n')
    return hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['deploy','benchmark','study','status'])
    p.add_argument('--source-slot',default='source')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--ssh',default='caliban');p.add_argument('--decisions',type=int,default=5000000)
    args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',args.ssh):p.error('use a configured SSH alias')
    if not re.fullmatch(r'source(?:-[a-z0-9-]+)?',args.source_slot):p.error('invalid immutable source slot')
    info=json.loads(remote(args.ssh,['python3','-c','import os,json; from pathlib import Path; print(json.dumps(dict(uid=os.getuid(),home=str(Path.home()))))']))
    runtime=Path(f'/run/user/{info["uid"]}/trace-transport-caliban-20261024');source=runtime/args.source_slot
    python=runtime/'.venv/bin/python';uv=runtime/'bootstrap/bin/uv';runs=runtime/'runs'
    if args.stage=='deploy':
        with tempfile.TemporaryDirectory(prefix='trace-caliban-') as temporary:
            archive=Path(temporary)/'source.zip';package(Path(__file__).resolve().parents[1],archive)
            subprocess.run(['scp','-q','-o','BatchMode=yes',str(archive),f'{args.ssh}:{runtime}/{args.source_slot}.zip'],check=True)
        code='''import json,hashlib,zipfile,shutil,sys
from pathlib import Path
runtime=Path(sys.argv[1]);source=runtime/sys.argv[2]
if source.exists():raise RuntimeError('immutable source directory already exists; do not overwrite a running study')
with zipfile.ZipFile(runtime/(sys.argv[2]+'.zip')) as z:
 if any(Path(n).is_absolute() or '..' in Path(n).parts for n in z.namelist()):raise ValueError('unsafe archive')
 z.extractall(source)
manifest=json.loads((source/'source-manifest.json').read_text())
if any(hashlib.sha256((source/p).read_bytes()).hexdigest()!=h for p,h in manifest.items()):raise ValueError('source hash mismatch')
(runtime/'runs').mkdir(exist_ok=True)
shutil.copyfile(source/'source-manifest.json',runtime/'runs/source-manifest.json')
'''
        remote(args.ssh,['python3','-c',code,runtime,args.source_slot])
        print(remote(args.ssh,['env',f'TMPDIR={runtime}',uv,'pip','install','--no-cache','--python',python,'--no-deps','-e',source]))
        print(remote(args.ssh,['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',python,'-m','pytest','-q','tests/test_goodput_deployment.py','tests/test_run_limited.py'],cwd=source))
    elif args.stage in ('benchmark','study'):
        stage=args.stage;name='trace-caliban-v4-'+stage
        argv=[python,'-m','scripts.run_limited','--name',name,'--log',runs/stage/'service.log',
              '--cpu','600','--memory-gib','8','--runtime-hours','72' if stage=='study' else '2','--',
              python,'-m','scripts.caliban_study',stage,'--root',runs/stage]
        if stage=='study':argv.extend(['--persistent',Path(info['home'])/'trace-transport-studies/caliban-v4-20261024','--decisions',str(args.decisions)])
        if args.resume:argv.append('--resume')
        print(remote(args.ssh,argv,cwd=source))
    else:
        code='''import json,sys,subprocess,time
from pathlib import Path
root=Path(sys.argv[1]);out={}
for stage in ['benchmark','study','selector-coverage']:
 r=root/stage;out[stage]={}
 unit='trace-selector-coverage' if stage=='selector-coverage' else 'trace-caliban-v4-'+stage
 state=subprocess.run(['systemctl','--user','show',unit,'--property=ActiveState,SubState,MemoryCurrent,TasksCurrent,CPUAffinity'],capture_output=True,text=True)
 out[stage]['service']=dict(line.split('=',1) for line in state.stdout.splitlines() if '=' in line)
 if (r/'plan.json').exists():
  data=json.loads((r/'plan.json').read_text());keys=['version','started_utc','splits','scope'] if stage=='selector-coverage' else ['version','started_utc','training_seeds','decisions_per_seed','environments'];out[stage]['plan']={k:data.get(k) for k in keys}
 if (r/'calibration.json').exists():out[stage]['calibration_controls_completed']=len(json.loads((r/'calibration.json').read_text()))
 for name in ['results.json','progress.json','study-failure.json','failure.json','reference.json']:
  if (r/name).exists():
   data=json.loads((r/name).read_text())
   if stage=='selector-coverage' and name=='results.json':
    out[stage]['completed_by_split']={k:dict(probes=len(v.get('probes',[])),controls=len(v.get('controls',{}))) for k,v in data.get('splits',{}).items()}
   else:out[stage][name]=data
 for path in r.glob('seed*/progress.json'):
  data=json.loads(path.read_text());entry={k:data.get(k) for k in ['phase','decisions','decisions_per_second','elapsed_seconds']}
  entry['heartbeat_age_seconds']=time.time()-path.stat().st_mtime
  if entry['phase']=='training' and out[stage]['service'].get('ActiveState')!='active':entry['phase']='stopped; last recorded training'
  out[stage][path.parent.name]=entry
print(json.dumps(out,indent=2))'''
        print(remote(args.ssh,[python,'-c',code,runs]))


if __name__=='__main__':main()
