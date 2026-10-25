"""Freeze source and a separate pinned CPU runtime for parallel local studies.

The new target is immutable; existing source and runtime directories are never
replaced. No credentials, existing environments or raw runs enter the bundle.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from scripts.caliban_deploy import package


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path)
    p.add_argument('--cpu-runtime',action='store_true',help='explicitly select the default CPU-only runtime')
    args=p.parse_args();out=args.destination.resolve()
    if out.exists():raise ValueError('immutable destination already exists')
    out.mkdir();package(Path(__file__).resolve().parents[1],out/'source.zip')
    with zipfile.ZipFile(out/'source.zip') as z:z.extractall(out)
    env={**os.environ,'UV_CACHE_DIR':'/tmp/trace-uv-cache','TMPDIR':str(out)}
    subprocess.run(['uv','venv','--python',str(Path(sys.executable).resolve()),str(out/'.venv')],check=True,env=env)
    python=out/'.venv/bin/python'
    subprocess.run(['uv','pip','install','--python',str(python),'--no-cache','torch==2.14.1','--index-url','https://download.pytorch.org/whl/cpu'],check=True,env=env)
    subprocess.run(['uv','pip','install','--python',str(python),'--no-cache','numpy==2.5.4'],check=True,env=env)
    target=out/'.venv/lib'/('python'+str(sys.version_info.major)+'.'+str(sys.version_info.minor))/'site-packages'
    (target/'trace_transport_frozen.pth').write_text(str(out/'src')+'\n')
    print(out)


if __name__=='__main__':main()
