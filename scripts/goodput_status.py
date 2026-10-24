"""Small read-only status view; training values are not held-out results."""
import argparse
import json
from pathlib import Path
import time


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('runs/goodput-v21'))
    args=p.parse_args();root=args.root
    training_active=False
    for path in sorted(root.glob('*/progress.json')):
        progress=read(path);config=read(path.parent/'config.json').get('training',{})
        if not config:continue
        age=time.time()-path.stat().st_mtime
        phase=progress.get('phase','starting')
        if phase=='training' and age>300:phase='stale heartbeat'
        training_active=training_active or phase=='training'
        print(f"{path.parent.name}: {phase}; {progress.get('decisions',0):,}/{config.get('decisions',0):,} endpoint opportunities; {progress.get('decisions_per_second',0):.0f}/s")
        print('  Recent TRAINING misses:',progress.get('recent_miss_rate'),'(not validation/test)')
    print('Study phase:',read(root/'progress.json').get('phase','training; evaluation pending' if training_active else 'not started'))
    for path in sorted(root.glob('*-selection.json')):
        result=read(path);print('Validation selection:',result.get('checkpoint'),'win:',result.get('comparison',{}).get('win'),'[selection data only]')
    if not (root/'results.json').exists():print('Reserved test results: not available; no test win claimed.')


if __name__=='__main__':main()
