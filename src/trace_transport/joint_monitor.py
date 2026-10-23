"""Read-only, dependency-light monitoring of local joint experiments."""
import json
import time
from pathlib import Path


def _read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def status(root):
    base=Path(root)/'runs/joint-v1';runs=[]
    for name in ['pilot','seed42','seed43','seed44','history1','sender-only','receiver-only']:
        path=base/name;config=_read(path/'config.json',{})
        progress=_read(path/'progress.json',{})
        if not config and not progress:continue
        phase=progress.get('phase','starting')
        try:age=max(0.,time.time()-(path/'progress.json').stat().st_mtime)
        except OSError:age=None
        # PIDs can be outside the web server's sandbox namespace. Use the
        # recorded phase and expose heartbeat age rather than falsely claiming exit.
        if phase=='training' and age is not None and age>300:phase='awaiting update'
        history=[]
        try:
            with (path/'training.jsonl').open('rb') as f:
                f.seek(0,2);end=f.tell();f.seek(max(0,end-500000));lines=f.read().splitlines()
            for line in lines[-100:]:
                try:
                    row=json.loads(line);history.append(dict(decisions=row['decisions'],miss=row.get('recent_miss_rate')))
                except (ValueError,KeyError):pass
        except OSError:pass
        runs.append(dict(name=name,phase=phase,last_update_age_seconds=age,config=config,progress=progress,history=history,
                         has_checkpoint=(path/'latest.pt').exists()))
    study=base/'study';results=_read(study/'results.json',{})
    tables={}
    for suite in ['final_test','ood']:
        rows=[]
        for name,value in results.get(suite,{}).items():
            if not isinstance(value,dict) or 'summary' not in value:continue
            records=value.get('records',[]);p95=[r['interactive_p95'] for r in records if r.get('interactive_p95') is not None]
            rows.append(dict(name=name,**value['summary'],mean_session_p95=sum(p95)/len(p95) if p95 else None))
        tables[suite]=rows
    return dict(runs=runs,study=_read(study/'progress.json',{}),selection=_read(study/'selection.json',{}),
                baseline=_read(study/'baseline.json',{}).get('selected'),tables=tables,
                comparisons={suite:results.get(suite,{}).get('comparison') for suite in ['final_test','ood']})


def report(root):
    result=_read(Path(root)/'runs/joint-v1/study/results.json')
    if result is None:raise KeyError('joint results are not available yet')
    return result
