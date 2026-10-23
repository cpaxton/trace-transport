"""Select pilot snapshots using validation only; never opens reserved tests."""
import argparse
import json
from pathlib import Path

from scripts.goodput_study import cases,write
from trace_transport.joint_models import load_checkpoint
from trace_transport.goodput_evaluation import compare,summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('runs/goodput-v2'));p.add_argument('--run',default='pilot42')
    args=p.parse_args();root=args.root;plan=json.loads((root/'plan.json').read_text())
    reference=json.loads((root/'reference.json').read_text());baseline=reference['baseline'];rows={};seen=set()
    for path in [root/args.run/'initial.pt',*sorted((root/args.run).glob('d*.pt')),root/args.run/'final.pt']:
        _,_,_,meta=load_checkpoint(path)
        decisions=meta['decisions']
        if decisions in seen:continue
        seen.add(decisions)
        record=cases(root,f'{args.run}-snapshot-{decisions}',plan['validation'],baseline=baseline,checkpoint=path)
        rows[str(path)]=dict(decisions=decisions,summary=summarize(record),comparison=compare(record,reference['validation']),records=record)
        write(root/(args.run+'-validation.json'),rows)
    selected=max(rows,key=lambda key:(all(rows[key]['comparison']['guards'].values()),rows[key]['summary']['verified_goodput']))
    write(root/(args.run+'-selection.json'),dict(checkpoint=selected,selection='validation guards then verified goodput',
        training_decisions=rows[selected]['decisions'],summary=rows[selected]['summary'],comparison=rows[selected]['comparison']))
    write(root/'progress.json',dict(phase='pilot-validation-complete',selected=selected))


if __name__=='__main__':main()
