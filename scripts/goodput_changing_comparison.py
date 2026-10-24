"""Frozen role comparison on new seeds, including coupled changing conditions.

This is an exploratory stress comparison of a trained checkpoint, not the reserved
acceptance study. No checkpoint or baseline is selected using these results.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.goodput_study import write
from trace_transport.goodput_evaluation import summarize, compare
from trace_transport.joint_evaluation import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('runs/goodput-v21/learn42/final.pt'))
    p.add_argument('--root', type=Path, default=Path('runs/goodput-v21'))
    p.add_argument('--output', type=Path, default=Path('runs/changing-comparison-v1'))
    p.add_argument('--cases', type=int, default=16)
    args = p.parse_args()
    baseline = json.loads((args.root / 'reference.json').read_text())['baseline']
    plan = dict(version='changing-comparison-v1', started_utc=datetime.now(timezone.utc).isoformat(),
                checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                baseline=baseline, production=6., drain=2., cases=args.cases,
                selection='final checkpoint fixed before stress evaluation; exploratory, not acceptance',
                suites={'changing': list(range(81000,81000+args.cases)),
                        'coupled-harsh': list(range(82000,82000+args.cases))})
    write(args.output / 'plan.json', plan)
    results = dict(plan=plan, suites={})
    for suite, seeds in plan['suites'].items():
        modes = {}
        for name, sender, receiver in [('adaptive',False,False),('learned-sender',True,False),
                                       ('learned-receiver',False,True),('joint-learned',True,True)]:
            records = []
            def record(row):
                records.append(row)
                write(args.output / f'{suite}-{name}.json', records)
                write(args.output / 'progress.json', dict(suite=suite,mode=name,completed=len(records),total=len(seeds)))
            rows = evaluate(seeds=seeds, checkpoint=args.checkpoint if sender or receiver else None,
                            baseline=baseline, sender_enabled=sender, receiver_enabled=receiver,
                            environments=2, production=6.,drain=2.,ood=suite=='coupled-harsh',
                            profile='goodput-v2',on_record=record)
            modes[name] = dict(summary=summarize(rows), records=rows)
        reference = modes['adaptive']['records']
        for name,data in modes.items():
            data['comparison'] = compare(data['records'],reference)
        results['suites'][suite] = modes
        write(args.output / 'results.json',results)
    results['completed_utc'] = datetime.now(timezone.utc).isoformat()
    write(args.output / 'results.json',results)
    write(args.output / 'progress.json',dict(phase='complete'))


if __name__ == '__main__':
    main()
