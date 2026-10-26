"""Summarize finite-control development headroom without a deployment claim."""
import argparse
import json
from pathlib import Path
from trace_transport.experiment_io import write
from trace_transport.goodput_evaluation import summarize


def admissible(candidate,reference):
    a,b=summarize([candidate]),summarize([reference])
    return (a['interactive_miss_rate']<=b['interactive_miss_rate']+.005 and a['wrong']==0
        and a['wire_per_useful_byte']<=1.05*b['wire_per_useful_byte']
        and a['completion_fraction']>=b['completion_fraction']-.005
        and a['mean_bulk_progress_jain']>=b['mean_bulk_progress_jain']-.05
        and a['zero_accepted_bulk_hosts']==0
        and all(a[k]>=.95*b[k] for k in ['image_goodput','map_goodput','confirmed_goodput','accepted_bulk_goodput']))


def report(data):
    modes=data['development'];name=data['plan']['controls'][0]['name'];reference=modes[name]['records']
    indexed={n:{r['seed']:r for r in m['records']} for n,m in modes.items()}
    chosen=[];selection=[]
    for ref in reference:
        feasible=[(n,rows[ref['seed']]) for n,rows in indexed.items() if admissible(rows[ref['seed']],ref)]
        # Keep the reference when no candidate can satisfy the strict host guard.
        winner,row=max(feasible,key=lambda pair:pair[1]['verified_bulk_bytes']) if feasible else (name,ref)
        chosen.append(row);selection.append(dict(seed=ref['seed'],family=ref['domain']['family'],control=winner,
            admissible_controls=[n for n,_ in feasible]))
    base=summarize(reference);best=summarize(chosen)
    return dict(version='reactive-development-headroom-v1',reference=name,
        scope='optimistic same-development-case hindsight selection among finite controls; not an optimal upper bound, online policy, unbiased effect estimate or held-out result; per-case guard checks are descriptive without inference',
        baseline_summary=base,hindsight_summary=best,
        hindsight_relative_gain=best['verified_goodput']/base['verified_goodput']-1 if base['verified_goodput'] else None,
        hindsight_selection=selection,controls={n:dict(summary=m['summary'],comparison=m['comparison']) for n,m in modes.items()},
        role_summary={n:{k:v for k,v in m.items() if k!='records'} for n,m in data['roles'].items()},
        action_summary=[{k:v for k,v in a.items() if k!='record'} for a in data['action_audit']])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();data=json.loads(args.results.read_text())
    if 'completed_utc' not in data:raise ValueError('diagnostics incomplete')
    write(args.output,report(data))


if __name__=='__main__':main()
