"""Final-checkpoint role ablations for the completed v5 pilot; validation only."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from trace_transport.experiment_io import write
from trace_transport.joint_evaluation import evaluate
from trace_transport.goodput_evaluation import summarize,compare


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    data=json.loads((root/'results.json').read_text());plan=data['plan']
    reference=data['modes']['adaptive']['records'];joint=data['modes']['shaped']['records']
    result=dict(version='progress-v5-role-audit-v1',started_utc=datetime.now(timezone.utc).isoformat(),modes={})
    for name,tx,rx in [('sender-only',True,False),('receiver-only',False,True)]:
        rows=evaluate(seeds=plan['validation_seeds'],checkpoint=root/'shaped/final.pt',baseline=plan['baseline'],
            sender_enabled=tx,receiver_enabled=rx,environments=2,production=12.,drain=3.,profile='goodput-v5')
        result['modes'][name]=dict(summary=summarize(rows),comparison=compare(rows,reference),records=rows,
            exactly_matches_reference_records=rows==reference,exactly_matches_joint_records=rows==joint)
        write(root/'roles.json',result)
    result['completed_utc']=datetime.now(timezone.utc).isoformat();write(root/'roles.json',result)


if __name__=='__main__':main()
