"""Stratified v4 learning curve and optimistic completion feasibility audit."""
import argparse
import ast
from datetime import datetime,timezone
import json
from pathlib import Path
import zipfile
from trace_transport.experiment_io import write
from trace_transport.goodput_deployment import sample_case
from trace_transport.learning_diagnostics import completion_headroom,stratify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('runs/learning-diagnostics/seed42-validation.json'))
    p.add_argument('--output',type=Path,default=Path('runs/learning-diagnostics/strata.json'))
    args=p.parse_args()
    def sampler_ast(source):
        return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='sample_case'),include_attributes=False)
    with zipfile.ZipFile('artifacts/caliban-v4-source.zip') as z:
        assert sampler_ast(z.read('src/trace_transport/goodput_deployment.py').decode())==sampler_ast(Path('src/trace_transport/goodput_deployment.py').read_text()),'historical sampler changed'
    curves=json.loads(args.input.read_text());initial=curves['initial.pt']['records']
    feasibility=[]
    for row in initial:
        cfg,w,label=sample_case(row['seed'],production=row['measurement_seconds'],drain=row['elapsed']-row['measurement_seconds'])
        assert w.clients==len(row['per_host_verified_bytes']) and label['family']==row['domain']['family']
        feasibility.append(dict(seed=row['seed'],family=label['family'],**completion_headroom(cfg,w),
                                zero_completed_bulk_hosts=sum(x==0 for x in row['per_host_verified_bytes'])))
    result=dict(version='learning-diagnostics-v1',completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='post-hoc validation diagnosis; no changed scores, selection rules or held-out tests',
        checkpoints={name:dict(decisions=data['decisions'],strata=stratify(data['records'])) for name,data in curves.items()},
        feasibility=feasibility)
    write(args.output,result)


if __name__=='__main__':main()
