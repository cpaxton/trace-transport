"""V5 metric audit on existing v4 validation worlds; no new learning claim."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
from trace_transport.experiment_io import write
from trace_transport.joint_evaluation import evaluate
from trace_transport.goodput_evaluation import summarize
from trace_transport.learning_diagnostics import stratify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/progress-baseline-v5/results.json'));args=p.parse_args()
    # Same selected Caliban reference. Explicit diagnostic reuse, not held-out data.
    seeds=[121007,121017]
    rows=evaluate(seeds=seeds,baseline=dict(rate='delay-fast',scheduler='edf',feedback='budgeted',initial_rate=512000.,batch=1,delay=0.),
                  environments=2,production=12.,drain=3.,profile='goodput-v5')
    write(args.output,dict(version='progress-baseline-audit-v1',completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='reused v4 validation cases for metric diagnosis only; no selection or performance win',
        summary=summarize(rows),strata=stratify(rows),records=rows))


if __name__=='__main__':main()
