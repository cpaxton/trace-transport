"""Training-weight-only sufficient bound for an always-teacher greedy policy."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import torch
from torch import nn
from trace_transport.experiment_io import write
from trace_transport.reactive_sender import BIASES
from trace_transport.reactive_sender_training import load


def bounds(actor):
    if not isinstance(actor.encoder[-1],nn.Tanh):raise ValueError('bounded Tanh encoder required')
    teacher=BIASES.index(1.);w=actor.head.weight.detach();b=actor.head.bias.detach()
    margins=[float(b[teacher]-b[i]-(w[teacher]-w[i]).abs().sum()) for i in range(len(BIASES)) if i!=teacher]
    return dict(teacher_logit_lower_margins=margins,guaranteed_greedy_teacher=all(m>1e-5 for m in margins),
        caveat='sufficient bound over bounded hidden activations; negative/zero does not prove adaptive behavior; margin tolerance protects floating-point comparisons')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    rows={}
    for path in sorted(args.root.glob('seed*/final.pt')):
        actor,meta=load(path);rows[path.parent.name]=dict(macro_decisions=meta['macro_decisions'],**bounds(actor))
    write(args.output,dict(version='reactive-sender-head-bound-v1',observed_utc=datetime.now(timezone.utc).isoformat(),
        scope='training weights only, no validation examples or outcomes; sufficient all-input teacher bound for Tanh encoders',runs=rows))


if __name__=='__main__':main()
