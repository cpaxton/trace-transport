"""Read-only inference proposal audit on four fixed stress cases.

Counts valid history rows, including rows for currently inactive endpoints;
these are model proposals, not necessarily actions applied by the environment.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from scripts.goodput_study import write
from trace_transport.joint_models import ReceiverActor, SenderActor, load_checkpoint
from trace_transport.joint_evaluation import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('runs/goodput-v21'))
    p.add_argument('--seed',type=int,default=82000)
    p.add_argument('--output',type=Path,default=Path('runs/changing-comparison-v1/action-audit.json'))
    args=p.parse_args();root=args.root
    reference=root/'reference.json' if (root/'reference.json').exists() else root/'plan.json'
    baseline=json.loads(reference.read_text())['baseline']
    checkpoint=root/('seed42' if (root/'seed42').exists() else 'learn42')/'final.pt'
    _,_,_,metadata=load_checkpoint(checkpoint)
    profile=metadata['profile']
    counts=Counter();stats=dict(sender_rows=0,absolute_rate_residual_sum=0.,absolute_window_residual_sum=0.)
    original_rx=ReceiverActor.act;original_tx=SenderActor.act
    def receiver(self,h,valid):
        action=original_rx(self,h,valid)
        counts.update(action[valid.any(-1)].tolist())
        return action
    def sender(self,h,valid,q,mask):
        action=original_tx(self,h,valid,q,mask);a=action[valid.any(-1)]
        stats['sender_rows']+=len(a)
        stats['absolute_rate_residual_sum']+=float(a[:,0].abs().sum())
        stats['absolute_window_residual_sum']+=float(a[:,1].abs().sum())
        return action
    ReceiverActor.act=receiver;SenderActor.act=sender
    try:
        rows=evaluate(seeds=list(range(args.seed,args.seed+4)),checkpoint=checkpoint,baseline=baseline,
                      environments=2,production=metadata['training']['production_seconds'],drain=metadata['training']['drain_seconds'],ood=profile=='goodput-v2',profile=profile)
    finally:
        ReceiverActor.act=original_rx;SenderActor.act=original_tx
    write(args.output,dict(seeds=[r['seed'] for r in rows],
          scope='valid-history model proposals, including inactive endpoint rows; not applied-action counts',
          receiver_choices=dict(counts),zero_choice='follow adaptive teacher ACK settings',sender=stats))


if __name__=='__main__':main()
