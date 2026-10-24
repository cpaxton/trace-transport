"""Read-only inference proposal audit on four fixed stress cases.

Counts valid history rows, including rows for currently inactive endpoints;
these are model proposals, not necessarily actions applied by the environment.
"""
from collections import Counter
import json
from pathlib import Path
import torch
from scripts.goodput_study import write
from trace_transport.joint_models import ReceiverActor, SenderActor
from trace_transport.joint_evaluation import evaluate


def main():
    root=Path('runs/goodput-v21')
    baseline=json.loads((root/'reference.json').read_text())['baseline']
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
        rows=evaluate(seeds=list(range(82000,82004)),checkpoint=root/'learn42/final.pt',baseline=baseline,
                      environments=2,production=6.,drain=2.,ood=True,profile='goodput-v2')
    finally:
        ReceiverActor.act=original_rx;SenderActor.act=original_tx
    write('runs/changing-comparison-v1/action-audit.json',dict(seeds=[r['seed'] for r in rows],
          scope='valid-history model proposals, including inactive endpoint rows; not applied-action counts',
          receiver_choices=dict(counts),zero_choice='follow adaptive teacher ACK settings',sender=stats))


if __name__=='__main__':main()
