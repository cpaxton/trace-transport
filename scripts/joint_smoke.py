"""Small CPU end-to-end acceptance run; not a performance experiment."""
import argparse
from pathlib import Path
from trace_transport.joint_training import train,JointTrainConfig
from trace_transport.joint_models import JointModelConfig
from trace_transport.joint_study import run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('runs/joint-acceptance'))
    args=p.parse_args()
    for name,seed,history,send,receive in [('seed42',42,4,True,True),('seed43',43,4,True,True),
        ('seed44',44,4,True,True),('history1',42,1,True,True),('sender-only',42,4,True,False),('receiver-only',42,4,False,True)]:
        path=args.output/name
        if not (path/'final.pt').exists():
            train(path,JointTrainConfig(decisions=200,environments=1,rollout=16,epochs=1,production_seconds=.2,
                drain_seconds=.5,seed=seed,learn_sender=send,learn_receiver=receive),JointModelConfig(history=history,width=32,layers=1))
    run(args.output,environments=2,quick=True)


if __name__=='__main__':main()
