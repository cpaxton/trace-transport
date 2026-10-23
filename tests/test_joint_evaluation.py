import pytest
pytest.importorskip('torch')
from trace_transport.joint_evaluation import compare,select_baseline


def record(seed=1,ontime=50,**changes):
    row=dict(seed=seed,desired={'interactive':100},interactive_on_time=ontime,
        image_bytes=1000,map_bytes=1000,elapsed=1.,delivered_bytes=3000,
        completed=90,messages=100,total_wire_bytes=4000,wrong=0)
    row.update(changes)
    return row


def test_win_requires_deadline_gain_and_every_resource_guard():
    reference=[record(seed=i) for i in range(8)]
    learned=[record(seed=i,ontime=75) for i in range(8)]
    assert compare(learned,reference)['win']
    for change in [dict(image_bytes=900),dict(map_bytes=900),dict(completed=89),
                   dict(total_wire_bytes=5000),dict(wrong=1)]:
        candidate=[dict(r,**change) for r in learned]
        assert not compare(candidate,reference)['win']
    assert not compare(reference,reference)['win']
    perfect=[record(seed=i,ontime=100) for i in range(8)]
    assert compare(perfect,perfect)['relative_miss_reduction'] is None
    with pytest.raises(ValueError,match='paired seed'):
        compare([record(seed=2)],[record(seed=1)])


def test_baseline_selection_protects_bulk_before_deadline_optimization():
    candidates={'starved':[record(ontime=100,image_bytes=1,map_bytes=1)],
                'eligible':[record(ontime=75)],'slower':[record(ontime=50)]}
    assert select_baseline(candidates)=='eligible'


def test_constant_actor_fast_path_matches_complete_simulated_evaluation(tmp_path,monkeypatch):
    import torch
    from trace_transport.joint_models import JointModelConfig,SenderActor,ReceiverActor,CentralCritic,save_checkpoint
    from trace_transport.joint_evaluation import constant_baseline_actor,evaluate
    cfg=JointModelConfig(history=4,width=32,layers=1);sender=SenderActor(cfg);receiver=ReceiverActor(cfg)
    assert constant_baseline_actor(sender,'sender') and constant_baseline_actor(receiver,'receiver')
    path=tmp_path/'initial.pt';save_checkpoint(path,sender,receiver,CentralCritic(),cfg,{'decisions':0})
    # Initial joint actors follow the fixed/EDF/budgeted teacher for every input.
    kwargs=dict(seeds=[701,702,703],environments=2,production=.1,drain=.5)
    observed=[]
    baseline=evaluate(on_record=observed.append,**kwargs)
    assert sorted(observed,key=lambda r:r["seed"])==baseline
    assert evaluate(checkpoint=path,**kwargs)==baseline
    from trace_transport import joint_evaluation
    monkeypatch.setattr(joint_evaluation,'constant_baseline_actor',lambda actor,role:False)
    assert evaluate(checkpoint=path,**kwargs)==baseline
    # A tiny nonzero learned control must never be classified as constant.
    with torch.no_grad():sender.control.weight[0,0]=1e-10
    assert not constant_baseline_actor(sender,'sender')


def test_asynchronous_worlds_match_synchronous_controls_with_nonzero_actions(tmp_path):
    import torch
    from trace_transport.joint_models import JointModelConfig,SenderActor,ReceiverActor,CentralCritic,save_checkpoint
    from trace_transport.joint_evaluation import evaluate
    cfg=JointModelConfig(history=4,width=32,layers=1);sender=SenderActor(cfg);receiver=ReceiverActor(cfg)
    with torch.no_grad():
        sender.control.bias.copy_(torch.tensor([.05,-.05]));receiver.head.bias[-1]=1.
    path=tmp_path/'controlled.pt';save_checkpoint(path,sender,receiver,CentralCritic(),cfg,{'decisions':1})
    kwargs=dict(seeds=[701,702,703],environments=2,production=.1,drain=.5,checkpoint=path)
    assert evaluate(**kwargs,asynchronous=True)==evaluate(**kwargs,asynchronous=False)
