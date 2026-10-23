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
