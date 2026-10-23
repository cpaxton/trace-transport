from trace_transport.joint_curriculum import sample_case


def test_wire_load_strata_are_correct_and_cases_reproducible():
    seen=set()
    for seed in range(40):
        a=sample_case(seed,production=2,drain=2);b=sample_case(seed,production=2,drain=2)
        assert a==b
        cfg,w,label=a;seen.add(label['target_load_bucket'])
        low,high={'feasible':(.2,.7),'near':(.7,1.1),'overloaded':(1.1,3)}[label['target_load_bucket']]
        assert low-1e-8<=label['actual_offered_ratio']<=high+1e-8
        assert 100000<=cfg.bandwidth_bps<=50000000
        assert 16000<=cfg.feedback_bandwidth_bps<=10000000
        assert w.producer_phase_jitter
    assert seen=={'feasible','near','overloaded'}


def test_easy_domain_has_wire_headroom_in_both_directions():
    for seed in range(20):
        cfg,w,label=sample_case(seed,stage='easy')
        assert max(label['directional_wire_load'])<=.4+1e-9
        assert cfg.loss==0 and cfg.ack_loss==0 and cfg.duplicate==0
        assert not w.shared_airtime
