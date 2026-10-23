import numpy as np
from trace_transport.goodput import GoodputEnv,sample_case


def test_backlog_refills_only_with_credit_and_counts_verified_fixed_window_bytes():
    cfg,w,label=sample_case(42,production=.4,drain=.2,stage='easy')
    env=GoodputEnv(cfg,w);env.reset(seed=42)
    while not env.done():env.step(np.zeros((env.count,3)),np.zeros(env.count,int))
    m=env.metrics()
    assert m['elapsed']>=.6-1e-9
    assert m['verified_bulk_bytes']>0 and m['wrong']==0
    assert m['confirmed_bulk_bytes']<=m['verified_bulk_bytes']
    assert m['measurement_seconds']==.4
    for s in env.sessions:
        assert sum(len(s.messages[mid].message.payload) for mid in s.live)<=w.byte_limit
    # Drained delivery is reported, but never counted retrospectively as speed.
    assert m['verified_bulk_bytes']<=m['image_bytes']+m['map_bytes']


def test_churn_stops_new_offers_without_evicting_reliable_messages():
    cfg,w,label=sample_case(42,production=.5,drain=.1,stage='easy')
    env=GoodputEnv(cfg,w,churn=True);env.reset(seed=42)
    while env.now<.22:env.step(np.zeros((2,3)),np.zeros(2,int))
    assert not env.active
    admitted=env.metrics()['messages']
    while env.now<.25:env.step(np.zeros((2,3)),np.zeros(2,int))
    assert env.metrics()['messages']==admitted


def test_goodput_reward_gives_no_credit_for_partial_or_post_window_bytes():
    cfg,w,_=sample_case(4,production=.2,drain=.2,stage='easy')
    env=GoodputEnv(cfg,w);env.reset(seed=4)
    env.now=.21
    m=next(iter(env.sessions[0].messages.values()))
    env.on_verified_delivery(0,1,m)
    assert not env.verified


def test_critical_cases_exist_without_passing_family_or_link_state_to_actor():
    families=set()
    for seed in range(100):
        cfg,w,label=sample_case(seed)
        families.add(label['family'])
        assert label['family'] in ('steady','wireless-loss','contention','ack-bottleneck','handoff','churn','loss-cause-switch')
        assert cfg.bandwidth_bps<=8000000
    assert len(families)==7
