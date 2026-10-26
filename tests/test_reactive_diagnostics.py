import pytest
pytest.importorskip('torch')
from dataclasses import replace
import numpy as np
from scripts import reactive_diagnostics as diagnostic
from scripts import reactive_pilot as pilot
from trace_transport.goodput_reactive import GoodputReactiveEnv,sample_case
from trace_transport.joint_session import ACK_ACTIONS


def test_banks_fail_closed_and_development_is_disjoint(monkeypatch):
    assert not set(diagnostic.bank(150000)) & set(pilot.validation_bank())
    monkeypatch.setattr(pilot,'DOMAIN',{'families':['present','missing']})
    monkeypatch.setattr(pilot,'sample_case',lambda seed:(None,None,{'family':'present'}))
    with pytest.raises(ValueError,match='missing'):pilot.validation_bank()
    monkeypatch.setattr(diagnostic,'DOMAIN',{'families':['present','missing']})
    monkeypatch.setattr(diagnostic,'sample_case',lambda seed:(None,None,{'family':'present'}))
    with pytest.raises(ValueError,match='incomplete'):diagnostic.bank(0)


def test_instrumentation_counts_effects_and_preserves_transport():
    cfg,w,_=sample_case(42,production=.5,drain=.2,stage='easy')
    cfg=replace(cfg,initial_rate=2000.,control_interval=.02,bandwidth_bps=1000000.,feedback_bandwidth_bps=1000000.,base_rtt=.01,jitter=0.,loss=0.,ack_loss=0.)
    plain=GoodputReactiveEnv(cfg,w);audit=diagnostic.InstrumentedEnv(cfg,w)
    plain.reset(seed=42);audit.reset(seed=42)
    tx=np.zeros((plain.count,3));slow=np.full(plain.count,ACK_ACTIONS.index((32,.25))+1)
    while plain.now<.2:
        plain.step(tx,slow);audit.step(tx,slow)
        if any(s.pending for s in plain.sessions):break
    assert any(s.pending for s in plain.sessions)
    plain.step(tx,np.ones(plain.count,int));audit.step(tx,np.ones(plain.count,int))
    assert audit.control_stats['changes_with_pending']>0
    assert audit.control_stats['control_flushes']>0
    assert audit.control_stats['control_flush_wire_bytes']>0
    while not plain.done():
        plain.step(tx,slow);audit.step(tx,slow)
    assert plain.metrics()==audit.metrics()


def test_pilot_records_the_exact_shaped_training_configuration(tmp_path,monkeypatch):
    import json,sys
    from dataclasses import asdict
    monkeypatch.chdir(tmp_path)
    (tmp_path/'source-manifest.json').write_text('{}')
    monkeypatch.setattr(sys,'argv',['reactive-pilot','--output',str(tmp_path/'run'),'--decisions','2'])
    class StopBeforeTraining(Exception):pass
    def inspect(output,cfg,model):
        plan=json.loads((tmp_path/'run/plan.json').read_text())
        assert cfg.progress_shaping is True
        assert plan['base_training']==asdict(cfg)
        raise StopBeforeTraining
    monkeypatch.setattr(pilot,'train',inspect)
    with pytest.raises(StopBeforeTraining):pilot.main()


def test_seeded_sampling_audit_is_repeatable_and_applies_non_teacher_actions(tmp_path):
    import torch
    from trace_transport.joint_models import JointModelConfig,SenderActor,ReceiverActor,CentralCritic,save_checkpoint
    torch.set_num_threads(1);torch.manual_seed(9)
    cfg=JointModelConfig(history=2,width=8,layers=1,heads=2)
    baseline=dict(rate='delay-fast',scheduler='edf',feedback='budgeted',initial_rate=512000.,batch=1,delay=0.)
    checkpoint=tmp_path/'actor.pt'
    save_checkpoint(checkpoint,SenderActor(cfg),ReceiverActor(cfg),CentralCritic(),cfg,
        dict(profile='goodput-v6',baseline=baseline,training={}))
    a=diagnostic.action_case(150000,checkpoint,baseline,production=.1,drain=.1,receiver_mode='sampled',sampling_seed=12)
    b=diagnostic.action_case(150000,checkpoint,baseline,production=.1,drain=.1,receiver_mode='sampled',sampling_seed=12)
    assert a==b
    assert sum(a['applied_receiver_choices'].values())==a['active_receiver_decisions']
    assert any(int(k)>0 and v>0 for k,v in a['applied_receiver_choices'].items())


def test_hindsight_report_rejects_throughput_with_deadline_regression():
    import json,copy
    from pathlib import Path
    from scripts.reactive_diagnostics_report import report
    from trace_transport.goodput_evaluation import summarize
    data=json.loads((Path(__file__).resolve().parents[1]/'artifacts/reactive-v6-pilot-results.json').read_text())
    ref=next(r for r in data['modes']['adaptive']['records'] if r['interactive_on_time']>0 and summarize([r])['zero_accepted_bulk_hosts']==0)
    bad=copy.deepcopy(ref);bad['verified_bulk_bytes']*=2;bad['interactive_on_time']=0
    payload=dict(plan={'controls':[{'name':'reference'}]},development={
        n:dict(records=[r],summary=summarize([r]),comparison=None) for n,r in [('reference',ref),('bad',bad)]},roles={},action_audit=[])
    result=report(payload)
    assert result['hindsight_selection'][0]['control']=='reference'
    assert result['hindsight_relative_gain']==0.


def test_selector_probes_are_causal_repeatable_and_have_no_family_label():
    from scripts.selector_coverage import probe_history
    from trace_transport.joint_session import SENDER_FEATURES
    baseline=dict(rate='delay-fast',scheduler='edf',feedback='budgeted',initial_rate=512000.,batch=1,delay=0.)
    a=probe_history(170000,baseline,seconds=.04,history=4)
    assert a==probe_history(170000,baseline,seconds=.04,history=4)
    assert np.asarray(a['history']).shape==(4,len(SENDER_FEATURES))
    assert a['schema']==list(SENDER_FEATURES)
    assert 'family' not in a and a['endpoint']==0
