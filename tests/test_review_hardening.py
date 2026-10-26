import pytest
pytest.importorskip('torch')
import torch
from trace_transport import joint_training as training
from scripts import goodput_v3_pilot as pilot
from scripts import caliban_study as study


def test_receiver_diagnostics_and_aggregation_follow_extended_action_table(monkeypatch):
    actions=training.ACK_ACTIONS+((64,.25),)
    monkeypatch.setattr(training,'ACK_ACTIONS',actions)
    monkeypatch.setattr(pilot,'ACK_ACTIONS',actions)
    last=len(actions)
    stats=training.receiver_choice_diagnostics(torch.tensor([0,last,last]),torch.tensor([0.,2.,4.]))
    assert len(stats['choice_counts'])==last+1 and stats['choice_counts'][last]==2
    assert stats['choice_mean_normalized_advantage'][last]==3.
    result=pilot.aggregate_receiver_telemetry([{'receiver':stats}])
    assert result['choices'][last]==2 and result['mean_normalized_advantage'][last]==3.
    with pytest.raises(ValueError):pilot.aggregate_receiver_telemetry([{'receiver':{'choice_counts':[1]}}])
    with pytest.raises(ValueError):training.receiver_choice_diagnostics(torch.tensor([last+1]),torch.tensor([0.]))


def test_calibrated_initial_rate_reaches_training_config():
    baseline=dict(rate='delay-fast',scheduler='edf',batch=4,delay=.005,initial_rate=128000.)
    cfg=study.config(42,100,2,baseline)
    assert cfg.teacher_initial_rate==128000. and cfg.teacher_batch==4 and cfg.teacher_delay==.005
    assert study.config(42,100,2).teacher_initial_rate==512000.


def test_bank_refuses_missing_family_and_invalid_counts(monkeypatch):
    monkeypatch.setattr(study,'FAMILIES',('present','missing'))
    monkeypatch.setattr(study,'sample_case',lambda seed:(None,None,{'family':'present'}))
    with pytest.raises(ValueError,match='missing'):study.bank(0,2)
    with pytest.raises(ValueError):study.bank(0,0)


def test_resume_rejects_changed_validation_membership():
    keys=['domain','training_seeds','decisions_per_seed','environments','model','source_sha256',
          'calibration_seeds','validation_seeds','reserved_test','reserved_ood']
    prior={k:None for k in keys};prior['validation_seeds']=[1,2]
    study.validate_resume_plan(prior,dict(prior))
    with pytest.raises(ValueError,match='validation_seeds'):
        study.validate_resume_plan(prior,{**prior,'validation_seeds':[1,3]})
