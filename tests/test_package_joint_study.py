import json
import pytest
from scripts.package_joint_study import package


def test_packaging_refuses_incomplete_or_quick_studies(tmp_path):
    study=tmp_path/'runs/study';study.mkdir(parents=True)
    (study/'progress.json').write_text(json.dumps({'phase':'waiting-for-training'}))
    (study/'results.json').write_text(json.dumps({'plan':{'quick':False}}))
    with pytest.raises(ValueError,match='completed full study'):
        package(tmp_path,tmp_path/'runs',tmp_path/'artifacts')
    (study/'progress.json').write_text(json.dumps({'phase':'complete'}))
    (study/'results.json').write_text(json.dumps({'plan':{'quick':True}}))
    with pytest.raises(ValueError,match='completed full study'):
        package(tmp_path,tmp_path/'runs',tmp_path/'artifacts')
    assert not (tmp_path/'artifacts').exists()


def test_initial_policy_cannot_be_reported_as_a_learning_win():
    from scripts.package_joint_study import learning_evidence
    comparisons={suite:{'comparison':{'win':True}} for suite in ['final_test','ood']}
    initial=learning_evidence({'decisions':0},comparisons)
    assert not initial['trained_policy_selected']
    assert not any(initial['learning_win'].values())
    assert all(learning_evidence({'decisions':5000000},comparisons)['learning_win'].values())
