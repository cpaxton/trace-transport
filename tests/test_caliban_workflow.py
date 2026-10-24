from pathlib import Path
import json
import zipfile
import pytest
pytest.importorskip('torch')
from scripts.caliban_deploy import package
from scripts.caliban_study import mirror_once,config


def test_persistent_mirror_keeps_selection_snapshots_and_uses_atomic_names(tmp_path):
    root=tmp_path/'runtime';persistent=tmp_path/'persistent';root.mkdir();persistent.mkdir()
    (root/'seed42').mkdir();(root/'plan.json').write_text('{}')
    for name in ['initial.pt','latest.pt','d000500000.pt']:(root/'seed42'/name).write_bytes(name.encode())
    mirror_once(root,persistent)
    assert (persistent/'seed42/d000500000.pt').read_bytes()==b'd000500000.pt'
    assert (persistent/'seed42/initial.pt').exists()
    (root/'seed42/latest.pt').write_bytes(b'new checkpoint');mirror_once(root,persistent)
    assert (persistent/'seed42/latest.pt').read_bytes()==b'new checkpoint'
    assert not list(persistent.rglob('*.copying'))
    assert not config(42,100,2).paired_teacher


def test_deployment_bundle_excludes_runtime_and_secrets(tmp_path):
    root=tmp_path/'root';root.mkdir()
    for name in ['pyproject.toml','uv.lock','README.md']:(root/name).write_text('test')
    (root/'.env.devices').write_text('private')
    (root/'runs').mkdir();(root/'runs/secret.pt').write_text('private')
    archive=tmp_path/'source.zip';manifest=package(root,archive)
    with zipfile.ZipFile(archive) as z:
        assert '.env.devices' not in z.namelist() and 'runs/secret.pt' not in z.namelist()
        assert json.loads(z.read('source-manifest.json'))==manifest


def test_evaluation_cli_uses_checkpoint_profile_not_first_record(monkeypatch,tmp_path):
    import trace_transport.joint_models as models
    import trace_transport.joint_evaluation as evaluation
    import trace_transport.goodput_evaluation as goodput
    from trace_transport.joint_cli import main
    captured={}
    monkeypatch.setattr(models,'load_checkpoint',lambda *a:(None,None,None,{'profile':'goodput-v4'}))
    def evaluate(**kwargs):captured.update(kwargs);return [{}]
    monkeypatch.setattr(evaluation,'evaluate',evaluate)
    monkeypatch.setattr(goodput,'summarize',lambda rows:{'selected':'goodput'})
    report=tmp_path/'report.json'
    assert main(['joint-evaluate','--checkpoint',str(tmp_path/'model.pt'),'--trials','1','--report',str(report)])==0
    assert captured['profile']=='goodput-v4'
    assert json.loads(report.read_text())['summary']=={'selected':'goodput'}
