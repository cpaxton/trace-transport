from pathlib import Path
from types import SimpleNamespace
import pytest
pytest.importorskip('torch')
from trace_transport import joint_study


def test_completed_joint_models_do_not_wait_for_unrelated_ablation(tmp_path,monkeypatch):
    names=['seed42','seed43','seed44']
    for name in names:
        path=tmp_path/name;path.mkdir();(path/'final.pt').touch()
    monkeypatch.setattr(joint_study,'load_checkpoint',lambda path:(None,None,SimpleNamespace(history=32),
        {'decisions':5000000,'training':{'decisions':5000000,'seed':int(Path(path).parent.name[4:])}}))
    joint_study._ready(tmp_path,names,{},tmp_path,False)
    with pytest.raises(RuntimeError,match='history1'):
        joint_study._ready(tmp_path,['history1'],{},tmp_path,False)
