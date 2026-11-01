import copy
import json
from pathlib import Path
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from scripts.v7_precision_replication import (validate,identity,fingerprint,collect_case,fixed_choices,report)
from scripts.v7_local_predictability import collect_case as original,fit_choices
from trace_transport.reactive_sender import FEATURES


def test_registered_replicas_are_fresh_and_bounded():
    c=json.loads(Path('experiments/v7-precision-replication-v1.json').read_text())
    s=validate(c,'pilot');assert len(s['cases'])==12 and s['replicas']==64 and s['replica_offset']==16
    c['profiles']['pilot']['replica_offset']=9
    with pytest.raises(ValueError,match='overlap'):validate(c,'pilot')
    c['profiles']['pilot']['replica_offset']=0
    with pytest.raises(ValueError,match='fresh'):validate(c,'pilot')


def test_same_snapshot_new_futures_and_changed_state_fails(tmp_path):
    task=dict(seed=700006,times=[.1],replicas=2,production=.3,drain=.1)
    old=original(task)
    task.update(states=old['states'],replica_offset=16,output=str(tmp_path))
    new=collect_case(task)
    assert new['complete'] and fingerprint(new['states'][0])==fingerprint(old['states'][0])
    old_ids={r['replica_seed'] for b in old['states'][0]['banks'].values() for arm in b for r in arm}
    new_ids={r['replica_seed'] for b in new['states'][0]['banks'].values() for arm in b for r in arm}
    assert not old_ids&new_ids and len(new_ids)==12
    assert json.loads((tmp_path/'case-700006-progress.json').read_text())['completed']==12
    bad=copy.deepcopy(task);bad['states'][0]['history'][-1][0]+=1
    with pytest.raises(ValueError,match='identity'):collect_case(bad)


def test_frozen_model_replay_and_all_controls(tmp_path):
    h=np.zeros((64,len(FEATURES))).tolist()
    states=[dict(seed=1,snapshot=0,status='complete',producer=0,seconds=.1,history=h,valid=[True]*64,
        reference_snapshot_sha256='frozen',banks={k:[[dict(return_value=float(a),verified=a,misses=0,wire=a) for _ in range(2)] for a in range(3)] for k in ['fit','score']})]
    for kind in ['current','history','transformer']:
        for seed in [1,2,3]:
            fit=fit_choices(states,states,kind,seed,1)
            torch.save(dict(kind=kind,features=list(FEATURES),weights=fit['weights'],normalization=fit['normalization']),tmp_path/f'{kind}-{seed}.pt')
    expected=fixed_choices(tmp_path,states,[1,2,3]);assert len(expected)==9
    result=report([dict(seed=1,states=states,complete=True)],tmp_path,dict(constant_action=0,model_seeds=[1,2,3]))
    assert len(result['modes'])==12
    assert result['modes']['teacher']['return_difference_ci95']==[0.,0.]
    assert result['modes']['snapshot-fit-table']['return_value_difference_from_teacher']==1.
    assert result['fit_score_action_agreement']==1.


def test_reference_hash_and_frozen_source_fail_closed():
    import hashlib
    from scripts.v7_precision_replication import check_reference
    data=dict(test=[dict(seed=i,states=[dict(status='complete')]*2) for i in range(12)])
    config=dict(reference_data_sha256=hashlib.sha256(json.dumps(data,indent=2).encode()+b'\n').hexdigest(),constant_action=0,
        profiles=dict(pilot=dict(cases=list(range(12))),engineering=dict(cases=[0,1])))
    result=dict(constant_action=0,plan=dict(source={'src/frozen.py':'abc'}))
    check_reference(config,data,result,{'src/frozen.py':'abc'})
    with pytest.raises(ValueError,match='source mismatch'):check_reference(config,data,result,{'src/frozen.py':'changed'})
    data['test'][0]['seed']=99
    with pytest.raises(ValueError,match='dataset mismatch'):check_reference(config,data,result,{'src/frozen.py':'abc'})
