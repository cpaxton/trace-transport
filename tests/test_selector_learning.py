import pytest
pytest.importorskip('torch')
import copy,json
from pathlib import Path
import numpy as np
import torch
from scripts.selector_learning import dataset,Selector,KINDS,normalizer,normalized
from trace_transport.joint_session import SENDER_FEATURES


def sample_data():
    source=json.loads((Path(__file__).resolve().parents[1]/'artifacts/reactive-v6-pilot-results.json').read_text())
    row=copy.deepcopy(source['modes']['adaptive']['records'][0]);seed=row['seed']
    return dict(plan=dict(controls=[{'name':'reference'}],splits={'development':[seed]},history=4),
        splits={'development':dict(controls={'reference':dict(records=[row])},probes=[dict(seed=seed,
            history=np.zeros((4,len(SENDER_FEATURES))).tolist(),valid=[False,True,True,True],schema=list(SENDER_FEATURES))])})


def test_dataset_rejects_missing_candidates_and_seed_membership():
    data=sample_data();part=dataset(data,'development');assert part['x'].shape==(1,4,len(SENDER_FEATURES))
    data['plan']['controls'].append({'name':'absent'})
    with pytest.raises(ValueError,match='coverage'):dataset(data,'development')
    data=sample_data();data['splits']['development']['controls']['reference']['records']=[]
    with pytest.raises(ValueError,match='membership'):dataset(data,'development')


def test_family_and_future_labels_cannot_enter_input_tensors():
    data=sample_data();a=dataset(data,'development')
    data['splits']['development']['probes'][0]['family']='do not use'
    data['splits']['development']['controls']['reference']['records'][0]['domain']['family']='changed'
    b=dataset(data,'development')
    assert torch.equal(a['x'],b['x']) and torch.equal(a['valid'],b['valid'])
    mean,scale=normalizer(a);assert torch.isfinite(normalized(a,mean,scale)).all()


@pytest.mark.parametrize('kind',KINDS)
def test_selectors_ignore_invalid_prefix_values(kind):
    torch.set_num_threads(1);torch.manual_seed(1)
    model=Selector(kind,history=4,choices=3).eval()
    x=torch.randn(2,4,len(SENDER_FEATURES));valid=torch.tensor([[False,False,True,True],[False,True,True,True]])
    corrupted=x.clone();corrupted[~valid]=100000.
    with torch.inference_mode():a=model(x,valid);b=model(corrupted,valid)
    assert a.shape==(2,3) and torch.equal(a,b)


def test_executed_baseline_matches_standard_evaluation_and_zero_switch(tmp_path):
    from scripts.selector_execute import execute_case,load_selector,BASELINE
    from trace_transport.joint_evaluation import evaluate
    torch.set_num_threads(1)
    actual=execute_case(170000,production=.1,drain=.1)
    expected=evaluate(seeds=[170000],baseline=BASELINE,environments=1,production=.1,drain=.1,profile='goodput-v6')
    assert actual['record']==expected[0]
    model=Selector('current-mlp',history=4,choices=1)
    for p in model.parameters():p.data.zero_()
    path=tmp_path/'model.pt'
    torch.save(dict(kind='current-mlp',history=4,choices=1,controls=[dict(name='baseline',baseline=BASELINE)],
        sender_features=list(SENDER_FEATURES),mean=torch.zeros(len(SENDER_FEATURES)),scale=torch.ones(len(SENDER_FEATURES)),state=model.state_dict()),path)
    switched=execute_case(170000,selector=load_selector(path),production=.1,drain=.1)
    assert switched['record']==actual['record']
    assert all(i==0 for i in switched['endpoint_choices'])


def test_saved_metadata_roundtrips_with_weights_only_loader(tmp_path):
    from scripts.selector_learning import save_model
    from scripts.selector_execute import load_selector,BASELINE,worker_trial
    import pickle
    model=Selector('current-mlp',history=4,choices=1)
    path=tmp_path/'portable.pt'
    save_model(path,model,4,[dict(name='baseline',baseline=BASELINE)],torch.zeros(len(SENDER_FEATURES)),
        torch.ones(len(SENDER_FEATURES)),{'runtime':{'torch':torch.__version__}})
    loaded,metadata=load_selector(path)
    assert type(metadata['plan']['runtime']['torch']) is str
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
    assert pickle.loads(pickle.dumps(worker_trial)) is worker_trial


def test_reused_baselines_require_matching_cases_and_exact_policy(tmp_path):
    import ast,hashlib,zipfile
    import scripts.selector_execute as module
    text=Path(module.__file__).read_text();archive=tmp_path/'source.zip';prior=tmp_path/'results.json'
    source={'src/physics.py':'unchanged','scripts/selector_execute.py':hashlib.sha256(text.encode()).hexdigest()}
    with zipfile.ZipFile(archive,'w') as z:z.writestr('scripts/selector_execute.py',text)
    data=dict(plan=dict(seeds=[1],source=source),modes={'adaptive':{'audits':[{'record':{'seed':1}}]}})
    prior.write_text(json.dumps(data))
    assert module.verified_baselines(prior,archive,[1],source)==data['modes']
    data['modes']['adaptive']['audits']=[];prior.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='incomplete'):module.verified_baselines(prior,archive,[1],source)
    data['modes']['adaptive']['audits']=[{'record':{'seed':1}}]
    changed=text.replace('return dict(record=row,','return dict(record=None,')
    source['scripts/selector_execute.py']=hashlib.sha256(changed.encode()).hexdigest();data['plan']['source']=source
    prior.write_text(json.dumps(data))
    with zipfile.ZipFile(archive,'w') as z:z.writestr('scripts/selector_execute.py',changed)
    with pytest.raises(ValueError,match='policy changed'):module.verified_baselines(prior,archive,[1],source)
