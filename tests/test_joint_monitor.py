import json
from pathlib import Path
from trace_transport.joint_monitor import status,report


def test_joint_monitor_reads_fixed_paths_and_handles_partial_results(tmp_path):
    run=tmp_path/'runs/joint-v1/seed42';run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps({'training':{'decisions':5000000}}))
    (run/'progress.json').write_text(json.dumps({'phase':'complete','decisions':5000010}))
    (run/'training.jsonl').write_text('{broken\n'+json.dumps({'decisions':100,'recent_miss_rate':.2})+'\n')
    result=status(tmp_path)
    assert result['runs'][0]['phase']=='complete'
    assert result['runs'][0]['history']==[{'decisions':100,'miss':.2}]
    assert result['tables']=={'final_test':[],'ood':[]}
    study=tmp_path/'runs/joint-v1/study';study.mkdir()
    data={'final_test':{'baseline':{'summary':{'interactive_miss_rate':.2},'records':[{'interactive_p95':.1}]}}}
    (study/'results.json').write_text(json.dumps(data))
    assert status(tmp_path)['tables']['final_test'][0]['mean_session_p95']==.1
    assert report(tmp_path)==data
