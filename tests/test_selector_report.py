import copy
import pytest
from scripts.selector_report import report,KINDS


def complete_data():
    names=['adaptive','low-rtt-bdp-rule']+[f'{kind}-{seed}' for kind in KINDS for seed in [1,2,3]]
    return dict(plan={'seeds':[1],'started_utc':'2026-10-26'},completed_utc='2026-10-26',modes={
        n:dict(summary={'verified_goodput':100.,'interactive_miss_rate':.1},
        comparison=None if n=='adaptive' else {'guards':{'deadline':True},'win':True},
        audits=[{'record':{'seed':1},'inference_ms':None}]) for n in names})


def test_complete_report_never_promotes_validation_to_accepted_win():
    out=report(complete_data())
    assert out['accepted_win'] is False and len(out['modes'])==14
    assert out['groups']['transformer']['all_quality_guards_pass_count']==3


def test_report_refuses_partial_or_missing_cases():
    data=complete_data();del data['completed_utc']
    with pytest.raises(ValueError,match='finish'):report(data)
    data=complete_data();del data['modes']['gru-3']
    with pytest.raises(ValueError,match='finish'):report(data)
    data=complete_data();data['modes']['gru-3']['audits']=[]
    with pytest.raises(ValueError,match='coverage'):report(data)
