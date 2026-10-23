import json
import pytest
from trace_transport.joint_traces import load_trace


def test_whole_trace_validation_and_fingerprint(tmp_path):
    path=tmp_path/'wifi.json'
    trace=dict(format_version='service-trace-v1',trace_id='heldout-wifi-run',duration_seconds=60,
               network={'bandwidth_bps':1000000,'capacity_changes':[[1,500000],[3,2000000]]})
    path.write_text(json.dumps(trace))
    cfg,meta=load_trace(path,required_seconds=40)
    assert cfg['feedback_delay_budget']==.25 and len(meta['sha256'])==64
    with pytest.raises(ValueError,match='entire session'):load_trace(path,required_seconds=80)
    trace['network']['initial_rate']=9999;path.write_text(json.dumps(trace))
    with pytest.raises(ValueError,match='unsupported'):load_trace(path,required_seconds=40)
