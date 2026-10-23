import io
import json
from types import SimpleNamespace

from scripts import device_trials


def test_device_harness_verifies_receiver_digest_and_persists_report(tmp_path,monkeypatch):
    processes=[]
    class Process:
        def __init__(self,args,**kwargs):
            self.args=args
            self.stdout=io.StringIO('{"ready":true,"port":9876}\n')
            self.returncode=None
            self.response=None
            processes.append(self)
        def communicate(self,timeout):
            self.returncode=0
            return json.dumps(self.response),''
        def poll(self):
            return self.returncode
    def send(text,address,port,**kwargs):
        import hashlib
        assert address=='192.0.2.1' and port==9876
        processes[-1].response={'completed':True,'sha256':hashlib.sha256(text.encode()).hexdigest()}
        return SimpleNamespace(completed=True,summary=lambda:{'completed':True})
    monkeypatch.setattr(device_trials.subprocess,'Popen',Process)
    monkeypatch.setattr(device_trials,'send_udp',send)
    report=tmp_path/'result.json'
    monkeypatch.setattr('sys.argv',['device_trials','--ssh','robot','--address','192.0.2.1',
        '--condition','wifi-stationary','--trials','1','--bytes','128','--report',str(report)])
    assert device_trials.main()==0
    data=json.loads(report.read_text())
    assert data['completed']==1 and data['trials'][0]['exact_confirmed']
    assert '--receipt-feedback' in processes[0].args[-1]
