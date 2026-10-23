from types import SimpleNamespace
import pytest
from trace_transport.joint_workers import EnvironmentPool


def test_worker_disconnect_identifies_exitcode_and_reproducible_seed():
    def disconnected():raise EOFError
    pool=EnvironmentPool.__new__(EnvironmentPool)
    pool.pipes=[SimpleNamespace(poll=lambda timeout:True,recv=disconnected)]
    pool.processes=[SimpleNamespace(join=lambda timeout:None,exitcode=-11)]
    pool.context=[{'seed':50003,'steps':124}]
    with pytest.raises(RuntimeError,match=r'exitcode=-11.*50003.*124'):
        pool.receive(0)
