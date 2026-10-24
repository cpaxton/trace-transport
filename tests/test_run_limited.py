import pytest
from scripts.run_limited import command


def test_experiment_runs_without_shell_or_terminal_and_has_resource_limits(tmp_path):
    argv=['/usr/bin/python3','-c','print("literal; $(no-shell)")']
    cmd=command('trace-goodput-smoke',argv,tmp_path/'log',directory=tmp_path)
    assert cmd[-3:]==argv
    assert '--property=MemoryMax=8G' in cmd and '--property=CPUQuota=400%' in cmd
    assert '--property=KillMode=control-group' in cmd
    assert '--property=OOMPolicy=stop' in cmd
    assert not any(x in cmd for x in ['--pty','bash','sh'])
    with pytest.raises(ValueError):command('main;bad',argv,tmp_path/'log',directory=tmp_path)
