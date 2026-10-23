from pathlib import Path
import os
import subprocess

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/install.sh'


def test_installer_exposes_locked_environment_command_and_reuses_checkout(tmp_path):
    checkout = tmp_path / 'checkout with spaces'
    checkout.mkdir()
    subprocess.run(['git', 'init', str(checkout)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(checkout), 'remote', 'add', 'origin',
                    'https://github.com/cpaxton/trace-transport.git'], check=True)
    executable = checkout / '.venv/bin/trace-transport'
    executable.parent.mkdir(parents=True)
    executable.write_text('#!/bin/sh\nexit 0\n')
    executable.chmod(0o755)
    commands = tmp_path / 'commands'
    commands.mkdir()
    uv = commands / 'uv'
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$INSTALL_LOG"\n')
    uv.chmod(0o755)
    env = dict(os.environ, PATH=str(commands) + ':' + os.environ['PATH'],
               TRACE_TRANSPORT_BIN_DIR=str(tmp_path / 'bin'), INSTALL_LOG=str(tmp_path / 'log'))
    for _ in range(2):
        subprocess.run(['bash', str(SCRIPT), '--dir', str(checkout)], env=env, check=True, capture_output=True)
    assert (tmp_path / 'bin/trace-transport').resolve() == executable
    assert 'sync --locked --extra learn' in (tmp_path / 'log').read_text()
    (tmp_path / 'bin/trace-transport').unlink()
    (tmp_path / 'bin/trace-transport').write_text('unrelated command')
    result = subprocess.run(['bash', str(SCRIPT), '--dir', str(checkout)], env=env, capture_output=True)
    assert result.returncode != 0
    assert (tmp_path / 'bin/trace-transport').read_text() == 'unrelated command'
