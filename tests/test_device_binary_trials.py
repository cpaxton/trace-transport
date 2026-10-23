import zipfile
from pathlib import Path
from scripts.device_binary_trials import png_payload,source_bundle


def test_device_probe_bundle_is_source_only_and_can_import_without_site_packages(tmp_path):
    import subprocess
    root=Path(__file__).resolve().parents[1]
    bundle=tmp_path/'source.zip'
    assert len(source_bundle(root,bundle))==64
    with zipfile.ZipFile(bundle) as archive:
        assert all(name.startswith('trace_transport/') and name.endswith('.py') for name in archive.namelist())
    result=subprocess.run(['python3','-S','-c',
        'import sys;sys.path.insert(0,sys.argv[1]);from trace_transport.binary_udp import receive_bytes',str(bundle)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert png_payload().startswith(b'\x89PNG\r\n\x1a\n')
    assert len(png_payload())>190000
