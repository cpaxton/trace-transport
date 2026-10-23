from pathlib import Path
import subprocess

from scripts.export_repository import export, source_paths


def test_source_export_excludes_local_runtime_and_credentials(tmp_path):
    root = tmp_path / "project"
    for name in ["README.md", "pyproject.toml", "src/trace_transport/web/app.js", ".aws/credentials",
                 ".venv/secret.py", ".env.devices", "runs/local.json", "checkpoints/local.pt"]:
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("example\n")
    paths = [path.as_posix() for path in source_paths(root)]
    assert paths == ["README.md", "pyproject.toml", "src/trace_transport/web/app.js"]
    metadata = export(root, tmp_path / "output")
    checkout = tmp_path / "clone"
    subprocess.run(["git", "clone", metadata["bundle"], str(checkout)], check=True, capture_output=True)
    assert (checkout / "README.md").read_text() == "example\n"
    assert not (checkout / ".aws").exists()
    assert Path(metadata["archive"]).is_file()
