"""Dependency-free atomic JSON output for simulator diagnostics and studies."""
import json
from pathlib import Path


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
