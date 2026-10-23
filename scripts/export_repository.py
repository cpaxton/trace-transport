"""Export a reviewable source-only Git repository, bundle, and archive.

The selected synthetic-data checkpoint and compact reports are included; runtime
environments, credentials, other checkpoints, and raw runs are excluded. --push
initializes the already-created, empty GitHub repository using ordinary Git.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def source_paths(root):
    root = Path(root).resolve()
    paths = [root / name for name in ("README.md", ".gitignore", "pyproject.toml", "uv.lock", ".python-version", "LICENSE",
                                      "paper/main.tex", "paper/references.bib", "artifacts/SHA256SUMS")]
    for pattern in ("src/**/*.py", "src/trace_transport/scenarios/*.json", "src/trace_transport/web/*.html",
                    "src/trace_transport/web/*.css", "src/trace_transport/web/*.js", "tests/*.py",
                    "scripts/*.py", "scripts/*.sh", "experiments/*.json", ".github/workflows/*.yml", "artifacts/*.json",
                    "artifacts/robust-v2-best.pt", "artifacts/receiver-v1-candidate.pt"):
        paths.extend(root.glob(pattern))
    return sorted({path.relative_to(root) for path in paths if path.is_file()})


def git(directory, *arguments):
    return subprocess.run(["git", "-C", str(directory), *arguments], check=True,
                          capture_output=True, text=True).stdout.strip()


def export(root, output, *, repository="https://github.com/cpaxton/trace-transport.git"):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="trace-source-", dir=output))
    files = source_paths(root)
    hashes = {}
    for relative in files:
        content = (root / relative).read_bytes()
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        hashes[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    git(staging, "init", "-b", "main")
    git(staging, "config", "user.name", "cpaxton")
    git(staging, "config", "user.email", "3286631+cpaxton@users.noreply.github.com")
    # The staging directory contains only the explicit allowlist. Force-add so
    # a user's global ignore rules cannot silently omit the model or CI files.
    git(staging, "add", "--force", "--all")
    git(staging, "diff", "--cached", "--check")
    git(staging, "commit", "-m", "feat: add Trace Transport simulator, learned policies, and local test bench")
    git(staging, "remote", "add", "origin", repository)
    bundle = output / "trace-transport.bundle"
    archive = output / "trace-transport-source.tar.gz"
    git(staging, "bundle", "create", str(bundle), "--all")
    git(staging, "archive", "--format=tar.gz", "--output", str(archive), "HEAD")
    metadata = {"repository": repository, "staging_repository": str(staging),
                "commit": git(staging, "rev-parse", "HEAD"), "files": hashes,
                "bundle": str(bundle), "archive": str(archive)}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("dist/publication"))
    parser.add_argument("--repository", default="https://github.com/cpaxton/trace-transport.git")
    parser.add_argument("--push", action="store_true", help="push to an empty repository using configured Git credentials")
    args = parser.parse_args()
    try:
        metadata = export(args.root, args.output, repository=args.repository)
    except subprocess.CalledProcessError as exc:
        parser.exit(1, f"Git export failed: {exc.stderr.strip()}\n")
    print(json.dumps({key: value for key, value in metadata.items() if key != "files"}, indent=2))
    print(f"Exported {len(metadata['files'])} reviewed source/artifact files.")
    if args.push:
        stage = metadata["staging_repository"]
        try:
            if git(stage, "ls-remote", "origin"):
                parser.error("destination is not empty; clone it normally to make further updates")
            git(stage, "push", "--set-upstream", "origin", "main")
        except subprocess.CalledProcessError as exc:
            parser.exit(1, f"GitHub push failed: {exc.stderr.strip()}\nThe bundle and source archive remain available.\n")
        print(f"Published: {args.repository}")


if __name__ == "__main__":
    main()
