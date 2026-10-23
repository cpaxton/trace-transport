# Contributing

Create a branch from current `main`, make the change, and open a pull request:

```bash
git switch main
git pull --ff-only
git switch -c feat/describe-the-change
# edit, verify, and commit
git push -u origin HEAD
gh pr create --base main
```

Do not push directly to `main`. Branch protection applies to administrators too.
Required checks are `simulator (3.10)`, `simulator (3.12)`, and
`learned-policies`. The branch must be up to date, review conversations resolved,
and all checks successful before merging. Force pushes and deletion of `main`
are disabled. Use squash merge; GitHub deletes the merged branch automatically.

Pull requests currently require zero independent approvals so the repository
owner can merge their own work. This still requires a PR and passing CI. Once a
second maintainer can review changes, raise the required approval count to one.
Agents must leave PRs open unless explicitly instructed to merge them.

For local verification:

```bash
uv sync --locked --extra learn
uv run --locked --extra learn pytest -q
node --check src/trace_transport/web/app.js
node --check src/trace_transport/web/joint.js
bash -n scripts/install.sh
```

Describe the problem, resulting behavior, and checks performed. For experimental
changes, distinguish training metrics, engineering acceptance, validation, and
held-out results. Never commit device credentials or raw local run directories.
