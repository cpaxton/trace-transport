# Repository workflow

- Work on a named feature or fix branch. Never commit or push directly to `main`.
- Publish changes through a pull request targeting `main`.
- Do not bypass branch protection or weaken required checks to publish a change.
- Do not merge a pull request unless the user explicitly requests the merge.
- Run checks appropriate to the change and report any incomplete verification.
- Keep credentials, device passwords, local environments, and raw training runs out of Git.
- Separate training reward from held-out performance. Report unfinished studies and negative results honestly.

- Maintain `docs/experiments/register.md`: add actual start/end dates for new experiments, link configurations and result artifacts, and keep planned, running, completed, retired and negative work distinct. Update the register and paper when reporting new scientific results.
