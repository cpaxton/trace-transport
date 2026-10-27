# V7 frozen-policy and credit diagnostics

Registered 2026-10-27. Smoke status: **running**, actual start 2026-10-27T23:26:56.437497+00:00; no results yet.
[Recorded run plan](../../artifacts/v7-policy-diagnostics-smoke-plan.json). Full profile remains planned.
This follow-up is stacked on PR #5 and does not train or select another model.
The completed 50k-decision study still has no demonstrated learning win.

We will test whether sampled deployment differs from greedy deployment, whether
training changed the sampled policy relative to initialization, and whether
truncated credit estimates preserve the direction of longer-horizon action value.
These are diagnostic questions, not acceptance criteria or optimizer changes.

## Fixed coverage

[Configuration](../../experiments/v7-policy-diagnostics-v1.json) fixes all model,
case and sampling seeds before execution. All six final models receive greedy
and sampled evaluation in the full profile. Sampling seeds 101–103 are reported
separately, with both the deterministic reference and the matched initialized
sampled policy. They are not pooled as independent network cases. The 24 prior
validation cases are explicitly reused for diagnosis; reserved acceptance sets
remain unopened. Report per-case quality, goodput, misses, corruption and fairness
through the existing evaluator; reward is a separate credit diagnostic.

For conditional credit, seed 42 of each architecture is fixed in advance rather
than selecting the best validation model. Four training cases cover wired LAN,
Wi-Fi contention, capacity opportunities and bufferbloat. At the first eligible
producer-0 decision at/after 2 seconds (before 6 seconds), clone the sampled-policy
prefix and force each of the three rate biases. Continue with the frozen sampled
policy until the true episode end, using 16 independent futures per action.
Already scheduled arrivals stay fixed; future packet/action streams are reseeded
independently. These are conditional independent-arm estimates, **not common
random numbers**. Bootstrap intervals describe finite simulation replicas only.

Compare full discounted return and MC advantage with the parent's time-discounted
GAE at 4 and 16 macro transitions. Report the corresponding conditional softmax
logit gradients. The critic observes a compressed state, actual PPO rollout
windows may start elsewhere, and shared network parameters aggregate decisions:
a disagreement is not by itself proof of a training bug or of an achievable win.
Skipped snapshots must remain visible. No family labels or future data enter the
actor. No action, reward, simulator physics or checkpoint weights change.

## Execution and artifacts

The development smoke uses two training cases, the same first-seed models, two
sampling seeds and four replicas/action: 18 performance episodes plus 48 credit
continuations (22 submitted tasks). It is an engineering check only. Full coverage
has 672 performance episodes plus 384 credit continuations (680 tasks).

Use an immutable source bundle from `scripts/caliban_deploy.py`, with its
`source-manifest.json` in the working directory. Full private checkpoints are
required for the critic; public actor-only exports fail closed. The runner checks
all recorded `src/` hashes against the bundle and records checkpoint SHA-256s.

```sh
python -m scripts.v7_policy_diagnostics \
  --config experiments/v7-policy-diagnostics-v1.json \
  --models-root /path/to/reactive-sender-v7-training \
  --output /path/to/new-diagnostic-run --profile smoke
```

Run on Caliban through `scripts.run_limited` with four workers, six CPU equivalents,
8 GiB memory, single-threaded BLAS/Torch, and a bounded runtime. Do not replace old
sources, editable installs, or checkpoint directories. Output includes the frozen
plan, task progress, raw per-case records, action distributions and conditional
credit replicas in `results.json`. Raw runs stay outside Git. Add actual UTC
start/end dates and curated artifacts to the register and paper after execution;
this document reports no new scientific result.
