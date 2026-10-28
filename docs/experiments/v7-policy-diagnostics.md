# V7 frozen-policy and credit diagnostics

Registered 2026-10-27. Smoke status: **complete, no demonstrated learning win**. Actual start/end:
2026-10-27 23:26:56–2026-10-28 00:00:27 UTC.
[Recorded run plan](../../artifacts/v7-policy-diagnostics-smoke-plan.json). Full profile remains planned.
This follow-up to merged PR #5 does not train or select another model.
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
credit replicas in `results.json`. Raw training runs stay outside Git. The completed smoke results below, their
actual UTC dates, and their limitations are recorded in the register and paper.

## Completed development smoke

[Results](../../artifacts/v7-policy-diagnostics-smoke-results.json): all 22 tasks
completed successfully in 33 minutes 31 seconds: 18 performance episodes and 48
conditional continuations. The seven diagnostic tests passed on Caliban. Only two
training cases (wired LAN and bufferbloat), one optimizer seed per architecture,
two sampling seeds and four replicas/action were used; this is not held-out
acceptance or a broad performance comparison.

| Deployment | Verified goodput change vs teacher | Change vs matched initial sampled policy |
| --- | --- | --- |
| Both greedy trained actors | 0% | Not applicable |
| Current MLP sampled, seed 101 | +0.34% | −0.70% |
| Current MLP sampled, seed 102 | +0.35% | +0.22% |
| History MLP sampled, seed 101 | +0.12% | −0.91% |
| History MLP sampled, seed 102 | −0.35% | −0.49% |
| Initial sampled, seed 101 | +1.04% | Baseline |
| Initial sampled, seed 102 | +0.13% | Baseline |

All evaluated modes had zero wrong deliveries, 100% completion and zero
interactive misses on these two training cases. All acceptance comparisons return
false. The small gains are not consistent improvements over initialization;
case-bootstrap intervals over only two cases provide very limited evidence.
Sampling seeds stay separate, rather than artificially enlarging the case count.

The credit probe supplies an exploratory lead. For the current MLP in bufferbloat,
forcing bias 1.2 increased estimated discounted continuation return by 0.202
vs bias 1.0 (independent-arm bootstrap 95% interval [0.054, 0.353]). Its estimated
full-return teacher-logit gradient was negative (−0.0326), whereas the 4/16-step
GAE gradients were positive (+0.00566/+0.00260). For the history MLP in wired LAN,
bias 1.2 increased return by 0.111 [0.020, 0.243], but its 4/16-step GAE logit
gradients for that action were negative. These are different on-policy prefixes;
they are not matched architecture comparisons. Other action intervals include
zero. Intervals are exploratory, based on four replicas, without multiple-testing
correction; no robust action ranking or policy improvement follows.

Both wired snapshots also have critic predictions around 7.5–7.8 versus estimated
conditional full-return values around 35.7–35.8. The critic observes a compressed
summary, not the full snapshot; this discrepancy does not prove a code bug.
The result motivates checking value estimation and credit horizons with more
replicas and states before another unchanged training run. It does not show that
longer GAE alone will solve the task. Full-profile diagnostics remain planned and
have not run; no new training or reserved acceptance evaluation was launched.
