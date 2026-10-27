# Next experiment: role-aligned reactive sender control

Status: **small and 50k studies complete; no demonstrated gain**. The proposal
was written 2026-10-26 after PR #4 merged. The separately registered
[initial pilot](reactive-sender-v7.md) launched 2026-10-27 UTC; no accepted learning
result is claimed. This follows the completed
[selector study](selector-learning-v1.md), rather than tuning its validation bank.

## Question

Can local history improve safe short-term adaptation once training roles, actions
and deployment roles agree? The prior selector memorized development labels,
generalized poorly, and applied bulk-endpoint labels across unlike endpoint roles.
It also chose once from a one-second prefix, so independently randomized future
changes could not be predicted. Those are confounds to remove before scaling RL.

## Narrow first contract

- Train and deploy only at bulk-producing sender roles. Collect examples from every
  deployed producer endpoint, not only endpoint 0. Reverse/interactive senders,
  receiver ACK policy, reliability and queue scheduling stay deterministic.
- Choose a bounded bias on the teacher's **target rate**: initially 0.8, 1.0 or
  1.2. Do not repeatedly multiply the current rate at each 2-ms kernel tick.
  A persistent target bias must not become exponential open-loop acceleration.
- Reconsider the bias at roughly 100 ms, extended using measured RTT when needed.
  The deterministic controller still handles packet-level pacing and recovery.
  Hold/update rules and inference delay must be explicit in the new policy schema.
- Start with small current-observation MLP and history models. No larger transformer
  until history adds value. Existing v6 checkpoints are not relabeled or reused as
  if they implement this policy contract.

The network task, exact-delivery verifier and goodput score remain the frozen
ones. A new controller adapter must preserve teacher behavior exactly at bias 1;
if physics or observations change, version that change and retain old profiles.
Never call a mutating baseline twice to compare its outputs.

## Data and decision targets

Collect decisions at multiple times along feasible changing network episodes,
including fades, reverse congestion, recovery and capacity changes. Regimes should
persist long enough for feedback to be informative. Future independent shocks
remain unpredictable; actors receive no family tags, event schedules, router
queues, true bandwidth or future outcomes.

Use a fixed physical-time history (initially one second), with elapsed-time masks
and field-aware aggregation. Integrate byte/rate counters rather than sparsely
sampling them; distinguish instantaneous values from interval rates. Do not expose
absolute episode time. Verify aggregation against original counters and preserve
causal ordering.

For a diagnostic action-value dataset, compare legal interventions at the **same
producer state** with the other endpoints on the teacher. Use uncertainty-aware
finite rollouts and report replica variance. Cloned RNG streams diverge once
packet draws differ; do not claim common-random-number coupling. A finite-horizon
rollout is a training diagnostic, not an optimal oracle or a proof that shaped
local labels preserve the full-task optimum. Partial bytes are never evaluation
success. Any learned objective or potential must be documented separately from
whole-object scoring.

Targets must describe the action that that endpoint actually executes. Whole-case
uniform-controller winners cannot silently supervise independent local actions.
Conservative teacher deferral can be calibrated on development folds; validation
must not set thresholds. No confidence rule guarantees unseen-case safety.

## Gates before a long run

1. Unit/integration checks: bias-1 transport parity; correct producer-role masking;
   no neural actions on reverse roles; target bias cannot compound per tick;
   causal physical-time aggregation; strict checkpoint schema; simulation costs
   include probes/feedback and the declared inference-delay model.
2. Development intervention audit: show action-induced differences exceed ordinary
   replica noise in realistic feasible states. Include a simple hand-coded reactive
   bias rule and a tuned deterministic target as strong comparators. If there is
   no guarded headroom, do not enlarge the model or train longer.
3. Tiny repeated-case fit with matched deployment roles and held-out episodes.
   Test shuffled/masked history against real history. A supervised tiny fit does
   not by itself validate PPO credit assignment.
4. Only after these pass: a registered three-seed pilot, capped initially at 50k
   macro decisions per seed. Also report physical time, distinct worlds, episodes,
   kernel ticks and wall-clock cost. Endpoint opportunities are not interchangeable
   with macro decisions. Increase the budget only on consistent development gains.

Before collection, allocate and commit explicit disjoint development, calibration
and prospective validation seed banks after checking existing study allocations.
Split whole episodes/paths, not correlated decision windows. The completed selector
validation cases must not become the new claim of independent generalization.
The original reserved acceptance sets remain unopened.

## Win conditions and alternatives

Keep the existing >=10% verified-goodput gain, positive paired case-level interval,
and all deadline, integrity, service, completion, wire and fairness guards.
Report training-seed variation separately. No threshold is relaxed to make the
new controller win; preliminary validation is still not final acceptance.

If this narrower reactive problem still has little observable headroom, keep the
strong deterministic transport. A distinct practical task is robot image freshness
and dependency-aware map scheduling: opt-in stale-image replacement with exact
recovery of accepted objects, compared with latest-only/EDF/fair-queue baselines.
That requires its own objective and prospective evaluation, not relabeling this
throughput experiment as a success.


## Initial Caliban implementation

The first registered implementation is a bounded engineering/pilot job, not the
long 50k-decision study. [Configuration](../../experiments/reactive-sender-v7-pilot.json)
allocates 24 training, 12 unused calibration, and 12 new validation cases, checked
against prior public seed allocations. All six final policies (two MLP variants x
three optimizer seeds) are reported without validation-based selection.

Fixed physical-time bins integrate decoded ACK/wire/loss rates and retain only
past instantaneous state at each bin boundary. Interval rates are retrospectively
spread uniformly over their measurement intervals; this is an approximation, not
packet-timestamp recovery. Actors never consume data later than their decision.
The critic additionally summarizes applied bias, hold and pending control state;
those centralized fields never enter actors.

The adapter changes only rate targets, with the teacher window unchanged. Choices
are reconsidered every 100 ms or twice the endpoint's measured RTT (bounded at
500 ms). An explicit 20-ms inference/dispatch delay is charged at the next kernel
boundary. On-device warm p95 actor timing must fit that budget to advance; timing
is descriptive, not a real-time deployment certification. Initialization samples
the teacher with probability 0.6 and each bounded alternative with 0.2; greedy
initialization is exactly the teacher.

A development-only two-replica intervention audit and role-aligned tiny fits gate
the budget. A positive signal plus exact tiny fits and on-device timing allows
2,000 producer macro decisions per run. Otherwise each run is limited to 512
engineering decisions, with no scaling justification. These are deliberately small
smokes/pilots; two replicas do not establish a reliable causal advantage. Longer
training still requires the stronger gates above. Raw task reward, physical-time
GAE, no progress shaping and no paired-teacher subtraction are used.

The initial collector uses 16 macro steps per rollout to checkpoint promptly.
Policy, critic and sampling use separate recorded seeds; critic initialization is
identical across model kinds at the same optimizer seed.

Runs retain initial/latest/final checkpoints and report macro decisions, kernel
steps, distinct training cases, completed episodes, reward and actor latency.
Training reward is separate from final-checkpoint validation. Frozen source,
configuration, exact actual budgets and gate outcomes are recorded before remote
training. No v7 training has started at the time this implementation is registered.


## Training-budget extension after the engineering pilot

The 2k-decision smoke is too short to test greedy adaptation: a training-weight-only
bound proves its completed heads still select the teacher over bounded finite hidden states.
This is not evidence against reactive control. The pre-existing 50k cap will be
used in a separate frozen run, with the same models, optimizer, reward and quality
guards. This explicitly amends the resource gate: numerical/role/timing checks,
small development intervention signals and the training-head bound justify a
meaningful exposure budget; pilot validation scores do not select a model or tune
parameters. No validation improvement is asserted to justify scaling.

[Extended config](../../experiments/reactive-sender-v7-training.json) fixes six
runs at 50k producer decisions each if the source-bound development/timing checks
pass, otherwise the engineering fallback. Training coverage expands from 24 to 48
cases (including the original 24); 24 prospective validation cases are allocated
before launch, and the earlier pilot validation is not reused. This is not an
isolated budget-only causal comparison because coverage also expands. The same
development sensitivity states are replayed to bind new configuration/source;
this is not independent replication. The old small job and source stay unchanged.

The [50k study](reactive-sender-v7-training.md) launched on Caliban at
2026-10-27 01:31:44 UTC with actual budgets/source/timing checks recorded. No final
performance result is claimed.

The 50k study completed 2026-10-27 at 04:44:41 UTC with no gain. See the
[complete outcome](reactive-sender-v7-training.md#complete-result-2026-10-27-utc);
no unchanged larger training run is planned from this result.
