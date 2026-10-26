# Next experiment: role-aligned reactive sender control

Status: **planned, not started**. Written 2026-10-26 after PR #4 merged. No new
training result or compute launch is claimed. This follows the completed
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
