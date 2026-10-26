# Can local history identify the right controller?

Actual start: 2026-10-26 UTC. **Running: data collection only.**
[Plan](../../artifacts/selector-coverage-v1-plan.json),
[frozen source](../../artifacts/selector-coverage-v1-source.zip),
[runner](../../scripts/selector_coverage.py).

The preceding [v6 control sweep](reactive-diagnostics-v6.md) found substantial
finite-control hindsight headroom but no single alternative passing all guards.
This motivates independent coverage before fitting or scaling a selector.

Caliban collects four development cases per family (48 total, seeds >=170000)
and two separate validation cases per family (24 total, seeds >=180000). Both
banks fail closed on missing families and are disjoint from one another and the
original pilot banks. This is synthetic development/validation, not the reserved
acceptance test set or measured robot Wi-Fi.

Each case runs all nine combinations of delay-fast/AIMD/BDP and ACK settings
(1,0), (8,5ms), (32,25ms): 648 full trials, each 12-second production plus 3-second
drain. Separate one-second delay-fast/immediate-ACK probes collect the last 64
observations from sender endpoint 0. Features contain only its local observations;
family, full network configuration and later outcomes are excluded from the tensor.
Whole-run candidate outcomes are stored separately as future diagnostic labels.
These labels cannot be actor inputs. A prefix cannot predict arbitrary future
shocks, so labels may be intrinsically ambiguous given the probe.

No model is trained yet. The next gate is tiny-data fitting and comparisons of
simple heuristics, MLP, GRU and transformer selection using the same local probes.
Report validation regret and guard violations, not just class accuracy. An offline
selection label is not proof that per-endpoint online switching works: mixed
endpoint choices, switching state, probe overhead, and coordination metadata must
be executed and charged in a subsequent evaluation. Existing controllers maintain
state such as cooldown and peak ACK rate; future switching must specify state
transfer, shadow updates and hysteresis rather than assuming free interchange.

The new source is immutable and explicitly inserted into the launcher/worker import
path. Caliban's old v4 source, editable installation and stopped service are
unchanged. CPU-only runtime: Python 3.12.3, Torch 2.14.1+cpu, NumPy 2.5.4. Service
`trace-selector-coverage`: six-core affinity (6–11), CPU quota 600%, 8 GiB maximum,
512 MiB swap, six-hour cap, no terminal attachment. JSON-only persistent mirroring
has a 128 MiB budget and preserves at least 1 GiB free space. No raw runs or device
credentials are published.

Check all Caliban studies with:

```bash
.venv/bin/python -m scripts.caliban_deploy status
```

The status output summarizes probe/control completion without dumping the dataset.
The failed v4 study remains incomplete and is not silently resumed.
