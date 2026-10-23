# Receiver experiments and device trials

[Documentation index](README.md) · [Project overview](../README.md)

## Receiver experiment: goals and win conditions

The hypothesis is that local arrival history helps a receiver trade feedback
overhead against sender stalls and recovery time. The largest opportunity is on
asymmetric paths where ACKs compete for scarce return capacity. On clean paths,
waiting to batch ACKs can make completion slower. There is no expected integrity
advantage: exact reconstruction is a requirement for every controller.

```bash
uv run --locked --extra learn python scripts/receiver_experiment.py
```

The experiment trains three small history transformers (seeds 42/43/44, 80
episodes each), with a frozen AIMD sender and two simultaneous Unicode transfers.
Receiver observations are 22 local features over eight time steps; they contain
arrivals, reordering, received ranges, pending feedback, and local timers, with
no sender state, link capacity, queue occupancy, or hidden loss probability.
Every 20 ms, two bounded actions choose a batch target of 1/2/4/8 ACK entries and
a maximum delay of 0–25 ms. Completion feedback flushes immediately, and a timer
flushes partial batches so the policy cannot permanently suppress feedback.

ACK batches carry actual received byte ranges and transmission IDs under a new
explicit packet kind (`ACK_BATCH=3`); older endpoints reject that kind. Encoded
metadata stays within 512 bytes, splitting batches as necessary. ACK delay is
carried on the wire and removed from the sender's RTT sample. This capability is
**simulator-only**; the UDP adapter retains its original immediate feedback.

Compare with immediate feedback and deterministic 2/4/8-entry delayed ACKs,
including an 8-entry/25-ms policy. Select the learned seed and the strongest
deterministic baseline using validation seeds 1000–1004, completion first and
then common transport-v2 return. Freeze those choices before testing seeds
2000–2019 on clean, asymmetric, lossy, changing-capacity, and outage cases.
Training excludes the outage preset. Scenario families otherwise overlap; this
is held-out randomness, not evidence of broad out-of-distribution generalization.

**Win condition:** both selected policies complete every test transfer exactly,
and the learned receiver either reduces completion time with a paired 95%
confidence interval below zero without increasing mean feedback bytes, or
reduces feedback bytes with a paired 95% interval below zero while the completion
time interval's upper bound stays within +5%. Bootstrap whole paired seeds,
keeping dependent flows and scenarios together. Report failures, total data
bytes, retransmissions, and undiscounted team return too. A better reward alone
does not establish a win. These small diagnostic intervals are not a deployment
claim, and simulated time currently excludes neural inference cost.

The runner writes its configuration, training logs, validation selection, raw
paired results, and receiver checkpoints to `runs/receiver-v1/`. Receiver and
sender checkpoints have distinct schemas and cannot be interchanged. Learned
metadata, retransmission, redundancy, and joint self-play are subsequent
experiments, each requiring an ablation against tuned deterministic controls.

#### Receiver-v1 result

The [recorded experiment](../artifacts/receiver-v1-results.json) **did not meet the
win condition**. Validation selected learned seed 43 and delayed-8/25-ms feedback.
On the reserved test seeds:

| Receiver | Exact completed transfers | Feedback bytes | Mean elapsed seconds |
| --- | ---: | ---: | ---: |
| Immediate ACK | 215/240 | 3,765,162 | 4.788 |
| Learned receiver, seed 43 | 223/240 | 3,099,580 | 4.354 |
| Deterministic batch 8, delay 25 ms | 240/240 | 2,505,897 | 3.777 |

All reconstructed payloads were correct. Elapsed time includes deadline duration
for failed transfers; it is not a successful-transfer latency statistic. Every
learned-policy failure occurred on the 24-kbit/s reverse path. Relative to the
validation-selected baseline, the learned receiver used 23.7% more feedback and
had 15.3% greater mean elapsed time. The useful gain here came from deterministic
batching, not learning. Three short training runs do not establish that learning
cannot help; they establish that this training budget and policy did not help.
The [candidate checkpoint](../artifacts/receiver-v1-candidate.pt) is retained for
reproduction, not promoted as a better default. The [configuration](../experiments/receiver-v1.json)
records training budgets, scenarios, seed splits, selection, and win conditions.

## Receipt recovery and receiver-v2

The new opt-in `ACK_RECEIPT=4` packet repeats the contiguous received prefix and
up to eight selective byte intervals. Losing an ACK no longer permanently loses
that receipt information: subsequent summaries repeat it. Summaries cannot credit
unsent segments, and only an explicitly echoed transmission supplies an RTT
sample. A completion receipt combines the full prefix with the message digest,
so one valid confirmation can finish the entire transfer despite earlier ACK
loss. The first completion receipt flushes immediately; subsequent duplicate
responses follow the bounded ACK scheduler. Metadata remains within 512 bytes.
Existing per-transmission ACK and batch modes remain available for comparisons.

Receiver-v2 compares three training profiles, each with three seeds and 80
episodes: preset networks with eight history steps, randomized persistent
degradation with eight steps, and the same randomized profile with one step.
The one-step policy still sees counters and timers maintained by the endpoint;
it tests the value of additional neural history, not an entirely memoryless host.
Randomized flows range from one to four, so equal episode budgets do not imply
equal transition or compute budgets. Validation uses seeds 3000–3004; diagnostic
tests use 4000–4019; the separate generalization suite uses 5000–5019. These are
new splits relative to receiver-v1. Selection remains completion, then return,
and no candidate is selected using the generalization suite.

| Receiver-v2 policy | Diagnostic completions | Generalization completions | Named Wi-Fi subset |
| --- | ---: | ---: | ---: |
| Preset training, history 8 | 239/240 | 551/560 | 159/160 |
| Persistent randomized training, history 8 | 238/240 | 551/560 | 160/160 |
| Persistent randomized training, history 1 | 238/240 | 552/560 | 160/160 |
| Deterministic batch 8, delay 25 ms | **240/240** | **553/560** | **160/160** |

No wrong payloads occurred. None of the learned candidates meets the win
condition. Longer history and domain randomization have not shown an overall
advantage in this small experiment. The generalization suite mixes new parameter
combinations with conditions beyond training support: six flows, 128-KiB
messages, a 12-kbit/s reverse path, longer RTTs, smaller queues, and stronger loss
bursts. The deterministic baseline also times out on the hardest reverse path;
these failure-inclusive results are not claims about successful-transfer latency.

A separate ablation holds the original receiver-v1 weights fixed and changes
only the feedback protocol. On the new diagnostic seeds, completion rises from
223/240 to 239/240 with receipt summaries. That supports the recovery change,
without attributing its benefit to retraining or randomization. The default UDP
listener retains its prior feedback; `listen --receipt-feedback` enables receipt
summaries for updated peers. Learned receiver scheduling remains simulator-only.

The [raw results](../artifacts/receiver-v2-results.json),
[training logs](../artifacts/receiver-v2-training.json.gz),
[configurations](../experiments/receiver-v2.json), and three experimental candidate
checkpoints are retained. To reproduce after installation:

```bash
trace-transport receiver-experiment --protocol receipts --profile presets \
  --validation-seed 3000 --test-seed 4000 --output runs/receiver-v2-presets
trace-transport receiver-experiment --protocol receipts --profile randomized \
  --validation-seed 3000 --test-seed 4000 --output runs/receiver-v2-randomized
trace-transport receiver-experiment --protocol receipts --profile randomized --history 1 \
  --validation-seed 3000 --test-seed 4000 --output runs/receiver-v2-memoryless
uv run --locked --extra learn python scripts/receiver_generalization.py
```

## Beyond receiver-v2: feedback timing and stronger tests

The next [experiment batch](experiments/protocol-v3.md) found a larger
improvement in feedback timing and sender timers. On fresh diagnostic seeds,
configuring a 250-ms feedback-wait budget at both endpoints and using a longer
ACK wait reduced mean completion time **37%**, ACK bytes **70%**, and total wire
bytes **32%**, with **240/240 exact completions**. A separate generalization run
improved from 551/560 to 560/560 completions. These are deterministic protocol
gains; neither the new categorical receiver nor a learned pacing/window
correction has beaten the strongest tuned controls.

The browser now offers **Receipt summaries · latency focused** and **Receipt
summaries · fewer ACKs**, alongside legacy ACKs. The latency-focused profile is
the browser default. Longer waits can save traffic while slowing some paths;
the report records the selected profile and configured timer budget. CLI:

```bash
trace-transport benchmark --feedback-profile budgeted --reward completion
trace-transport benchmark --feedback-profile sparse --reward completion
```

These profiles are simulation settings with a shared configured delay budget,
not on-wire negotiation or deployed learned receiver behavior. The experiment
report includes control sweeps, 320-episode training runs for three seeds,
validation-selected checkpoints, a total-wire accounting check, and a new
mixed bulk/short-message challenge. Keeping that stronger test set exposes
remaining delivery failures and multi-second short-message tail latency.

## Wi-Fi model structure

Randomization happens **between episodes**; within an episode the hidden service
conditions persist and evolve. A degradation has a randomized onset, a gradual
decline, a sustained poor interval, and recovery. Forward/reverse capacity,
residual loss probability, and access delay move together. Packet drops are
random conditional on that persistent state. FIFO queues add traffic-dependent
delay and overflow, so actions affect subsequent observations. Actors receive
only local endpoint telemetry, never the hidden schedule or link parameters.

The browser scenario menu includes `wifi-steady`, `wifi-contention`,
`wifi-mobile-fade`, and `wifi-burst-loss`. These model available service above the
MAC layer; they do not implement 802.11 backoff, rate selection, RF propagation,
or MAC retransmissions. They are **stylized, uncalibrated Wi-Fi conditions**.
Capacity and impairment schedules are included in reports for inspection.
The frozen `reference-v1` benchmark manifest remains unchanged.

History can be useful if recent arrival gaps, duplicates, and progress indicate
that a bad interval is developing or persisting. That is a hypothesis to test,
not a guarantee that the current transformer has learned it. The one-step
comparison above currently provides no evidence that the longer window helps.
Next, record real Wi-Fi traces, fit persistence/recovery and burst statistics,
reserve entire traces/devices for testing, and report reaction lag and recovery
cost alongside completion and wire overhead. Avoid fitting and testing on
different slices of the same recording.

## Real devices: Herman and Caliban

`herman` (mobile robot, Wi-Fi) and `caliban` (Jetson Orin, Wi-Fi/Ethernet) are
planned physical targets. Install the same checkout on each device with the
curl installer. The local SSH username/password are kept in ignored
`.env.devices` with owner-only permissions; credentials are never part of the
source export. Configure SSH authentication separately using keys or an
authenticated ControlMaster connection. The harness uses noninteractive SSH.

```bash
# Replace the address with the robot's UDP-reachable LAN address.
uv run --locked python scripts/device_trials.py --ssh jetson1@herman \
  --address ROBOT_IP --condition wifi-stationary --trials 10 \
  --interface wlan0 --report runs/devices/herman-wifi.json

uv run --locked python scripts/device_trials.py --ssh caliban \
  --address 192.168.1.55 --condition ethernet --trials 10 \
  --report runs/devices/caliban-ethernet.json
```

The harness starts a listener on an assigned UDP port, waits for readiness, sends
Unicode text, verifies the receiver's SHA-256 against the source, and saves
per-trial completion, goodput, retransmission, RTT, and feedback counters. It
retains uniquely named remote payload files in `/tmp`. `--interface` snapshots
`iw` link telemetry before and after the batch. `--condition` is an operator
label: it does not change the device's connection or move the robot. Reverse the
roles for upload/download measurements, and record separate stationary/moving
Wi-Fi and Ethernet sessions. This session's restricted network prevents live
device validation; adapter recovery is tested using in-memory sockets.

Current transfers prioritize complete, exact text. Streaming video needs a
separate workload with frame deadlines and stale-frame handling; control/status
messages may require different delivery priorities. Those streaming semantics
are not implemented by this single-message harness.
