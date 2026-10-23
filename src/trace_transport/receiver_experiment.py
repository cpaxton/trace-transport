"""Train and evaluate a receiver with a frozen AIMD sender.

trace-transport receiver-experiment --protocol receipts --profile randomized
Selection uses validation only; final test is evaluated after selection.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from trace_transport.learning import generalized_advantages
from trace_transport.receiver_learning import episode, make_model, save, load
from trace_transport.simulator import NetworkConfig
from trace_transport.receiver_curriculum import DOMAIN, sample_receiver_episode


def scenarios():
    return {
        'clean': NetworkConfig(loss=0, ack_loss=0, jitter=.002, deadline=12),
        'asymmetric-64k': NetworkConfig(feedback_bandwidth_bps=64000, loss=.02, ack_loss=.02, deadline=12),
        'asymmetric-24k': NetworkConfig(feedback_bandwidth_bps=24000, loss=.02, ack_loss=.02, deadline=12),
        'lossy': NetworkConfig(loss=.1, ack_loss=.15, corruption=.01, ack_corruption=.01, deadline=15),
        'changing': NetworkConfig(feedback_bandwidth_bps=64000, capacity_changes=((.3, 250000), (.8, 2000000)), deadline=15),
        'outage': NetworkConfig(feedback_bandwidth_bps=48000, outages=((.2, .5),), ack_loss=.1, deadline=15),
    }


def train(seed, episodes, output, *, receipt_feedback=False, profile="presets", history=8):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = make_model(seed, history=history)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    buffer = []
    logs = []
    start = time.monotonic()
    for index in range(episodes):
        name = rng.choice(['clean', 'asymmetric-64k', 'asymmetric-24k', 'lossy', 'changing'])
        config = scenarios()[name]
        flows, payload_bytes = 2, rng.choice([16384,32768,49152])
        if profile == 'randomized':
            name = 'randomized'
            config, flows, payload_bytes = sample_receiver_episode(rng)
        results, scores, trajectory = episode(config, seed=seed*100000+index,
                                               payload_bytes=payload_bytes, flows=flows,
                                               model=model, stochastic=True, receipt_feedback=receipt_feedback)
        h, valid, latent, oldp, values, rewards, terminal, active = [torch.stack(x) for x in zip(*trajectory)]
        advantage, returns = generalized_advantages(rewards, values, terminal, torch.zeros(flows), gamma=.999)
        mask = active.flatten()
        buffer.append((h.flatten(0, 1)[mask], valid.flatten(0, 1)[mask], latent.flatten(0, 1)[mask],
                       oldp.flatten()[mask], advantage.flatten()[mask], returns.flatten()[mask]))
        if len(buffer) >= 4 or index == episodes-1:
            tensors = [torch.cat(x) for x in zip(*buffer)]
            hist, valid, latent, oldp, adv, returns = tensors
            adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
            stop = False
            for _ in range(2):
                for ids in torch.randperm(len(adv)).split(256):
                    logp, entropy, value = model.evaluate(hist[ids], valid[ids], latent[ids])
                    ratio = (logp - oldp[ids]).exp()
                    loss = -torch.minimum(ratio*adv[ids], ratio.clamp(.8,1.2)*adv[ids]).mean()
                    loss = loss + .5*(value-returns[ids]).square().mean() - .001*entropy.mean()
                    if ((ratio-1) - (logp-oldp[ids])).mean().item() > .015:
                        stop = True
                        break
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                    optimizer.step()
                if stop:
                    break
            buffer.clear()
        logs.append(dict(episode=index, scenario=name, flows=flows, network=asdict(config), payload_bytes=payload_bytes, completed=sum(r.completed for r in results),
                         score=float(scores.mean())))
        if (index+1) % 20 == 0:
            print(f'seed={seed} episodes={index+1}/{episodes} last20_completion={sum(x["completed"] for x in logs[-20:])}/{sum(x["flows"] for x in logs[-20:])}', flush=True)
    version = 'receiver-v2' if receipt_feedback else 'receiver-v1'
    path = output / f'{version}-s{seed}.pt'
    save(path, model, dict(seed=seed, episodes=episodes, sender='aimd', reward='transport-v2',
                          wall_seconds=time.monotonic()-start, receipt_feedback=receipt_feedback, profile=profile))
    (output / f'train-s{seed}.json').write_text(json.dumps(logs, indent=2)+'\n')
    return path


def evaluate(model, baseline, seeds, *, receipt_feedback=False, feedback_delay_budget=None):
    records = []
    for name, config in scenarios().items():
        if feedback_delay_budget is not None:
            config=replace(config,feedback_delay_budget=feedback_delay_budget)
        for seed in seeds:
            results, score, _ = episode(config, seed=seed, model=model, baseline=baseline, receipt_feedback=receipt_feedback)
            records.append(dict(scenario=name, seed=seed,
                completed=sum(r.completed for r in results), transfers=len(results),
                wrong=sum(r.received_text is not None and r.received_text != r.sent_text for r in results),
                elapsed=float(np.mean([r.elapsed_seconds for r in results])),
                feedback_bytes=sum(r.feedback_wire_bytes for r in results),
                data_bytes=sum(r.data_wire_bytes for r in results),
                retries=sum(r.retransmitted_payload_bytes for r in results),
                score=float(score.mean())))
    return records


def summary(records):
    return dict(completed=sum(r['completed'] for r in records), transfers=sum(r['transfers'] for r in records),
                wrong=sum(r['wrong'] for r in records), mean_score=float(np.mean([r['score'] for r in records])),
                mean_elapsed=float(np.mean([r['elapsed'] for r in records])),
                feedback_bytes=sum(r['feedback_bytes'] for r in records))


def paired_comparison(learned, baseline):
    """Bootstrap paired seeds as blocks, retaining all scenarios in each block."""
    assert [(r['scenario'], r['seed']) for r in learned] == [(r['scenario'], r['seed']) for r in baseline]
    seeds = sorted({r['seed'] for r in learned})
    rng = np.random.default_rng(12345)
    comparisons = {}
    for metric in ['elapsed', 'feedback_bytes', 'data_bytes', 'score']:
        differences = np.array([np.mean([a[metric]-b[metric] for a,b in zip(learned,baseline)
                                        if a['seed'] == seed]) for seed in seeds])
        draws = rng.choice(differences, size=(2000, len(seeds)), replace=True).mean(axis=1)
        reference = np.mean([r[metric] for r in baseline])
        comparisons[metric] = dict(mean_delta=float(differences.mean()),
                                  ci95=[float(x) for x in np.quantile(draws,[.025,.975])],
                                  relative_delta=float(differences.mean()/max(abs(reference),1e-9)))
    ls, bs = summary(learned), summary(baseline)
    exact = ls['wrong'] == 0 and ls['completed'] == ls['transfers'] == bs['completed']
    faster = comparisons['elapsed']['ci95'][1] < 0
    leaner = comparisons['feedback_bytes']['ci95'][1] < 0
    # The latency budget applies to its confidence bound, not just its mean.
    bounded_latency = comparisons['elapsed']['ci95'][1] <= .05 * bs['mean_elapsed']
    return dict(metrics=comparisons, all_exact_and_complete=exact,
                diagnostic_win=exact and ((faster and comparisons['feedback_bytes']['mean_delta'] <= 0)
                                         or (leaner and bounded_latency)),
                bootstrap='2000 paired seed-block resamples; equal scenario weight; diagnostic CI')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=['v1','receipts'], default='v1')
    parser.add_argument('--profile', choices=['presets','randomized'], default='presets')
    parser.add_argument('--validation-seed', type=int, default=1000)
    parser.add_argument('--test-seed', type=int, default=2000)
    parser.add_argument('--history', type=int, default=8)
    parser.add_argument('--episodes', type=int, default=80)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--validation-trials', type=int, default=5)
    parser.add_argument('--test-trials', type=int, default=20)
    parser.add_argument('--output', type=Path, default=Path('runs/receiver-v1'))
    args = parser.parse_args(argv)
    if min(args.history, args.episodes, args.validation_trials, args.test_trials) < 1 or not args.seeds:
        parser.error('budgets must be positive')
    if args.validation_trials > 1000 or args.test_trials > 1000:
        parser.error('trial ranges must not overlap')
    if set(range(args.validation_seed,args.validation_seed+args.validation_trials)) & set(range(args.test_seed,args.test_seed+args.test_trials)):
        parser.error('validation and test seeds overlap')
    receipt_feedback = args.protocol == 'receipts'
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = dict(history=args.history, version='receiver-v2' if receipt_feedback else 'receiver-v1', protocol=args.protocol,
                profile=args.profile, training_domain=DOMAIN if args.profile == 'randomized' else 'preset scenarios', training_seeds=args.seeds, episodes=args.episodes,
                validation_seeds=list(range(args.validation_seed,args.validation_seed+args.validation_trials)),
                test_seeds=list(range(args.test_seed,args.test_seed+args.test_trials)),
                network_configs={k:asdict(v) for k,v in scenarios().items()},
                sender='frozen AIMD', reward='transport-v2 undiscounted episode return',
                optimizer=dict(algorithm='PPO', learning_rate=1e-4, epochs=2, minibatch=256,
                               episodes_per_update=4, clip=.2, target_kl=.015, entropy=.001,
                               gamma=.999, gae_lambda=.95, gradient_norm=.5),
                software=dict(torch=torch.__version__, numpy=np.__version__),
                selection='maximize completed transfers, then mean score; validation only',
                win_condition='100% exact completion for both; faster with 95% paired CI and no extra feedback, '
                              'or fewer feedback bytes with 95% paired CI and latency CI upper bound <= +5%; '
                              'compare only with validation-selected deterministic baseline',
                limitations=['CPU simulated inference has no simulated compute delay',
                             'receiver batching is simulator-only, not enabled in UDP adapter'])
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    candidates = {f'learned-s{seed}':train(seed,args.episodes,args.output,receipt_feedback=receipt_feedback,profile=args.profile,history=args.history) for seed in args.seeds}
    # Same batching kernel and maximum ACK delay for deterministic controls.
    baselines = {'immediate':(1,0.), 'delayed-2':(2,.01), 'delayed-4':(4,.01),
                 'delayed-8':(8,.01), 'delayed-8-25ms':(8,.025)}
    validation = {}
    for name in [*candidates, *baselines]:
        records = evaluate(load(candidates[name]) if name in candidates else None,
                           baselines.get(name), plan['validation_seeds'], receipt_feedback=receipt_feedback)
        validation[name] = dict(summary=summary(records), records=records)
        print('validation', name, validation[name]['summary'], flush=True)
    rank = lambda name: (validation[name]['summary']['completed'], validation[name]['summary']['mean_score'])
    selected = max(candidates, key=rank)
    selected_baseline = max(baselines, key=rank)
    # Selection is fixed before any test outcome is inspected.
    (args.output/'selection.json').write_text(json.dumps(dict(learned=selected,baseline=selected_baseline,
        validation=validation),indent=2)+'\n')
    final = {}
    for name in [selected, *baselines]:
        records = evaluate(load(candidates[name]) if name in candidates else None,
                           baselines.get(name), plan['test_seeds'], receipt_feedback=receipt_feedback)
        final[name] = dict(summary=summary(records), records=records)
        print('test', name, final[name]['summary'], flush=True)
    best = args.output/f'{plan["version"]}-best.pt'
    best.write_bytes(candidates[selected].read_bytes())
    report = dict(plan=plan, selected=selected, selected_baseline=selected_baseline,
                  validation=validation, final_test=final,
                  comparison=paired_comparison(final[selected]['records'],final[selected_baseline]['records']),
                  checkpoint_sha256=hashlib.sha256(best.read_bytes()).hexdigest())
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
