#!/usr/bin/env python3
"""A/B BS16 check for the ST Dissoc. / DAD-FSM throughput anomaly.

Runs the old (forward-only / zero-target proxy) and new (score-inclusive)
variants in an alternating order on the same GPU, same batch-16 inputs,
50 warmup + 3x500 batches, and records per-round times plus GPU clocks.

This script only writes `benchmark_v2_ab_bs16.json`; it never touches the
existing benchmark JSONs.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
import sys
sys.path.insert(0, str(HERE))

from bench_v2_extra import _st_dissoc, _dadfsm
from bench_v2_scored import _st_scored, _dad_scored


def _gpu_state():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu',
             '--format=csv,noheader'], text=True).strip().splitlines()
        # Prefer GPU 1 (5880); keep all lines defensively.
        return out
    except Exception as exc:
        return [f'unavailable: {exc}']


def _build(name):
    if name == 'st_old':
        model, fn, xs, meta = _st_dissoc(ROOT / 'comparison/da/ST Dissoc./default/benchmark_v2.json')
    elif name == 'st_scored':
        model, fn, xs, meta = _st_scored(ROOT / 'comparison/da/ST Dissoc./default/benchmark_v2_scored.json')
    elif name == 'dad_old':
        model, fn, xs, meta = _dadfsm(ROOT / 'comparison/da/DAD-FSM/default/benchmark_v2.json')
    elif name == 'dad_scored':
        model, fn, xs, meta = _dad_scored(ROOT / 'comparison/da/DAD-FSM/default/benchmark_v2_scored.json')
    else:
        raise ValueError(name)
    x1, x16 = xs
    return model.cuda().eval(), fn, x16


def _time_bs16(fn, x16, warmup, iters):
    with torch.inference_mode():
        for _ in range(warmup):
            fn(x16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(x16)
        torch.cuda.synchronize()
        return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['st', 'dad', 'both'], default='both')
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--iters', type=int, default=500)
    p.add_argument('--rounds', type=int, default=3)
    args = p.parse_args()
    methods = ['st', 'dad'] if args.method == 'both' else [args.method]
    out = {}
    for method in methods:
        old_name = f'{method}_old'
        new_name = f'{method}_scored'
        print(f'[{method}] building old/scored models', flush=True)
        old_model, old_fn, x16 = _build(old_name)
        new_model, new_fn, new_x16 = _build(new_name)
        # Build once per variant; alternate old -> scored for each round.
        order = []
        for r in range(args.rounds):
            order += [old_name, new_name]
        times = {old_name: [], new_name: []}
        gpu = {'start': _gpu_state()}
        for name in order:
            fn = old_fn if name == old_name else new_fn
            x = x16 if name == old_name else new_x16
            print(f'[{method}] {name} BS16 round', flush=True)
            t = _time_bs16(fn, x, args.warmup, args.iters)
            times[name].append(t)
            gpu.setdefault(name, []).append(_gpu_state())
        gpu['end'] = _gpu_state()
        old_med = statistics.median(times[old_name])
        new_med = statistics.median(times[new_name])
        out[method] = {
            'protocol': f'BS16 {args.warmup} warmup + {args.rounds}x{args.iters} batches; '
                        'old/scored alternate each round; same GPU/inputs',
            'order': order,
            'old_round_times_s': times[old_name],
            'scored_round_times_s': times[new_name],
            'old_median_s': old_med,
            'scored_median_s': new_med,
            'old_fps': 16 * args.iters / old_med,
            'scored_fps': 16 * args.iters / new_med,
            'gpu_state': gpu,
            'note': 'BS16 FPS = 16 * iters / median(round_time). Old ST returns a tuple; '
                    'old DAD returns a batch scalar. Scored variants return [B] per-sample scores.',
        }
        del old_model, new_model
        torch.cuda.empty_cache()
    path = ROOT / 'comparison/da/fairness_bs16_ab.json'
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print('wrote', path, flush=True)
    print(json.dumps({k: {'old_fps': v['old_fps'], 'scored_fps': v['scored_fps']}
                      for k, v in out.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
