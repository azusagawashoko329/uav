#!/usr/bin/env python3
"""Score-inclusive v2 benchmarks for ST Dissoc. and DAD-FSM.

The original v2 JSONs measured the network forward only (ST) or used an
MSE-to-zero proxy (DAD).  This script adds the official scoring path and writes
separate `benchmark_v2_scored.json` files; the original benchmark JSONs are not
overwritten.

ST Dissoc.:
  static_dec, pred_dec, cluster_map = model(x, None, ['E'])
  pred_recon = pred_dec + static_dec
  pred_target = static_in + motion_target
  score = reciprocal(mean(mean((pred_recon-pred_target)^2, dim=1) * cluster_map))

DAD-FSM:
  pred = model(rgb, seg)
  score = MSE(pred, future_segmentation_target)   # official target, not zeros

Run:
  CUDA_VISIBLE_DEVICES=GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6 \
  python comparison/benchmarks/bench_v2_scored.py --method both
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from bench_v2_common import measure, write_result
from bench_v2_extra import _dadfsm, _st_dissoc


def _repath(meta: dict, out_path: Path, variant: str, notes: str) -> dict:
    meta = dict(meta)
    meta['variant'] = variant
    meta['benchmark_source'] = str(out_path.relative_to(ROOT))
    meta['notes'] = notes
    meta['scoring_included'] = True
    return meta


def _st_scored(out_path: Path):
    base, _, _, old_meta = _st_dissoc(out_path)
    raw = base.m
    from arch.module.Basic import video_static_motion

    class STScored(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            static_in, motion_in, _, motion_target = video_static_motion(x, 3, 5)
            static_dec, pred_dec, cluster_map = self.m(x, None, ['E'])
            pred_recon = pred_dec + static_dec
            pred_target = static_in + motion_target
            recon = ((pred_recon - pred_target) ** 2).mean(dim=1, keepdim=True)
            loss = (recon * cluster_map).mean(dim=(1, 2, 3))
            return 1.0 / loss

    model = STScored(raw).cuda().eval()
    p = sum(v.numel() for v in model.parameters())
    meta = _repath(
        old_meta, out_path, 'default (scored)', (
            'Score-inclusive variant: official stage-E forward plus '
            '`loss_map(mean((pred_recon-pred_target)^2, dim=1), cluster_map)` and '
            'reciprocal. The old benchmark_v2.json measured stage-E forward only. '
            'Input/params match the original ST row; scoring is included here.'
        ))
    meta['inference_params_m'] = p / 1e6
    meta['training_trainable_params_m'] = p / 1e6
    meta['adapted'] = True
    meta['output_shape_bs1'] = '[1]'
    meta['output_shape_bs16'] = '[16]'
    meta['scores_per_sample'] = 1
    meta['batch_output_aggregated'] = False
    return model, lambda x: model(x), (
        torch.randn(1, 15, 256, 256, device='cuda'),
        torch.randn(16, 15, 256, 256, device='cuda'),
    ), meta


def _dad_scored(out_path: Path):
    base, _, _, old_meta = _dadfsm(out_path)
    raw = base.m

    class DADScored(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, inp):
            x, seg, target = inp
            pred = self.m(x, seg)
            return ((pred - target) ** 2).mean(dim=(1, 2, 3))  # [B]

    model = DADScored(raw).cuda().eval()
    p = sum(v.numel() for v in model.parameters())
    meta = _repath(
        old_meta, out_path, 'default (scored)', (
            'Score-inclusive variant: detector forward plus official MSE against '
            'the future segmentation target (64x64), not MSE-to-zero. Segmentation '
            'maps are still precomputed inputs; segmentation generation is excluded. '
            'The old benchmark_v2.json used a zero-target proxy.'
        ))
    meta['inference_params_m'] = p / 1e6
    meta['training_trainable_params_m'] = p / 1e6
    meta['adapted'] = True
    meta['output_shape_bs1'] = '[1]'
    meta['output_shape_bs16'] = '[16]'
    meta['scores_per_sample'] = 1
    meta['batch_output_aggregated'] = False
    return model, lambda inp: model(inp), (
        (torch.randn(1, 4, 3, 224, 224, device='cuda'),
         torch.randn(1, 4, 3, 224, 224, device='cuda'),
         torch.randn(1, 3, 64, 64, device='cuda')),
        (torch.randn(16, 4, 3, 224, 224, device='cuda'),
         torch.randn(16, 4, 3, 224, 224, device='cuda'),
         torch.randn(16, 3, 64, 64, device='cuda')),
    ), meta


BUILDERS = {
    'st': (_st_scored, ROOT / 'comparison/da/ST Dissoc./default/benchmark_v2_scored.json'),
    'dad': (_dad_scored, ROOT / 'comparison/da/DAD-FSM/default/benchmark_v2_scored.json'),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['st', 'dad', 'both'], default='both')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--no-profiler', action='store_true')
    args = p.parse_args()
    methods = ['st', 'dad'] if args.method == 'both' else [args.method]
    for name in methods:
        builder, out_path = BUILDERS[name]
        print(f'[{name}] building score-inclusive graph', flush=True)
        model, fn, xs, meta = builder(out_path)
        x1, x16 = xs
        if args.smoke:
            with torch.inference_mode():
                out1 = fn(x1)
                out16 = fn(x16)
            print('smoke output:', meta['method'], meta['variant'],
                  type(out1).__name__, getattr(out1, 'shape', None),
                  type(out16).__name__, getattr(out16, 'shape', None), flush=True)
            del model
            torch.cuda.empty_cache()
            continue
        result = measure(fn, model, x1, x16, meta,
                         run_profiler=not args.no_profiler)
        write_result(out_path, result)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
