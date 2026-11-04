#!/usr/bin/env python3
"""One-forward BS1/BS16 output-shape check (not a benchmark rerun).

Run one method per process (`--only <name>`) so that the cloned repos' top-level
`models` packages do not shadow each other.  It never writes benchmark JSONs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))


def _shape(v):
    if isinstance(v, torch.Tensor):
        return f'Tensor{tuple(v.shape)}'
    if isinstance(v, (tuple, list)):
        return '[' + ', '.join(_shape(x) for x in v) + ']'
    return type(v).__name__


def _check(name, fn, x1, x16, model):
    with torch.inference_mode():
        y1 = fn(x1)
        y16 = fn(x16)
    print(f'{name:24s} BS1={_shape(y1):32s} BS16={_shape(y16)}', flush=True)
    del model, y1, y16
    torch.cuda.empty_cache()


def run_one(name):
    import torch.nn.functional as F

    if name in {'ganomaly', 'memae', 'mkd'}:
        import bench_v2_external as ext
        shapes = {
            'ganomaly': ((3, 256, 256), 'GANomaly'),
            'memae': ((3, 16, 256, 256), 'MemAE'),
            'mkd': ((3, 128, 128), 'MKD'),
        }
        shape, out = shapes[name]
        builder = {'ganomaly': ext._ganomaly, 'memae': ext._memae, 'mkd': ext._mkd}[name]
        model, fn, _ = builder(ROOT / f'comparison/da/{out}/default/benchmark_v2.json')
        x1 = torch.randn(1, *shape, device='cuda')
        x16 = torch.randn(16, *shape, device='cuda')
        _check(name, fn, x1, x16, model)
        return

    if name in {'lightvit', 'madvit', 'astt', 'hstforu'}:
        import bench_v2_extra as extra
        builder, out = extra.BUILDERS[name]
        model, fn, (x1, x16), _ = builder(out)
        _check(name, fn, x1, x16, model)
        return

    if name in {'st_scored', 'dad_scored'}:
        from bench_v2_scored import _dad_scored, _st_scored
        builder, out = {
            'st_scored': (_st_scored, ROOT / 'comparison/da/ST Dissoc./default/benchmark_v2_scored.json'),
            'dad_scored': (_dad_scored, ROOT / 'comparison/da/DAD-FSM/default/benchmark_v2_scored.json'),
        }[name]
        model, fn, (x1, x16), _ = builder(out)
        _check(name, fn, x1, x16, model)
        return

    if name == 'andt_adapted':
        sys.path.insert(0, str(ROOT / 'comparison/da/ANDT/default/scripts'))
        from andt_adapted_model import build_adapted_andt
        model = build_adapted_andt(
            num_frames=6, image_size=256, patch_size=16, emb_dim=768,
            mlp_dim=4096, num_heads=6, num_layers=2, num_classes=25,
            dropout_rate=0.1,
        ).cuda().eval()

        def fn(x):
            out = model(x)
            return F.mse_loss(out, torch.zeros_like(out))

        x1 = torch.randn(1, 6, 3, 256, 256, device='cuda')
        x16 = torch.randn(16, 6, 3, 256, 256, device='cuda')
        _check(name, fn, x1, x16, model)
        return

    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', required=True,
                   choices=['ganomaly', 'memae', 'mkd', 'lightvit', 'madvit',
                            'astt', 'hstforu', 'st_scored', 'dad_scored',
                            'andt_adapted'])
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    run_one(args.only)


if __name__ == '__main__':
    main()
