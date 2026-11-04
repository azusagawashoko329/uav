#!/usr/bin/env python3
"""LEGACY v1 benchmark for external methods (GANomaly, MemAE, MKD).

WARNING: this script writes the frozen v1 files comparison/da/benchmark_*.json
and uses profiler-only FLOPs, assumed K=T for MemAE, and final-layer distance
for MKD. It is NOT valid for formal v2 numbers. See
comparison/benchmarks/README.md and comparison/BENCHMARK_PROTOCOL.md; v2
results must go to benchmark_v2_<method>.json.

adVAE and ANDT are intentionally not timed here: adVAE's official repo is the
tabular version, and ANDT's released forward is a code variant with pdb/decoder
issues (see BENCHMARK_PROTOCOL.md).

Run one method per process on the RTX 5880 Ada:
  CUDA_VISIBLE_DEVICES=GPU-3ae8a423-... python comparison/benchmarks/bench_external.py --method ganomaly
  ... --method memae
  ... --method mkd
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[2]
CLONE = Path('/data1/data8tb/liangyuxuan/Comparison method')
OUT = ROOT / 'comparison/da'
WARMUP = 50
ITERS = 500
ROUNDS = 3


def profile_flops(full, x):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_flops=True) as prof:
        with torch.inference_mode():
            full(x)
    return sum(getattr(e, 'flops', 0) or 0 for e in prof.events()) / 1e9


def bench(full, x1, x16, out_per_sample):
    for _ in range(WARMUP):
        with torch.inference_mode():
            full(x1)
            full(x16)
    flops = profile_flops(full, x1) / out_per_sample
    lat = []
    for _ in range(ROUNDS):
        for _ in range(ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                full(x1)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)
    lat_p50 = statistics.median(lat)
    lat_mean = statistics.mean(lat)
    p95 = sorted(lat)[int(0.95 * len(lat)) - 1]
    round_times = []
    for _ in range(ROUNDS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(ITERS):
                full(x16)
        torch.cuda.synchronize()
        round_times.append(time.perf_counter() - t0)
    fps_bs16 = (16 * out_per_sample * ITERS) / statistics.median(round_times)
    return {
        'flops_g_per_output_frame': flops,
        'latency_bs1_p50_ms_per_output_frame': lat_p50 / out_per_sample,
        'latency_bs1_mean_ms_per_output_frame': lat_mean / out_per_sample,
        'latency_bs1_p95_ms_per_output_frame': p95 / out_per_sample,
        'fps_bs1': out_per_sample * 1000.0 / lat_mean,
        'fps_bs16': fps_bs16,
        'bs16_round_times_s': round_times,
    }


def ganomaly():
    sys.path.insert(0, str(CLONE / 'GANomaly'))
    from lib.networks import NetD, NetG
    opt = SimpleNamespace(isize=256, nz=100, nc=3, ngf=64, ndf=64, ngpu=1, extralayers=0)
    g = NetG(opt).cuda().eval()
    d = NetD(opt).cuda().eval()
    pg = sum(p.numel() for p in g.parameters())
    pd = sum(p.numel() for p in d.parameters())

    def full(x):
        _, li, lo = g(x)
        return ((li - lo) ** 2).mean(dim=1)

    r = bench(full, torch.randn(1, 3, 256, 256, device='cuda'),
              torch.randn(16, 3, 256, 256, device='cuda'), 1)
    return {
        'method': 'GANomaly', 'input': '256x256x1 (RGB)',
        'total_params_m': pg / 1e6,
        'trainable_params_m': (pg + pd) / 1e6,
        'training_only_params_m': pd / 1e6,
        'notes': 'inference graph G=143.950M; D is training-only 44.739M; trainable column includes G+D.',
        **r,
    }


def memae():
    sys.path.insert(0, str(CLONE / 'MemAE'))
    from models.memae_3dconv import AutoEncoderCov3DMem
    m = AutoEncoderCov3DMem(chnum_in=3, mem_dim=2000, shrink_thres=0.0025).cuda().eval()
    p = sum(p.numel() for p in m.parameters())

    def full(x):
        r = m(x)['output'] - x
        sp = (r ** 2).sum(dim=1).sqrt()
        return sp.mean(dim=(2, 3))  # B x T frame scores

    r = bench(full, torch.randn(1, 3, 16, 256, 256, device='cuda'),
              torch.randn(16, 3, 16, 256, 256, device='cuda'), 16)
    return {
        'method': 'MemAE', 'input': '256x256x16 (RGB clip, T=16)',
        'total_params_m': p / 1e6, 'trainable_params_m': p / 1e6,
        'notes': 'MemDim=2000, C=3; 16 output frames per clip; metrics divided by T=16.',
        **r,
    }


def mkd():
    sys.path.insert(0, str(CLONE / 'MKD'))
    import torchvision
    from models.network import make_arch
    cfg = {
        'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'B': [16, 16, 'M', 16, 128, 'M', 16, 16, 256, 'M', 16, 16, 512, 'M', 16, 16, 512, 'M'],
    }
    source = torchvision.models.vgg16(weights=None).features.cuda().eval()
    cloner = make_arch('B', cfg, use_bias=False, batch_norm=True).cuda().eval()
    ps = sum(p.numel() for p in source.parameters())
    pc = sum(p.numel() for p in cloner.parameters())

    def full(x):
        a = source(x)
        b = cloner.features(x)
        if a.shape[-2:] != b.shape[-2:]:
            b = torch.nn.functional.interpolate(b, size=a.shape[-2:], mode='bilinear', align_corners=False)
        return ((a - b) ** 2).mean(dim=(1, 2, 3))

    r = bench(full, torch.randn(1, 3, 128, 128, device='cuda'),
              torch.randn(16, 3, 128, 128, device='cuda'), 1)
    return {
        'method': 'MKD', 'input': '128x128x1 (RGB; MVTec config)',
        'total_params_m': (ps + pc) / 1e6, 'trainable_params_m': pc / 1e6,
        'notes': 'source VGG16 features frozen; cloner cfg B trainable; 128x128 per MKD MVTec config; DA may adapt to 256.',
        **r,
    }


def main():
    print('WARNING: v1 LEGACY benchmark; not valid for formal v2 numbers. '
          'See comparison/benchmarks/README.md.', flush=True)
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=['ganomaly', 'memae', 'mkd'])
    p.add_argument('--overwrite-v1', action='store_true',
                   help='allow overwriting the frozen v1 benchmark_<method>.json (not for formal v2)')
    a = p.parse_args()
    out = OUT / f'benchmark_{a.method}.json'
    if out.exists() and not a.overwrite_v1:
        raise SystemExit(
            f'refusing to overwrite frozen v1 file {out}; '
            'see comparison/benchmarks/README.md (pass --overwrite-v1 only if intentional)')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    fn = {'ganomaly': ganomaly, 'memae': memae, 'mkd': mkd}[a.method]
    r = fn()
    r.update({'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'precision': 'fp32; amp off; tf32 off',
              'warmup': WARMUP, 'iterations': ITERS, 'rounds': ROUNDS})
    out.write_text(json.dumps(r, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(r, ensure_ascii=False), flush=True)
    print('wrote', out)
    gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
