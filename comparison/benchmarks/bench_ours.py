#!/usr/bin/env python3
"""LEGACY v1 benchmark for the six Ours DA variants.

WARNING: this script writes the frozen v1 file comparison/da/benchmark_ours.json
and uses profiler-only FLOPs plus the old total/trainable parameter names. It is
NOT valid for formal v2 numbers. See comparison/benchmarks/README.md and
comparison/BENCHMARK_PROTOCOL.md; v2 results must go to benchmark_v2_ours.json.

Primary scope: preprocessed input tensor on GPU -> frame-level anomaly score.
Includes the frozen DINOv3 teacher, OCBE and student. FP32, no AMP/TF32.
Run on the RTX 5880 Ada (the caller sets CUDA_VISIBLE_DEVICES).
"""
from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import anomaly_readouts, build_appearance_teacher
from scripts.da_variant_runner import VARIANT_SPECS
from scripts.uit_appearance_runner import build_variant
from torch.profiler import ProfilerActivity, profile

WARMUP = 50
ITERS = 500
ROUNDS = 3
INPUT = '256x256x1 (RGB)'


def measure_variant(teacher, vid, spec):
    ocbe, student = build_variant({'variant': spec['builder'], 'seed': 42}, 42)
    ocbe = ocbe.cuda().eval()
    student = student.cuda().eval()
    taps_all = spec['taps']

    teacher_params = sum(p.numel() for p in teacher.parameters())
    trainable = sum(p.numel() for p in list(ocbe.parameters()) + list(student.parameters()))
    total = teacher_params + trainable

    def full(x):
        all_taps = teacher(x)['patches']
        taps = [all_taps[i] for i in taps_all]
        rec = student(ocbe(taps))
        return anomaly_readouts(taps, rec, k=5)['topk']

    x1 = torch.randn(1, 3, 256, 256, device='cuda')
    x16 = torch.randn(16, 3, 256, 256, device='cuda')
    for _ in range(WARMUP):
        with torch.inference_mode():
            full(x1)
            full(x16)

    # FLOPs: one BS1 forward, profiler CPU-side flops (same convention as protocol)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_flops=True) as prof:
        with torch.inference_mode():
            full(x1)
    flops = sum(getattr(e, 'flops', 0) or 0 for e in prof.events()) / 1e9

    # Latency@BS1: sync every iteration, P50 over 3x500
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
    lat_p95 = sorted(lat)[int(0.95 * len(lat)) - 1]

    # FPS@BS16: continuous batches, sync only at round end
    round_times = []
    for _ in range(ROUNDS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(ITERS):
                full(x16)
        torch.cuda.synchronize()
        round_times.append(time.perf_counter() - t0)
    fps_bs16 = (16 * ITERS) / statistics.median(round_times)

    result = {
        'id': f'da-ours-{vid}',
        'variant': vid,
        'label': spec['label'],
        'input': INPUT,
        'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'precision': 'fp32; amp off; tf32 off',
        'warmup': WARMUP, 'iterations': ITERS, 'rounds': ROUNDS,
        'total_params_m': total / 1e6,
        'trainable_params_m': trainable / 1e6,
        'teacher_params_m': teacher_params / 1e6,
        'flops_g': flops,
        'latency_bs1_p50_ms': lat_p50,
        'latency_bs1_mean_ms': lat_mean,
        'latency_bs1_p95_ms': lat_p95,
        'fps_bs1': 1000.0 / lat_mean,
        'fps_bs16': fps_bs16,
        'bs16_round_times_s': round_times,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    del ocbe, student, x1, x16
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    print('WARNING: v1 LEGACY benchmark; not valid for formal v2 numbers. '
          'See comparison/benchmarks/README.md.', flush=True)
    path = ROOT / 'comparison/da/benchmark_ours.json'
    if path.exists() and '--overwrite-v1' not in sys.argv:
        raise SystemExit(
            f'refusing to overwrite frozen v1 file {path}; '
            'see comparison/benchmarks/README.md (pass --overwrite-v1 only if intentional)')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    teacher = build_appearance_teacher(layers=list(range(12))).cuda().eval()
    out = {
        'protocol': 'comparison/BENCHMARK_PROTOCOL.md',
        'scope': 'preprocessed GPU tensor -> frame-level anomaly score; includes DINOv3 teacher',
        'variants': {},
    }
    for vid, spec in VARIANT_SPECS.items():
        print(f'=== {spec["label"]} {vid} ===', flush=True)
        out['variants'][vid] = measure_variant(teacher, vid, spec)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print('wrote', path)
    print('| ID | total M | trainable M | FLOPs G | Latency P50 ms | FPS BS1 | FPS BS16 |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for vid, r in out['variants'].items():
        print(f"| {r['id']} | {r['total_params_m']:.3f} | {r['trainable_params_m']:.3f} | "
              f"{r['flops_g']:.3f} | {r['latency_bs1_p50_ms']:.3f} | {r['fps_bs1']:.1f} | {r['fps_bs16']:.1f} |")


if __name__ == '__main__':
    main()
