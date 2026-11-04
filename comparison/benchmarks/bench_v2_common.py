#!/usr/bin/env python3
"""Shared v2 timing/JSON helpers for comparison benchmarks."""
from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from flops_counter import flops_report

ROOT = Path(__file__).resolve().parents[2]
WARMUP = 50
ITERS = 500
ROUNDS = 3
PROFILING_CONFIG = (
    'RTX 5880 Ada GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6; '
    'FP32; AMP off; TF32 off; eval; inference_mode for timing; '
    'BS1 50/500/3; BS16 50/500/3'
)


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ['git', '-c', 'safe.directory=*', '-C', str(path), 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _p95(values: list[float]) -> float:
    return sorted(values)[max(0, int(0.95 * len(values)) - 1)]


def measure(fn, model, x1, x16, meta: dict, run_profiler: bool = True) -> dict:
    """Run the v2 protocol: FLOPs report + BS1 P50/Mean/P95 + BS16 rounds."""
    k = int(meta['effective_output_scores_k'])
    stride = int(meta['window_stride'])
    t = int(meta['input_frames_t'])
    if k < 1 or stride < 1 or t < 1:
        raise ValueError('T/K/stride must be positive')

    for _ in range(WARMUP):
        with torch.inference_mode():
            fn(x1)
            fn(x16)

    fl = flops_report(fn, model, x1, run_profiler=run_profiler)
    total_flops = int(fl['total_flops'])
    flops_window_g = total_flops / 1e9
    flops_per_output = flops_window_g / k

    bs1_rounds_ms = []
    for _ in range(ROUNDS):
        times = []
        for _ in range(ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                fn(x1)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        bs1_rounds_ms.append(times)
    all_ms = [v for r in bs1_rounds_ms for v in r]
    p50 = statistics.median(all_ms)
    mean = statistics.mean(all_ms)
    p95 = _p95(all_ms)

    bs16_round_times_s = []
    for _ in range(ROUNDS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(ITERS):
                fn(x16)
        torch.cuda.synchronize()
        bs16_round_times_s.append(time.perf_counter() - t0)
    bs16_median = statistics.median(bs16_round_times_s)
    fps_bs16 = (16 * k * ITERS) / bs16_median
    fps_stream_estimated_bs16 = (16 * stride * ITERS) / bs16_median

    result = dict(meta)
    result.update({
        'protocol_version': 2,
        'measurement_status': 'measured' if fl['flops_verified'] else 'partial/unverified',
        'profiling_config': PROFILING_CONFIG,
        'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'python': platform.python_version(),
        'warmup': WARMUP,
        'iterations': ITERS,
        'rounds': ROUNDS,
        'flops_g_per_output_frame': flops_per_output,
        'flops_window_g': flops_window_g,
        'flops_verified': bool(fl['flops_verified']),
        'flops_counter': 'FlopCounterMode(TorchDispatchMode)+native_mha_flop; hand module formulas; profiler common subset',
        'flops_ops_covered': fl['flops_ops_covered'],
        'flops_ops_missing': fl['flops_ops_missing'],
        'flops_module_ops': fl.get('flops_module_ops', []),
        'flops_common_subset_check': fl['flops_common_subset_check'],
        'flops_duplicate_check': fl['flops_duplicate_check'],
        'flops_breakdown': fl['breakdown'],
        'per_op_flops': fl['per_op'],
        'hand_check': fl['hand_check'],
        'profiler_check': fl['profiler_check'],
        'profiler_total_flops': fl['profiler_total_flops'],
        'window_latency_ms': mean,
        'latency_bs1_p50_ms_per_output_frame': p50 / k,
        'latency_bs1_mean_ms_per_output_frame': mean / k,
        'latency_bs1_p95_ms_per_output_frame': p95 / k,
        'fps_bs1': k * 1000.0 / mean,
        'fps_bs16': fps_bs16,
        'fps_independent_window': k * 1000.0 / mean,
        'score_density': k / stride,
        'fps_stream_estimated': stride * 1000.0 / mean,
        'fps_stream_estimated_bs16': fps_stream_estimated_bs16,
        'fps_stream_measured': None,
        'bs1_rounds_ms': bs1_rounds_ms,
        'bs16_round_times_s': bs16_round_times_s,
    })
    result['score_density'] = k / stride
    result['benchmark_source'] = meta.get('benchmark_source')
    return result


def write_result(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print('wrote', path, flush=True)
    print(json.dumps({k: v for k, v in data.items()
                      if k in ('method', 'variant', 'measurement_status',
                               'inference_params_m', 'training_trainable_params_m',
                               'flops_g_per_output_frame', 'flops_verified',
                               'latency_bs1_p50_ms_per_output_frame', 'fps_bs1',
                               'fps_bs16', 'score_density', 'frame_coverage')},
                     ensure_ascii=False), flush=True)
