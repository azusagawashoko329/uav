#!/usr/bin/env python3
"""TF1 benchmarks for Future Frame and MLEP (uav_tf1 conda env).

These two official repos are TensorFlow 1.x. We build their inference graphs
with TF 1.15, measure params/latency on the RTX 5880, and report FLOPs from the
TF1 profiler. Because the v2 PyTorch operator-counter cross-check does not
apply to TF graphs, FLOPs are marked partial/unverified.

Run with:
  CUDA_VISIBLE_DEVICES=1 conda run -n uav_tf1 python comparison/benchmarks/bench_tf1_extra.py --method futureframe
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
CLONE = Path('/data1/data8tb/liangyuxuan/Comparison method')
OUT = ROOT / 'comparison/da'
WARMUP = 50
ITERS = 500
ROUNDS = 3
GPU_NAME = 'NVIDIA RTX 5880 Ada Generation'
# TF1 profiler results from the same graphs on CPU (profiler analysis is very
# slow on the full GPU graph; the graph/FLOPs are device-independent).
KNOWN_FLOPS = {
    'futureframe': 74476028824,
    'mlep': 319512526355,
}


def git_commit(path: Path):
    import subprocess
    try:
        return subprocess.check_output(
            ['git', '-c', 'safe.directory=*', '-C', str(path), 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _futureframe_graph(batch: int):
    import sys
    sys.path.insert(0, str(CLONE / 'Future Frame/Codes'))
    import unet
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(tf.float32, [batch, 256, 256, 12], name='input')
        y = unet.unet(x, layers=4, features_root=64, filter_size=3,
                      pool_size=2, output_channel=3)
        params = sum(int(np.prod(v.shape)) for v in tf.trainable_variables())
    return g, x, y, params


def _mlep_graph(batch: int):
    import sys
    sys.path.insert(0, str(CLONE / 'MLEP'))
    from models import prediction_networks_dict
    net = prediction_networks_dict['cyclegan_convlstm']
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(tf.float32, [batch, 4, 224, 224, 3], name='input')
        y, _, _ = net(inputs=x, use_decoder=True)
        params = sum(int(np.prod(v.shape)) for v in tf.trainable_variables())
    return g, x, y, params


METHODS = {
    'futureframe': {
        'name': 'Future Frame',
        'input_hwt': '256x256x4',
        'shape': (256, 256, 12),
        'builder': _futureframe_graph,
        'commit_path': CLONE / 'Future Frame',
        'out': OUT / 'Future Frame/default',
        'notes': 'Official TF1 generator (U-Net), 4 input frames -> next frame; inference does not use FlowNet. '
                 'No official checkpoint; random init. FLOPs from TF1 profiler only (partial/unverified).',
    },
    'mlep': {
        'name': 'MLEP',
        'input_hwt': '224x224x4',
        'shape': (4, 224, 224, 3),
        'builder': _mlep_graph,
        'commit_path': CLONE / 'MLEP',
        'out': OUT / 'MLEP/default',
        'notes': 'Official TF1 cyclegan_convlstm prediction net, 4 input frames -> next frame; '
                 'no official checkpoint; random init. FLOPs from TF1 profiler only (partial/unverified).',
    },
}


def _p95(values):
    return sorted(values)[max(0, int(0.95 * len(values)) - 1)]


def _config():
    c = tf.ConfigProto()
    c.gpu_options.allow_growth = True
    return c


def measure(method: str, run_profiler: bool = True):
    spec = METHODS[method]
    shape = spec['shape']

    # One graph/session per batch size because MLEP's model embeds batch size.
    g1, x1, y1, params = spec['builder'](1)
    with g1.as_default():
        init1 = tf.global_variables_initializer()
    with tf.Session(graph=g1, config=_config()) as sess:
        sess.run(init1)
        feed1 = {x1: np.random.randn(1, *shape).astype(np.float32)}
        for _ in range(WARMUP):
            sess.run(y1, feed_dict=feed1)

        if run_profiler:
            rm = tf.RunMetadata()
            sess.run(y1, feed_dict=feed1,
                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                     run_metadata=rm)
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            prof = tf.profiler.profile(graph=g1, run_meta=rm, cmd='op', options=opts)
            total_flops = int(prof.total_float_ops)
        else:
            total_flops = KNOWN_FLOPS[method]

        rounds = []
        for _ in range(ROUNDS):
            ts = []
            for _ in range(ITERS):
                t0 = time.perf_counter()
                sess.run(y1, feed_dict=feed1)
                ts.append((time.perf_counter() - t0) * 1000.0)
            rounds.append(ts)
    all_ms = [v for r in rounds for v in r]
    p50, mean, p95 = statistics.median(all_ms), statistics.mean(all_ms), _p95(all_ms)

    g16, x16, y16, _ = spec['builder'](16)
    with g16.as_default():
        init16 = tf.global_variables_initializer()
    with tf.Session(graph=g16, config=_config()) as sess:
        sess.run(init16)
        feed16 = {x16: np.random.randn(16, *shape).astype(np.float32)}
        for _ in range(WARMUP):
            sess.run(y16, feed_dict=feed16)
        round_times = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            for _ in range(ITERS):
                sess.run(y16, feed_dict=feed16)
            round_times.append(time.perf_counter() - t0)
    bs16 = statistics.median(round_times)
    fps_bs16 = 16 * ITERS / bs16

    out_path = spec['out'] / 'benchmark_v2.json'
    result = {
        'protocol_version': 2,
        'measurement_status': 'partial/unverified',
        'method': spec['name'],
        'method_display': spec['name'],
        'variant': 'default',
        'supersedes': None,
        'code_commit': git_commit(spec['commit_path']),
        'input_hwt': spec['input_hwt'],
        'input_frames_t': 4,
        'effective_output_scores_k': 1,
        'window_stride': 1,
        'frame_coverage': 1.0,
        'weights_state': 'random init (official checkpoint unavailable)',
        'da_adapted': False,
        'auc_source': 'user table / paper (DA-adapted baseline)',
        'inference_params_m': params / 1e6,
        'training_trainable_params_m': params / 1e6,
        'training_only_params_m': 0.0,
        'frozen_inference_params_m': 0.0,
        'flops_g_per_output_frame': total_flops / 1e9,
        'flops_window_g': total_flops / 1e9,
        'flops_verified': False,
        'flops_counter': 'TensorFlow 1.15 profiler float_operation (CPU graph; same graph as GPU run)',
        'flops_ops_covered': ['TF1 graph ops (Conv2D, Conv2DBackpropInput, MatMul, etc.)'],
        'flops_ops_missing': ['TF1 profiler is not cross-checked by the v2 operator counter'],
        'flops_common_subset_check': 'not applicable: TF1 graph',
        'flops_duplicate_check': 'not applicable: TF1 graph',
        'latency_bs1_p50_ms_per_output_frame': p50,
        'latency_bs1_mean_ms_per_output_frame': mean,
        'latency_bs1_p95_ms_per_output_frame': p95,
        'fps_bs1': 1000.0 / mean,
        'fps_bs16': fps_bs16,
        'fps_independent_window': 1000.0 / mean,
        'score_density': 1.0,
        'fps_stream_estimated': 1000.0 / mean,
        'fps_stream_measured': None,
        'window_latency_ms': mean,
        'bs1_rounds_ms': rounds,
        'bs16_round_times_s': round_times,
        'profiling_config': f'{GPU_NAME} (TF1 CUDA_VISIBLE_DEVICES=1); FP32; '
                            'BS1 50/500/3; BS16 50/500/3; TF1.15',
        'gpu': GPU_NAME,
        'torch': 'n/a (TensorFlow 1.15)',
        'cuda': '10.0 (TF1 env)',
        'cudnn': '7.6.5 (TF1 env)',
        'python': platform.python_version(),
        'warmup': WARMUP,
        'iterations': ITERS,
        'rounds': ROUNDS,
        'benchmark_source': str(out_path.relative_to(ROOT)),
        'notes': spec['notes'],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: result[k] for k in ('method', 'inference_params_m',
                                             'flops_g_per_output_frame', 'flops_verified',
                                             'latency_bs1_p50_ms_per_output_frame',
                                             'fps_bs1', 'fps_bs16')}, ensure_ascii=False))
    print('wrote', out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=sorted(METHODS))
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--no-profiler', action='store_true')
    a = p.parse_args()
    if a.smoke:
        global WARMUP, ITERS, ROUNDS
        WARMUP, ITERS, ROUNDS = 1, 1, 1
    measure(a.method, run_profiler=not a.no_profiler)


if __name__ == '__main__':
    main()
