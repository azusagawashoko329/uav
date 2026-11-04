#!/usr/bin/env python3
"""Score-inclusive TF2 runtime-adapted benchmark for Future Frame / MLEP.

The original TF2-adapted JSONs measured only the generator/prediction tensor.
This script adds the official PSNR against the ground-truth next frame and writes
separate `benchmark_v2_scored.json` files; the original JSONs are not overwritten.

Official scoring:
  Future Frame: psnr_error(pred, gt) = mean(10*log10(1 / mean(((pred+1)/2-(gt+1)/2)^2)))
  MLEP:         psnr_error(pred, gt) = mean(10*log10(1 / mean(((gt-pred)/2)^2)))

Both use a random GT next-frame tensor for timing; the score graph is the
official PSNR path.  The generator/prednet FLOPs remain the TF1 profiler values;
PSNR elementwise ops are not counted in that FLOPs number.

Run:
  CUDA_VISIBLE_DEVICES=GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6 \
  python comparison/benchmarks/bench_tf2_scored.py --method both
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from bench_tf2_compat import (  # noqa: E402
    CLONE, OUT, KNOWN_FLOPS, TF1_PARAMS, _config, _p95, _prepare_runtime_cwd,
)
import tensorflow as tf  # noqa: E402


def _futureframe_score_graph(batch: int):
    sys.path.insert(0, str(CLONE / 'Future Frame/Codes'))
    import unet
    g = tf.Graph()
    with g.as_default():
        x = tf.compat.v1.placeholder(tf.float32, [batch, 256, 256, 12], name='input')
        target = tf.compat.v1.placeholder(tf.float32, [batch, 256, 256, 3], name='target')
        pred = unet.unet(x, layers=4, features_root=64, filter_size=3,
                         pool_size=2, output_channel=3)
        mse = tf.reduce_mean(tf.square(((pred + 1.0) / 2.0) - ((target + 1.0) / 2.0)),
                             axis=[1, 2, 3])
        psnr = 10.0 * tf.math.log(1.0 / mse) / tf.math.log(10.0)
        score = psnr  # per-sample PSNR, shape [B]
        vs = tf.compat.v1.trainable_variables()
        params = sum(int(np.prod(v.shape)) for v in vs)
    return g, {x: (256, 256, 12), target: (256, 256, 3)}, score, params


def _mlep_score_graph(batch: int):
    sys.path.insert(0, str(CLONE / 'MLEP'))
    from models import prediction_networks_dict
    net = prediction_networks_dict['cyclegan_convlstm']
    g = tf.Graph()
    with g.as_default():
        x = tf.compat.v1.placeholder(tf.float32, [batch, 4, 224, 224, 3], name='input')
        target = tf.compat.v1.placeholder(tf.float32, [batch, 224, 224, 3], name='target')
        pred, _, _ = net(inputs=x, use_decoder=True)
        mse = tf.reduce_mean(tf.square((target - pred) / 2.0), axis=[1, 2, 3])
        psnr = 10.0 * tf.math.log(1.0 / mse) / tf.math.log(10.0)
        score = psnr  # per-sample PSNR, shape [B]
        vs = tf.compat.v1.trainable_variables()
        params = sum(int(np.prod(v.shape)) for v in vs)
    return g, {x: (4, 224, 224, 3), target: (224, 224, 3)}, score, params


METHODS = {
    'futureframe': {
        'name': 'Future Frame',
        'input_hwt': '256x256x4',
        'builder': _futureframe_score_graph,
        'commit_path': CLONE / 'Future Frame',
        'out': OUT / 'Future Frame/default/benchmark_v2_scored.json',
        'flops': KNOWN_FLOPS['futureframe'],
        'params': TF1_PARAMS['futureframe'],
        'scoring': 'official PSNR against GT next frame (normalized [-1,1])',
        'training_trainable_m': 14.535684,
        'training_only_m': 6.831489,
    },
    'mlep': {
        'name': 'MLEP',
        'input_hwt': '224x224x4',
        'builder': _mlep_score_graph,
        'commit_path': CLONE / 'MLEP',
        'out': OUT / 'MLEP/default/benchmark_v2_scored.json',
        'flops': KNOWN_FLOPS['mlep'],
        'params': TF1_PARAMS['mlep'],
        'scoring': 'official PSNR against GT next frame ((gt-pred)/2)',
        'training_trainable_m': 12.560835,
        'training_only_m': 0.0,
    },
}


def _run_batch(builder, batch, warmup, iters, rounds):
    g, feed_spec, score, params = builder(batch)
    with g.as_default():
        init = tf.compat.v1.global_variables_initializer()
    rng = np.random.RandomState(0)
    feed = {pl: rng.randn(batch, *shape).astype(np.float32)
            for pl, shape in feed_spec.items()}
    with tf.compat.v1.Session(graph=g, config=_config()) as sess:
        sess.run(init)
        for _ in range(warmup):
            sess.run(score, feed_dict=feed)
        if batch == 1:
            out_rounds = []
            for _ in range(rounds):
                times = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    sess.run(score, feed_dict=feed)
                    times.append((time.perf_counter() - t0) * 1000.0)
                out_rounds.append(times)
            return out_rounds, params
        round_times = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(iters):
                sess.run(score, feed_dict=feed)
            round_times.append(time.perf_counter() - t0)
        return round_times, params


def measure(method: str, warmup: int, iters: int, rounds: int, skip_bs16: bool = False) -> dict:
    spec = METHODS[method]
    print(f'[{method}] score-inclusive BS1', flush=True)
    bs1_rounds, params = _run_batch(spec['builder'], 1, warmup, iters, rounds)
    all_ms = [v for r in bs1_rounds for v in r]
    p50 = statistics.median(all_ms)
    mean = statistics.mean(all_ms)
    p95 = _p95(all_ms)
    if skip_bs16:
        bs16_rounds = []
        params16 = params
        fps_bs16 = None
        print(f'[{method}] BS16 skipped (--skip-bs16)', flush=True)
    else:
        print(f'[{method}] score-inclusive BS16', flush=True)
        bs16_rounds, params16 = _run_batch(spec['builder'], 16, warmup, iters, rounds)
        bs16_median = statistics.median(bs16_rounds)
        fps_bs16 = (16 * iters) / bs16_median
    if params != params16 or params != spec['params']:
        raise RuntimeError(f'param mismatch: {params} vs {params16}, expected {spec["params"]}')
    total_flops = int(spec['flops'])
    return {
        'protocol_version': 2,
        'measurement_status': 'partial/unverified',
        'runtime_adapted': True,
        'model_code_modified': False,
        'method': spec['name'],
        'method_display': spec['name'],
        'variant': 'default (TF2 scored)',
        'supersedes': None,
        'adapted_reason': (
            'Original TF1.15/CUDA10 runtime cannot execute on RTX 5880 sm_89; '
            'unchanged TF1 model graph runs on TF2.20 with contrib shims. '
            'This file adds the official PSNR scoring path.'
        ),
        'code_commit': __import__('subprocess').check_output(
            ['git', '-c', 'safe.directory=*', '-C', str(spec['commit_path']),
             'rev-parse', 'HEAD'], text=True, stderr=__import__('subprocess').DEVNULL).strip(),
        'input_hwt': spec['input_hwt'],
        'input_frames_t': 4,
        'effective_output_scores_k': 1,
        'window_stride': 1,
        'frame_coverage': 1.0,
        'score_density': 1.0,
        'weights_state': 'random init (official checkpoint unavailable)',
        'da_adapted': False,
        'auc_source': 'user table / paper',
        'inference_params_m': params / 1e6,
        'training_trainable_params_m': spec['training_trainable_m'],
        'training_only_params_m': spec['training_only_m'],
        'frozen_inference_params_m': 0.0,
        'benchmark_source': str(spec['out'].relative_to(ROOT)),
        'notes': (
            f'Score-inclusive runtime-adapted benchmark; scoring = {spec["scoring"]}. '
            'Random GT next-frame tensor is fed for timing. The original '
            'benchmark_v2_tf2compat.json remains prediction-only.'
            + (' BS16 was not measured in this run; see prediction-only BS16 for reference.' if skip_bs16 else '')
        ),
        'timing_scope': 'prediction_plus_psnr',
        'scoring_included': True,
        'output_shape_bs1': '[1]',
        'output_shape_bs16': '[16]',
        'output_shape_bs16_note': ('graph output is [B]; BS16 throughput was not measured'
                                   if skip_bs16 else None),
        'bs16_throughput_measured': not skip_bs16,
        'scores_per_sample': 1,
        'batch_output_aggregated': False,
        'scoring_note': spec['scoring'],
        'flops_scope': 'generator/prednet only; PSNR elementwise ops excluded from FLOPs',
        'profiling_config': (
            'NVIDIA RTX 5880 Ada Generation (GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6); '
            f'FP32; TF2.20 runtime-adapted; BS1 {warmup}/{iters}/{rounds}; '
            f'BS16 {warmup}/{iters}/{rounds}; TF Session.run synchronous'
        ),
        'gpu': 'NVIDIA RTX 5880 Ada Generation',
        'gpu_uuid': 'GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6',
        'tensorflow': tf.__version__,
        'cuda': tf.sysconfig.get_build_info().get('cuda_version'),
        'cudnn': tf.sysconfig.get_build_info().get('cudnn_version'),
        'python': platform.python_version(),
        'warmup': warmup,
        'iterations': iters,
        'rounds': rounds,
        'flops_g_per_output_frame': total_flops / 1e9,
        'flops_window_g': total_flops / 1e9,
        'flops_verified': False,
        'flops_counter': (
            'TensorFlow 1.15 profiler float_operation (CPU graph; generator/prednet only); '
            'PSNR elementwise ops are not included'
        ),
        'flops_ops_covered': ['TF1 graph ops (Conv2D, Conv2DBackpropInput, MatMul, etc.)'],
        'flops_ops_missing': [
            'PSNR elementwise ops are outside the TF1 profiler FLOPs number',
            'TF1 profiler is not cross-checked by the v2 PyTorch operator counter',
        ],
        'flops_common_subset_check': 'not applicable: TF graph',
        'flops_duplicate_check': 'not applicable: TF graph',
        'flops_breakdown': [],
        'profiler_total_flops': total_flops,
        'window_latency_ms': mean,
        'latency_bs1_p50_ms_per_output_frame': p50,
        'latency_bs1_mean_ms_per_output_frame': mean,
        'latency_bs1_p95_ms_per_output_frame': p95,
        'fps_bs1': 1000.0 / mean,
        'fps_bs16': fps_bs16,
        'fps_independent_window': 1000.0 / mean,
        'fps_stream_estimated': 1000.0 / mean,
        'fps_stream_estimated_bs16': fps_bs16,
        'fps_stream_measured': None,
        'bs1_rounds_ms': bs1_rounds,
        'bs16_round_times_s': bs16_rounds,
        'tf1_params_check': {
            'tf2_params': params,
            'tf1_params': spec['params'],
            'match': params == spec['params'],
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['futureframe', 'mlep', 'both'], default='both')
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--iters', type=int, default=500)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--skip-bs16', action='store_true')
    p.add_argument('--smoke-batch', type=int, default=1)
    args = p.parse_args()
    _prepare_runtime_cwd()
    methods = ['futureframe', 'mlep'] if args.method == 'both' else [args.method]
    for method in methods:
        if args.smoke:
            # Build and run one scored forward only; do not write results.
            spec = METHODS[method]
            g, feed_spec, score, params = spec['builder'](args.smoke_batch)
            with g.as_default():
                init = tf.compat.v1.global_variables_initializer()
            rng = np.random.RandomState(0)
            feed = {pl: rng.randn(args.smoke_batch, *shape).astype(np.float32)
                    for pl, shape in feed_spec.items()}
            with tf.compat.v1.Session(graph=g, config=_config()) as sess:
                sess.run(init)
                val = sess.run(score, feed_dict=feed)
            print('smoke', method, 'batch', args.smoke_batch, 'params', params,
                  'score', val, 'shape', getattr(val, 'shape', None), flush=True)
            continue
        result = measure(method, args.warmup, args.iters, args.rounds, skip_bs16=args.skip_bs16)
        out = METHODS[method]['out']
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        print('wrote', out, flush=True)
        print(json.dumps({k: v for k, v in result.items() if k in (
            'method', 'measurement_status', 'runtime_adapted', 'scoring_included',
            'inference_params_m', 'flops_g_per_output_frame', 'flops_verified',
            'latency_bs1_p50_ms_per_output_frame', 'fps_bs1', 'fps_bs16',
            'tf1_params_check')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
