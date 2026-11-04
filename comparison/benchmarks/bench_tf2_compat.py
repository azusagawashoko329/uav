#!/usr/bin/env python3
"""Runtime-adapted v2 benchmark for Future Frame / MLEP (TF1 model code on TF2).

The official repos are TF1 graph-mode code.  Their TF1.15 wheel is pinned to
CUDA 10.0/cuDNN 7.6.5 and cannot execute cuBLAS/cuDNN on the RTX 5880 Ada
(sm_89).  This script runs the *unchanged* model code on TF2.20 (CUDA 12.5,
cuDNN 9) with the tiny contrib API shim in tf2_compat_shims.py, so we can get a
latency on the same GPU under the v2 timing protocol.

This is NOT the original TF1.15 runtime.  Outputs are written to
benchmark_v2_tf2compat.json and labelled runtime-adapted; the official TF1.15
record stays partial/unverified because its GPU latency is unavailable.
FLOPs/params come from the TF1.15 profiler on the identical graph; the TF2
shim's exact parameter count is checked against the TF1 value.

Run:
  cd /home/lyx/miniconda3/envs/uav_tf1/lib   # provides libdevice.10.bc in cwd
  CUDA_VISIBLE_DEVICES=GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6 \
  python comparison/benchmarks/bench_tf2_compat.py --method both

(If libdevice.10.bc is not in cwd, the script copies it to a temp cwd.)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from tf2_compat_shims import install  # noqa: E402

install(tf)

CLONE = Path('/data1/data8tb/liangyuxuan/Comparison method')
OUT = ROOT / 'comparison/da'
WARMUP = 50
ITERS = 500
ROUNDS = 3
GPU_NAME = 'NVIDIA RTX 5880 Ada Generation'
GPU_UUID = 'GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6'

# TF1.15 profiler float_operation on the identical graphs (CPU graph).
KNOWN_FLOPS = {
    'futureframe': 74476028824,
    'mlep': 319512526355,
}
TF1_PARAMS = {
    'futureframe': 7704195,
    'mlep': 12560835,
}


def git_commit(path: Path):
    import subprocess
    try:
        return subprocess.check_output(
            ['git', '-c', 'safe.directory=*', '-C', str(path), 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _find_libdevice():
    candidates = [
        Path(os.environ.get('TF_LIBDEVICE', '')),
        Path(os.environ.get('CONDA_PREFIX', '')) / 'lib/libdevice.10.bc',
        Path('/home/lyx/miniconda3/envs/uav_tf1/lib/libdevice.10.bc'),
    ]
    candidates += sorted(Path('/usr/local').glob('cuda*/nvvm/libdevice/libdevice.10.bc'))
    candidates += sorted(Path('/usr').glob('local/cuda*/nvvm/libdevice/libdevice.10.bc'))
    for p in candidates:
        if p and p.is_file():
            return p
    return None


def _prepare_runtime_cwd():
    """TF2's MLIR kernel generator looks for ./libdevice.10.bc in the cwd."""
    src = _find_libdevice()
    if src is None:
        print('WARNING: libdevice.10.bc not found; GPU graph may fail for rsqrt',
              flush=True)
        return None
    tmp = Path(tempfile.mkdtemp(prefix='tf2_libdevice_'))
    shutil.copy2(src, tmp / 'libdevice.10.bc')
    os.chdir(tmp)
    print('runtime cwd', tmp, 'libdevice', src, flush=True)
    return tmp


def _futureframe_graph(batch: int):
    sys.path.insert(0, str(CLONE / 'Future Frame/Codes'))
    import unet
    g = tf.Graph()
    with g.as_default():
        x = tf.compat.v1.placeholder(tf.float32, [batch, 256, 256, 12], name='input')
        y = unet.unet(x, layers=4, features_root=64, filter_size=3,
                      pool_size=2, output_channel=3)
        vs = tf.compat.v1.trainable_variables()
        params = sum(int(np.prod(v.shape)) for v in vs)
    return g, x, y, params


def _mlep_graph(batch: int):
    sys.path.insert(0, str(CLONE / 'MLEP'))
    from models import prediction_networks_dict
    net = prediction_networks_dict['cyclegan_convlstm']
    g = tf.Graph()
    with g.as_default():
        x = tf.compat.v1.placeholder(tf.float32, [batch, 4, 224, 224, 3], name='input')
        y, _, _ = net(inputs=x, use_decoder=True)
        vs = tf.compat.v1.trainable_variables()
        params = sum(int(np.prod(v.shape)) for v in vs)
    return g, x, y, params


METHODS = {
    'futureframe': {
        'name': 'Future Frame',
        'input_hwt': '256x256x4',
        'shape': (256, 256, 12),
        'builder': _futureframe_graph,
        'commit_path': CLONE / 'Future Frame',
        'out': OUT / 'Future Frame/default/benchmark_v2_tf2compat.json',
        'notes': (
            'Official TF1 generator (U-Net), 4 frames -> next frame; inference does not '
            'use FlowNet. No official checkpoint; random init. Model code unchanged; '
            'TF2.20 contrib shims only. FLOPs from TF1.15 profiler on identical graph; '
            'params exact-match TF1. Timing is runtime-adapted, not TF1.15.'
        ),
    },
    'mlep': {
        'name': 'MLEP',
        'input_hwt': '224x224x4',
        'shape': (4, 224, 224, 3),
        'builder': _mlep_graph,
        'commit_path': CLONE / 'MLEP',
        'out': OUT / 'MLEP/default/benchmark_v2_tf2compat.json',
        'notes': (
            'Official TF1 cyclegan_convlstm prediction net, 4 frames -> next frame; '
            'no official checkpoint; random init. Model code unchanged; TF2.20 '
            'contrib shims (including TF1-compatible ConvLSTMCell) only. FLOPs from '
            'TF1.15 profiler on identical graph; params exact-match TF1. Timing is '
            'runtime-adapted, not TF1.15.'
        ),
    },
}


def _config():
    c = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    c.gpu_options.allow_growth = True
    return c


def _p95(values):
    return sorted(values)[max(0, int(0.95 * len(values)) - 1)]


def _run_batch(builder, shape, batch, warmup, iters, rounds):
    g, x, y, params = builder(batch)
    with g.as_default():
        init = tf.compat.v1.global_variables_initializer()
    rng = np.random.RandomState(0)
    feed_val = rng.randn(batch, *shape).astype(np.float32)
    with tf.compat.v1.Session(graph=g, config=_config()) as sess:
        sess.run(init)
        feed = {x: feed_val}
        for _ in range(warmup):
            sess.run(y, feed_dict=feed)

        if batch == 1:
            out_rounds = []
            for _ in range(rounds):
                times = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    sess.run(y, feed_dict=feed)
                    times.append((time.perf_counter() - t0) * 1000.0)
                out_rounds.append(times)
            return out_rounds, params

        round_times = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(iters):
                sess.run(y, feed_dict=feed)
            round_times.append(time.perf_counter() - t0)
        return round_times, params


def measure(method: str, warmup: int, iters: int, rounds: int) -> dict:
    spec = METHODS[method]
    print(f'[{method}] building BS1', flush=True)
    bs1_rounds, params = _run_batch(
        spec['builder'], spec['shape'], 1, warmup, iters, rounds)
    all_ms = [v for r in bs1_rounds for v in r]
    p50 = statistics.median(all_ms)
    mean = statistics.mean(all_ms)
    p95 = _p95(all_ms)

    print(f'[{method}] building BS16', flush=True)
    bs16_rounds, params16 = _run_batch(
        spec['builder'], spec['shape'], 16, warmup, iters, rounds)
    bs16_median = statistics.median(bs16_rounds)
    fps_bs16 = (16 * iters) / bs16_median

    if params != params16:
        raise RuntimeError(f'param mismatch between BS1/BS16: {params} vs {params16}')
    if method in TF1_PARAMS and params != TF1_PARAMS[method]:
        raise RuntimeError(
            f'{method} params {params} != TF1 params {TF1_PARAMS[method]}; '
            'shim does not reproduce the official graph')

    total_flops = int(KNOWN_FLOPS[method])
    result = {
        'protocol_version': 2,
        'measurement_status': 'partial/unverified',
        'runtime_adapted': True,
        'model_code_modified': False,
        'method': spec['name'],
        'method_display': spec['name'],
        'variant': 'default (TF2.20 runtime-adapted)',
        'supersedes': None,
        'adapted_reason': (
            'Original TF1.15 wheel uses CUDA 10.0/cuDNN 7.6.5 and cannot execute '
            'cuBLAS/cuDNN on RTX 5880 Ada (sm_89). The unchanged TF1 model graph is '
            'run on TF2.20 with contrib API shims; latency is runtime-adapted, not '
            'the original TF1.15 runtime.'
        ),
        'code_commit': git_commit(spec['commit_path']),
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
        'training_trainable_params_m': params / 1e6,
        'training_only_params_m': 0.0,
        'frozen_inference_params_m': 0.0,
        'benchmark_source': str(spec['out'].relative_to(ROOT)),
        'notes': spec['notes'],
        'profiling_config': (
            f'{GPU_NAME} ({GPU_UUID}); FP32; TF2.20 runtime-adapted; '
            f'BS1 {warmup}/{iters}/{rounds}; BS16 {warmup}/{iters}/{rounds}; '
            'TF Session.run is synchronous'
        ),
        'gpu': GPU_NAME,
        'gpu_uuid': GPU_UUID,
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
            'TensorFlow 1.15 profiler float_operation (CPU graph; identical graph); '
            'TF2 runtime does not cross-check the TF1 profiler count'
        ),
        'flops_ops_covered': ['TF1 graph ops (Conv2D, Conv2DBackpropInput, MatMul, etc.)'],
        'flops_ops_missing': ['TF1 profiler is not cross-checked by the v2 PyTorch operator counter'],
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
            'tf1_params': TF1_PARAMS[method],
            'match': params == TF1_PARAMS[method],
        },
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['futureframe', 'mlep', 'both'], default='both')
    p.add_argument('--warmup', type=int, default=WARMUP)
    p.add_argument('--iters', type=int, default=ITERS)
    p.add_argument('--rounds', type=int, default=ROUNDS)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        args.warmup, args.iters, args.rounds = 3, 10, 1

    _prepare_runtime_cwd()
    methods = ['futureframe', 'mlep'] if args.method == 'both' else [args.method]
    for method in methods:
        result = measure(method, args.warmup, args.iters, args.rounds)
        out = METHODS[method]['out']
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        print('wrote', out, flush=True)
        print(json.dumps({k: v for k, v in result.items() if k in (
            'method', 'measurement_status', 'runtime_adapted',
            'inference_params_m', 'flops_g_per_output_frame', 'flops_verified',
            'latency_bs1_p50_ms_per_output_frame', 'fps_bs1', 'fps_bs16',
            'tf1_params_check')}, ensure_ascii=False), flush=True)
        gc.collect()


if __name__ == '__main__':
    main()
