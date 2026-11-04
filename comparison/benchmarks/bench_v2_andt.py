#!/usr/bin/env python3
"""v2 benchmark for the adapted 6-frame ANDT model.

The released ANDT decoder only handles one frame. This benchmark uses the
minimal adapted decode path de_dense(feat[:,0]) from
comparison/da/ANDT/default/scripts/andt_adapted_model.py. Results are labelled
adapted, not official.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ADAPTER = ROOT / 'comparison/da/ANDT/default/scripts'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ADAPTER))

from bench_v2_common import git_commit, measure, write_result
from andt_adapted_model import build_adapted_andt

CLONE = Path('/data1/data8tb/liangyuxuan/Comparison method')
OUT = ROOT / 'comparison/da/ANDT/default'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', type=Path, default=OUT)
    p.add_argument('--no-profiler', action='store_true')
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')

    model = build_adapted_andt(
        num_frames=6, image_size=256, patch_size=16, emb_dim=768,
        mlp_dim=4096, num_heads=6, num_layers=2, num_classes=25,
        dropout_rate=0.1,
    ).cuda().eval()
    params = sum(p.numel() for p in model.parameters())
    out_path = args.out_dir / 'benchmark_v2.json'
    meta = {
        'method': 'ANDT',
        'method_display': 'ANDT',
        'variant': 'default',
        'adapted': True,
        'adapted_decoder': 'de_dense(feat[:,0])',
        'supersedes': None,
        'code_commit': git_commit(CLONE / 'ANDT'),
        'input_hwt': '256x256x6',
        'input_frames_t': 6,
        'effective_output_scores_k': 1,
        'window_stride': 1,
        'frame_coverage': 1.0,
        'weights_state': 'random init (official checkpoint unavailable)',
        'da_adapted': False,
        'auc_source': 'ANDT paper / user table (official 6-frame model); efficiency is the adapted decoder, not official',
        'inference_params_m': params / 1e6,
        'training_trainable_params_m': params / 1e6,
        'training_only_params_m': 0.0,
        'frozen_inference_params_m': 0.0,
        'benchmark_source': str(out_path.relative_to(ROOT)),
        'notes': (
            'ADAPTED 6-frame variant, not the authors official code. Upstream Decoder.forward only '
            'supports 1 frame; this uses de_dense(feat[:,0]) -> 256x16x16 -> decoder to predict one '
            'next frame. T=6, K=1, stride=1; frame_coverage ~1.0 excluding the first 5 frames. '
            'No official checkpoint; FLOPs/latency are weight-independent.'
        ),
    }
    x1 = torch.randn(1, 6, 3, 256, 256, device='cuda')
    x16 = torch.randn(16, 6, 3, 256, 256, device='cuda')
    result = measure(lambda x: model(x), model, x1, x16, meta,
                     run_profiler=not args.no_profiler)
    write_result(out_path, result)
    print('params M', params / 1e6, flush=True)
    del model, x1, x16
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
