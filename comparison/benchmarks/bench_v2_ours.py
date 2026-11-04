#!/usr/bin/env python3
"""v2 benchmark for the six Ours DA variants.

Writes only benchmark_v2_ours_<variant>.json and benchmark_v2_ours.json.
Never writes the frozen v1 benchmark_ours.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from bench_v2_common import ROOT as REPO_ROOT, git_commit, measure, write_result
from model import anomaly_readouts, build_appearance_teacher
from scripts.da_variant_runner import VARIANT_SPECS
from scripts.uit_appearance_runner import build_variant


class OursModel(nn.Module):
    def __init__(self, teacher, ocbe, student, taps):
        super().__init__()
        self.teacher = teacher
        self.ocbe = ocbe
        self.student = student
        self.taps = list(taps)

    def forward(self, x):
        all_taps = self.teacher(x)['patches']
        taps = [all_taps[i] for i in self.taps]
        rec = self.student(self.ocbe(taps))
        return anomaly_readouts(taps, rec, k=5)['topk']


def build_meta(vid: str, spec: dict, teacher_params: int, trainable: int,
               out_path: Path) -> dict:
    return {
        'method': 'ours',
        'method_display': 'Ours',
        'variant': vid,
        'label': spec['label'],
        'supersedes': 'comparison/da/benchmark_ours.json',
        'code_commit': git_commit(REPO_ROOT),
        'input_hwt': '256x256x1',
        'input_frames_t': 1,
        'effective_output_scores_k': 1,
        'window_stride': 1,
        'frame_coverage': 1.0,
        'weights_state': 'architecture-init',
        'da_adapted': True,
        'auc_source': f'experiments/da/appearance/20261103_variant_matrix_{spec["label"].lower()}_{vid}_s42 (seed42, 30ep)',
        'inference_params_m': (teacher_params + trainable) / 1e6,
        'training_trainable_params_m': trainable / 1e6,
        'training_only_params_m': 0.0,
        'frozen_inference_params_m': teacher_params / 1e6,
        'benchmark_source': str(out_path.relative_to(REPO_ROOT)),
        'notes': (
            'DINOv3 teacher runs all 12 blocks; taps are output selection, not compute pruning. '
            f'teacher_taps={spec["taps"]}; readout=top5; architecture-init (FLOPs/latency are weight-independent).'
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', type=Path, default=REPO_ROOT / 'comparison/da')
    p.add_argument('--variants', nargs='*', default=list(VARIANT_SPECS))
    p.add_argument('--no-profiler', action='store_true')
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')

    teacher = build_appearance_teacher(layers=list(range(12))).cuda().eval()
    teacher_params = sum(p.numel() for p in teacher.parameters())
    x1 = torch.randn(1, 3, 256, 256, device='cuda')
    x16 = torch.randn(16, 3, 256, 256, device='cuda')
    aggregate = {'protocol_version': 2, 'method': 'ours', 'variants': {}}
    for vid in VARIANT_SPECS:
        p = args.out_dir / f'benchmark_v2_ours_{vid}.json'
        if p.is_file():
            aggregate['variants'][vid] = json.loads(p.read_text())

    for vid in args.variants:
        spec = VARIANT_SPECS[vid]
        print(f'=== {spec["label"]} {vid} ===', flush=True)
        ocbe, student = build_variant({'variant': spec['builder'], 'seed': 42}, 42)
        ocbe = ocbe.cuda().eval()
        student = student.cuda().eval()
        trainable = sum(p.numel() for p in ocbe.parameters()) + sum(p.numel() for p in student.parameters())
        model = OursModel(teacher, ocbe, student, spec['taps']).cuda().eval()
        out_path = args.out_dir / f'benchmark_v2_ours_{vid}.json'
        meta = build_meta(vid, spec, teacher_params, trainable, out_path)
        result = measure(lambda x: model(x), model, x1, x16, meta,
                         run_profiler=not args.no_profiler)
        write_result(out_path, result)
        aggregate['variants'][vid] = result
        del model, ocbe, student
        gc.collect()
        torch.cuda.empty_cache()

    agg_path = args.out_dir / 'benchmark_v2_ours.json'
    agg_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + '\n')
    print('wrote', agg_path, flush=True)
    print('| variant | FLOPs G | P50 ms | FPS BS1 | FPS BS16 | verified |', flush=True)
    print('|---|---:|---:|---:|---:|---|', flush=True)
    for vid, r in aggregate['variants'].items():
        print(f"| {vid} | {r['flops_g_per_output_frame']:.3f} | "
              f"{r['latency_bs1_p50_ms_per_output_frame']:.3f} | "
              f"{r['fps_bs1']:.1f} | {r['fps_bs16']:.1f} | {r['flops_verified']} |", flush=True)


if __name__ == '__main__':
    main()
