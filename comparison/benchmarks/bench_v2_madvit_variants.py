#!/usr/bin/env python3
"""Measure MADViT scoring variants.

The cloned repo has no variant switch. We keep the same teacher+student
forward and change only the scoring function:
  Full        : repo default token-only path, decomposed per sample for BS16.
  Full-paper  : paper Sec. 5.4 Full score = mean-layer feature loss +
                final CLS/DIST token KL, returned per sample [B].
  Multi-layer : all 12 per-patch feature losses, no token KL, per sample [B].
  Last-layer  : final CLS/DIST token KL only, per sample [B].

Multi-layer / Last-layer share the same forward graph and are reconstructed
from the paper ablation; they are reference-only, not formal reproductions.
All modes return one score per sample.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from bench_v2_common import git_commit, measure, write_result
from bench_v2_extra import _stub_import_deps

CLONE = Path('/data1/data8tb/liangyuxuan/Comparison method')
OUT = ROOT / 'comparison/da/MADViT'


def _load_madvit():
    _stub_import_deps()
    code = CLONE / 'MADViT/code'
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location(
        'madvit_train_variants', code / 'train_val_VIT_Distillation.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MADVariant(nn.Module):
    def __init__(self, teacher, student, mod, mode):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.mode = mode
        self._full_loss = mod.multi_layer_distillation_loss

    def _token_loss(self, t_tok, s_tok):
        temperature = 4.0
        return 0.5 * (
            F.kl_div(F.log_softmax(s_tok[0] / temperature, dim=-1),
                     F.softmax(t_tok[0] / temperature, dim=-1), reduction='batchmean') +
            F.kl_div(F.log_softmax(s_tok[1] / temperature, dim=-1),
                     F.softmax(t_tok[1] / temperature, dim=-1), reduction='batchmean')
        ) * (temperature ** 2)

    def _feature_loss(self, t_feats, s_feats):
        losses = [1 - F.cosine_similarity(t[:, 2:].flatten(1), s[:, 2:].flatten(1)).mean()
                  for t, s in zip(t_feats, s_feats)]
        return sum(losses) / len(losses)

    def _token_loss_per_sample(self, t_tok, s_tok):
        temperature = 4.0

        def kl_per_sample(s, t):
            return F.kl_div(F.log_softmax(s / temperature, dim=-1),
                            F.softmax(t / temperature, dim=-1),
                            reduction='none').sum(dim=-1)

        return 0.5 * (kl_per_sample(s_tok[0], t_tok[0]) +
                      kl_per_sample(s_tok[1], t_tok[1])) * (temperature ** 2)

    def _feature_loss_per_sample(self, t_feats, s_feats):
        # Paper Sec. 5.4 / 3.4: cosine similarity per patch token (dim=-1),
        # then mean over patches; average the per-sample loss over layers.
        losses = [(1 - F.cosine_similarity(t[:, 2:], s[:, 2:], dim=-1)).mean(dim=-1)
                  for t, s in zip(t_feats, s_feats)]
        return sum(losses) / len(losses)

    def forward(self, x):
        t_feats, t_tok = self.teacher(x)
        s_feats, s_tok = self.student(x)
        if self.mode == 'Full':
            # Repo default returns token_loss only; decompose the same KL per sample.
            return self._token_loss_per_sample(t_tok, s_tok)
        if self.mode == 'Full-paper':
            return (self._feature_loss_per_sample(t_feats, s_feats) +
                    self._token_loss_per_sample(t_tok, s_tok))
        if self.mode == 'Multi-layer':
            return self._feature_loss_per_sample(t_feats, s_feats)
        if self.mode == 'Last-layer':
            # Paper Sec. 5.4: "Last-layer tokens only" = final CLS/DIST token loss only.
            return self._token_loss_per_sample(t_tok, s_tok)
        raise ValueError(self.mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modes', nargs='*', default=['Full-paper'],
                   help='default only writes the new Full-paper variant; pass others explicitly')
    p.add_argument('--no-profiler', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--output-name', default='benchmark_v2.json')
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    mod = _load_madvit()
    teacher = mod.Teacher('deit_base_distilled_patch16_224', pretrained=False).cuda().eval()
    student = mod.Student('deit_tiny_distilled_patch16_224', pretrained=False).cuda().eval()
    pt = sum(p.numel() for p in teacher.parameters())
    ps = sum(p.numel() for p in student.parameters())
    x1 = torch.randn(1, 3, 224, 224, device='cuda')
    x16 = torch.randn(16, 3, 224, 224, device='cuda')

    if args.smoke:
        for mode in args.modes:
            model = MADVariant(teacher, student, mod, mode).cuda().eval()
            with torch.inference_mode():
                y1 = model(x1)
                y16 = model(x16)
            print(f'smoke {mode}: BS1 {tuple(y1.shape) if y1.dim() else "scalar"} '
                  f'BS16 {tuple(y16.shape) if y16.dim() else "scalar"}', flush=True)
            del model, y1, y16
            gc.collect()
            torch.cuda.empty_cache()
        return

    for mode in args.modes:
        out_path = OUT / mode / args.output_name
        model = MADVariant(teacher, student, mod, mode).cuda().eval()
        meta = {
            'method': 'MADViT',
            'method_display': 'MADViT',
            'variant': mode,
            'adapted': mode in {'Multi-layer', 'Last-layer'},
            'supersedes': None,
            'code_commit': git_commit(CLONE / 'MADViT'),
            'input_hwt': '224x224x1',
            'input_frames_t': 1,
            'effective_output_scores_k': 1,
            'window_stride': 1,
            'frame_coverage': 1.0,
            'weights_state': 'random init (teacher/student; no checkpoint)',
            'da_adapted': False,
            'auc_source': 'user table / MADViT paper',
            'inference_params_m': (pt + ps) / 1e6,
            'training_trainable_params_m': ps / 1e6,
            'training_only_params_m': 0.0,
            'frozen_inference_params_m': pt / 1e6,
            'benchmark_source': str(out_path.relative_to(ROOT)),
            'notes': (
                ('Paper Sec. 5.4 Full scoring: per-patch (1 - cosine), mean over patches, then '
                 'mean over 12 layers + T^2/2 * sum of CLS/DIST KL; per-sample [B].'
                 if mode == 'Full-paper' else
                 'Repo-default Full: multi_layer_distillation_loss returns token_loss only; '
                 'the same KL is decomposed per sample for BS16. reference-only, not paper Full.'
                 if mode == 'Full' else
                 f'Reconstructed {mode} scoring from paper Sec. 5.4; per-sample [B]; '
                 'reference-only, not a formal reproduction.')
            ),
            'output_shape_bs1': '[1]',
            'output_shape_bs16': '[16]',
            'scores_per_sample': 1,
            'batch_output_aggregated': False,
            'paper_matched': mode == 'Full-paper',
            'paper_matched_note': (
                'paper Sec. 5.4 scoring and official DeiT teacher/student graph; '
                'random-init weights (efficiency is weight-independent; AUC not paired)'
                if mode == 'Full-paper' else
                'repo-default token-only scoring; not paper Full'
                if mode == 'Full' else
                'reconstructed from paper Sec. 5.4; not a formal reproduction'),
            'ranking_status': 'formal' if mode == 'Full-paper' else 'reference-only',
        }
        result = measure(lambda x: model(x), model, x1, x16, meta,
                         run_profiler=not args.no_profiler)
        write_result(out_path, result)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
