#!/usr/bin/env python3
"""Measure MADViT Full / Multi-layer / Last-layer scoring variants.

The cloned repo has no variant switch. We keep the same teacher+student
forward and change only the scoring function:
  Full        : all 12 skip-connection feature losses + CLS/DIST token KL
                (exactly multi_layer_distillation_loss in the repo)
  Multi-layer : all 12 feature losses, no token KL
  Last-layer  : final CLS/DIST token KL only (paper Sec. 4.4)

Multi-layer / Last-layer share the same forward graph and are reconstructed
from the paper ablation; results are labelled adapted.
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

    def forward(self, x):
        t_feats, t_tok = self.teacher(x)
        s_feats, s_tok = self.student(x)
        if self.mode == 'Full':
            return self._full_loss(t_feats, t_tok, s_feats, s_tok)
        if self.mode == 'Multi-layer':
            return self._feature_loss(t_feats, s_feats)
        if self.mode == 'Last-layer':
            # Paper Sec. 4.4: "Last-layer tokens only" = final CLS/DIST token loss only.
            return self._token_loss(t_tok, s_tok)
        raise ValueError(self.mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modes', nargs='*', default=['Full', 'Multi-layer', 'Last-layer'])
    p.add_argument('--no-profiler', action='store_true')
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

    for mode in args.modes:
        out_path = OUT / mode / 'benchmark_v2.json'
        model = MADVariant(teacher, student, mod, mode).cuda().eval()
        meta = {
            'method': 'MADViT',
            'method_display': 'MADViT',
            'variant': mode,
            'adapted': mode != 'Full',
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
                f'Scoring variant {mode}. Definitions from MADViT paper Sec. 4.4: '
                'Full=all 12 feature losses + final token KL; Multi-layer=all 12 feature losses '
                'only; Last-layer=final CLS/DIST token KL only. Full is the repo multi_layer_'
                'distillation_loss; Multi-layer/Last-layer share the same forward graph and only '
                'change scoring, so this is an adapted reconstruction of the paper ablation.'
            ),
        }
        result = measure(lambda x: model(x), model, x1, x16, meta,
                         run_profiler=not args.no_profiler)
        write_result(out_path, result)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
