#!/usr/bin/env python3
"""v2 benchmark for external methods: GANomaly, MemAE, MKD.

Writes only benchmark_v2_<method>.json. Never writes the frozen v1 files.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from bench_v2_common import git_commit, measure, write_result

CLONE = Path('/data1/data8tb/liangyuxuan/Comparison method')
OUT = ROOT / 'comparison/da'


def _meta(method, variant, input_hwt, t, k, stride, frame_coverage,
          inference_m, trainable_m, training_only_m, frozen_m,
          code_commit, weights_state, auc_source, notes, supersedes, out_path):
    return {
        'method': method,
        'method_display': method,
        'variant': variant,
        'supersedes': supersedes,
        'code_commit': code_commit,
        'input_hwt': input_hwt,
        'input_frames_t': t,
        'effective_output_scores_k': k,
        'window_stride': stride,
        'frame_coverage': frame_coverage,
        'weights_state': weights_state,
        'da_adapted': False,
        'auc_source': auc_source,
        'inference_params_m': inference_m,
        'training_trainable_params_m': trainable_m,
        'training_only_params_m': training_only_m,
        'frozen_inference_params_m': frozen_m,
        'benchmark_source': str(out_path.relative_to(ROOT)),
        'notes': notes,
    }


def _ganomaly(out_path):
    sys.path.insert(0, str(CLONE / 'GANomaly'))
    from lib.networks import NetD, NetG
    opt = SimpleNamespace(isize=256, nz=100, nc=3, ngf=64, ndf=64, ngpu=1, extralayers=0)
    g = NetG(opt).cuda().eval()
    d = NetD(opt).cuda().eval()
    pg = sum(p.numel() for p in g.parameters())
    pd = sum(p.numel() for p in d.parameters())

    def fn(x):
        _, li, lo = g(x)
        return ((li - lo) ** 2).mean(dim=1)

    meta = _meta(
        'GANomaly', 'default', '256x256x1', 1, 1, 1, 1.0,
        pg / 1e6, (pg + pd) / 1e6, pd / 1e6, 0.0,
        git_commit(CLONE / 'GANomaly'), 'random init',
        'ANDT Table VI / user table (DA-adapted baseline); not the same config as this efficiency run',
        'G is the inference graph; D is training-only (G+D in training trainable). '
        'Full G forward includes encoder1 -> decoder -> encoder2; scoring is latent MSE.',
        'comparison/da/benchmark_ganomaly.json', out_path)
    return g, fn, meta


class _VGGSrc(nn.Module):
    """VGG16 source with the same 13 feature taps as MKD's Vgg16."""

    TAPS = (1, 4, 6, 9, 11, 13, 16, 18, 20, 23, 25, 27, 30)

    def __init__(self):
        super().__init__()
        import torchvision
        self.features = nn.ModuleList(list(torchvision.models.vgg16(weights=None).features)).eval()

    def forward(self, x):
        out = []
        for i in range(31):
            x = self.features[i](x)
            if i in self.TAPS:
                out.append(x)
        return out


class _MKDModel(nn.Module):
    # MKD's VGG.forward registers a gradient hook at layer 11; that fails under
    # no_grad/inference_mode. This is the same feature collection without hook.
    CLONER_TAPS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38)

    def __init__(self, source, cloner, lamda=0.01):
        super().__init__()
        self.source = source
        self.cloner = cloner
        self.lamda = lamda

    def _cloner_features(self, x):
        out = []
        for i, layer in enumerate(self.cloner.features):
            x = layer(x)
            if i in self.CLONER_TAPS:
                out.append(x)
        return out

    def forward(self, x):
        yp = self._cloner_features(x)
        yr = self.source(x)
        losses = []
        for i in (6, 9, 12):
            a, b = yp[i], yr[i]
            if a.shape[-2:] != b.shape[-2:]:
                b = F.interpolate(b, size=a.shape[-2:], mode='bilinear',
                                  align_corners=False)
            cos = F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)
            mse = ((a - b) ** 2).mean(dim=(1, 2, 3))
            losses.append(1 - cos + self.lamda * mse)
        return sum(losses)


def _mkd(out_path):
    sys.path.insert(0, str(CLONE / 'MKD'))
    from models.network import make_arch
    cfg = {
        'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'B': [16, 16, 'M', 16, 128, 'M', 16, 16, 256, 'M', 16, 16, 512, 'M', 16, 16, 512, 'M'],
    }
    source = _VGGSrc().cuda().eval()
    cloner = make_arch('B', cfg, use_bias=False, batch_norm=True).cuda().eval()
    ps = sum(p.numel() for p in source.parameters())
    pc = sum(p.numel() for p in cloner.parameters())
    model = _MKDModel(source, cloner).cuda().eval()

    meta = _meta(
        'MKD', 'default', '128x128x1', 1, 1, 1, 1.0,
        (ps + pc) / 1e6, pc / 1e6, 0.0, ps / 1e6,
        git_commit(CLONE / 'MKD'),
        'random init (VGG16 source; original pretrained weights unavailable)',
        'ANDT Table VI / user table (DA-adapted baseline); efficiency is MVTec 128 config',
        'Multi-layer scoring at cloner/source taps (6,9,12): cosine + 0.01*MSE, per original test_functions.py. '
        'Source VGG16 is frozen; cloner cfg B is trainable. V1 only used the final layer and was wrong.',
        'comparison/da/benchmark_mkd.json', out_path)
    return model, lambda x: model(x), meta


def _memae(out_path):
    sys.path.insert(0, str(CLONE / 'MemAE'))
    from models.memae_3dconv import AutoEncoderCov3DMem
    m = AutoEncoderCov3DMem(chnum_in=3, mem_dim=2000, shrink_thres=0.0025).cuda().eval()
    p = sum(p.numel() for p in m.parameters())

    def fn(x):
        r = m(x)['output'] - x
        sp = (r ** 2).sum(dim=1).sqrt()  # B x T x H x W
        return sp.mean(dim=(1, 2, 3))    # one clip-level score, as original test

    meta = _meta(
        'MemAE', 'default', '256x256x16', 16, 1, 1, 1.0,
        p / 1e6, p / 1e6, 0.0, 0.0,
        git_commit(CLONE / 'MemAE'), 'random init',
        'ANDT Table VI / user table (DA-adapted baseline); not the same config as this efficiency run',
        'Original script_testing.py returns one clip-level reconstruction error per 16-frame clip '
        '(mean over T,H,W), mapped to a center frame; K=1, stride=1. The old v1 BxT/16 calculation was wrong. '
        'frame_coverage is approximately 1.0 excluding boundary frames.',
        'comparison/da/benchmark_memae.json', out_path)
    return m, fn, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=['ganomaly', 'memae', 'mkd'])
    p.add_argument('--out-dir', type=Path, default=OUT)
    p.add_argument('--no-profiler', action='store_true')
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')

    out_path = args.out_dir / f'benchmark_v2_{args.method}.json'
    if args.method == 'ganomaly':
        model, fn, meta = _ganomaly(out_path)
        x1 = torch.randn(1, 3, 256, 256, device='cuda')
        x16 = torch.randn(16, 3, 256, 256, device='cuda')
    elif args.method == 'memae':
        model, fn, meta = _memae(out_path)
        x1 = torch.randn(1, 3, 16, 256, 256, device='cuda')
        x16 = torch.randn(16, 3, 16, 256, 256, device='cuda')
    else:
        model, fn, meta = _mkd(out_path)
        x1 = torch.randn(1, 3, 128, 128, device='cuda')
        x16 = torch.randn(16, 3, 128, 128, device='cuda')

    result = measure(fn, model, x1, x16, meta, run_profiler=not args.no_profiler)
    write_result(out_path, result)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
