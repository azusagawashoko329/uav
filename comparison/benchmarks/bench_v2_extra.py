#!/usr/bin/env python3
"""v2 benchmarks for additional cloned methods.

Supported: LightViT-AD, MADViT, ST Dissoc., ASTT (STE-TTE), DAD-FSM.
Each method is built from its upstream code; no checkpoints are required for
FLOPs/latency. Methods whose code cannot be imported (HSTforU: mmcv; Future
Frame/MLEP: TF1; adVAE: tabular only) stay blocked.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
import types
from pathlib import Path

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


def _stub_import_deps():
    """MADViT repo is missing matplotlib/gaussian_blur/utils at HEAD."""
    mpl = types.ModuleType('matplotlib')
    plt = types.ModuleType('matplotlib.pyplot')
    mpl.pyplot = plt
    sys.modules['matplotlib'] = mpl
    sys.modules['matplotlib.pyplot'] = plt
    gb = types.ModuleType('gaussian_blur')
    gb.GaussianBlur = type('GaussianBlur', (), {'__init__': lambda self, *a, **k: None})
    sys.modules['gaussian_blur'] = gb
    ut = types.ModuleType('utils')
    ut.annot_peaks = lambda *a, **k: None
    sys.modules['utils'] = ut


def _meta(method, variant, input_hwt, t, k, stride, frame_coverage,
          inference_m, trainable_m, training_only_m, frozen_m,
          code_commit, weights_state, auc_source, notes, out_path,
          adapted=False):
    return {
        'method': method,
        'method_display': method,
        'variant': variant,
        'adapted': adapted,
        'supersedes': None,
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


def _lightvit(out_path):
    from importlib.util import module_from_spec, spec_from_file_location
    p = CLONE / 'LightViT-AD/lightvit_ad/models.py'
    spec = spec_from_file_location('lightvit_models', p)
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    teacher = m.ViTTeacher(pretrained=False).cuda().eval()
    student = m.ViTStudent(pretrained=False).cuda().eval()

    class LightViT(nn.Module):
        def __init__(self, t, s, mod):
            super().__init__()
            self.teacher = t
            self.student = s
            self._score = mod.anomaly_score

        def forward(self, x):
            t_tokens, latent = self.teacher(x)
            s_tokens = self.student(latent)
            return self._score(t_tokens, s_tokens)

    model = LightViT(teacher, student, m).cuda().eval()
    pt = sum(p.numel() for p in teacher.parameters())
    ps = sum(p.numel() for p in student.parameters())
    meta = _meta(
        'LightViT-AD', 'default', '224x224x1', 1, 1, 1, 1.0,
        (pt + ps) / 1e6, ps / 1e6, 0.0, pt / 1e6,
        git_commit(CLONE / 'LightViT-AD'),
        'random init (teacher/student; no checkpoint)',
        'user table / LightViT-AD paper',
        'Teacher + 6-block student; forward includes anomaly_score MSE. Teacher is frozen in '
        'training; trainable count is student only. Repo package __init__ imports matplotlib, so '
        'models.py is loaded directly; no checkpoint.',
        out_path)
    return model, lambda x: model(x), (
        torch.randn(1, 3, 224, 224, device='cuda'),
        torch.randn(16, 3, 224, 224, device='cuda'),
    ), meta


def _madvit(out_path):
    _stub_import_deps()
    code = CLONE / 'MADViT/code'
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location(
        'madvit_train', code / 'train_val_VIT_Distillation.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    teacher = m.Teacher('deit_base_distilled_patch16_224', pretrained=False).cuda().eval()
    student = m.Student('deit_tiny_distilled_patch16_224', pretrained=False).cuda().eval()

    class MAD(nn.Module):
        def __init__(self, t, s, mod):
            super().__init__()
            self.teacher = t
            self.student = s
            self._loss = mod.multi_layer_distillation_loss

        def forward(self, x):
            t_feats, t_tok = self.teacher(x)
            s_feats, s_tok = self.student(x)
            return self._loss(t_feats, t_tok, s_feats, s_tok)

    model = MAD(teacher, student, m).cuda().eval()
    pt = sum(p.numel() for p in teacher.parameters())
    ps = sum(p.numel() for p in student.parameters())
    meta = _meta(
        'MADViT', 'Full', '224x224x1', 1, 1, 1, 1.0,
        (pt + ps) / 1e6, ps / 1e6, 0.0, pt / 1e6,
        git_commit(CLONE / 'MADViT'),
        'random init (teacher/student; no checkpoint)',
        'user table / MADViT paper',
        'CODE DEFAULT ONLY: the repo has a single scoring path (all 12 skip connections + '
        'token loss), closest to the table "Full" variant; there is no Full/Multi-layer/'
        'Last-layer switch in code. Do not reuse this measurement for Multi-layer/Last-layer. '
        'Repo is missing gaussian_blur.py and utils.py; import-only stubs are used, model code '
        'is unchanged.',
        out_path)
    return model, lambda x: model(x), (
        torch.randn(1, 3, 224, 224, device='cuda'),
        torch.randn(16, 3, 224, 224, device='cuda'),
    ), meta


def _st_dissoc(out_path):
    repo = CLONE / 'ST Dissoc.'
    sys.path.insert(0, str(repo))
    from arch.model.PredRes_Model import PredRes_AE_Cluster_Model
    model = PredRes_AE_Cluster_Model(
        static_channel_in=3, static_channel_out=3,
        static_layer_struct=[1, 1, 1], static_layer_nums=3,
        motion_channel_in=12, motion_channel_out=3,
        motion_layer_struct=[1, 1, 1], motion_layer_nums=3,
        img_channel=3, frame_nums=5, cluster_num=32,
        blur_ratio=1, seq_tag=False, model_type='res',
    ).cuda().eval()
    # Upstream uses padding=False, which becomes (False, False) and fails
    # forward in current PyTorch; 0 is the intended value.
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d) and tuple(mod.padding) == (False, False):
            mod.padding = (0, 0)

    class ST(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x, None, ['E'])

    model = ST(model).cuda().eval()
    p = sum(p.numel() for p in model.parameters())
    meta = _meta(
        'ST Dissoc.', 'default', '256x256x5', 5, 1, 1, 1.0,
        p / 1e6, p / 1e6, 0.0, 0.0,
        git_commit(CLONE / 'ST Dissoc.'),
        'random init (no checkpoint)',
        'user table / ST Dissoc. paper',
        'Input is 5 RGB frames flattened to 15 channels (B,15,256,256). Upstream padding=False '
        'is treated as 0 for current PyTorch. Forward uses stage E (static+motion encoders, '
        'decoders, cluster map); output is one cluster map per window.',
        out_path, adapted=True)
    return model, lambda x: model(x), (
        torch.randn(1, 15, 256, 256, device='cuda'),
        torch.randn(16, 15, 256, 256, device='cuda'),
    ), meta


def _astt(out_path):
    repo = CLONE / 'ASTT'
    sys.path.insert(0, str(repo))
    from model_ste_tte import VisionTransformer
    base = VisionTransformer(
        image_size=(384, 384), patch_size=(16, 16), emb_dim=768,
        mlp_dim=1536, num_heads=6, num_layers=2, num_classes=25,
        attn_dropout_rate=0.0, dropout_rate=0.1, num_frames=4,
    ).cuda().eval()

    class ASTTScore(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(x)
            return F.mse_loss(out, torch.zeros_like(out))

    model = ASTTScore(base).cuda().eval()
    p = sum(p.numel() for p in model.parameters())
    meta = _meta(
        'ASTT', 'ste_tte', '384x384x4', 4, 1, 1, 1.0,
        p / 1e6, p / 1e6, 0.0, 0.0,
        git_commit(CLONE / 'ASTT'),
        'random init (no checkpoint)',
        'user table / ASTT paper',
        'STE-TTE variant (model_ste_tte.py): ViT-B/32 spatial transformer at 384 + temporal '
        'transformer over 4 frames, decoder predicts next 256x256 frame; forward includes MSE '
        'scoring. No checkpoint.',
        out_path)
    return model, lambda x: model(x), (
        torch.randn(1, 4, 3, 384, 384, device='cuda'),
        torch.randn(16, 4, 3, 384, 384, device='cuda'),
    ), meta


def _dadfsm(out_path):
    # DADFSM hardcodes pretrained=True for ResNet; avoid download and keep the
    # same architecture by forcing weights=None.
    import torchvision.models as tvm
    orig = tvm.resnet50
    tvm.resnet50 = lambda pretrained=True, **kw: orig(weights=None, **kw)
    sys.path.insert(0, str(CLONE / 'DAD-FSM'))
    from our_model import DADFSM

    class Args:
        img_size = (224, 224)
        in_channels = 3
        method = 'resnet50'
        num_frames = 4
        trans_linear_in_dim = 2048

    class DAD(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, inp):
            x, y = inp
            out = self.m(x, y)
            return F.mse_loss(out, torch.zeros_like(out))

    model = DAD(DADFSM(Args())).cuda().eval()
    p = sum(p.numel() for p in model.parameters())
    meta = _meta(
        'DAD-FSM', 'default', '224x224x4', 4, 1, 1, 1.0,
        p / 1e6, p / 1e6, 0.0, 0.0,
        git_commit(CLONE / 'DAD-FSM'),
        'random init (ResNet50 source weights unavailable)',
        'user table / DAD-FSM paper',
        'Two input streams (RGB + segmentation), each 4 frames; full model includes ResNet50 '
        'backbone, spatial/temporal attention, decoder and MSE scoring. ResNet50 is random-init '
        'because the upstream pretrained weights are not bundled; FLOPs/latency are '
        'weight-independent.',
        out_path, adapted=True)
    x1 = (torch.randn(1, 4, 3, 224, 224, device='cuda'),
          torch.randn(1, 4, 3, 224, 224, device='cuda'))
    x16 = (torch.randn(16, 4, 3, 224, 224, device='cuda'),
           torch.randn(16, 4, 3, 224, 224, device='cuda'))
    return model, lambda inp: model(inp), (x1, x16), meta


def _install_mmcv_shims():
    """HSTforU uses mmcv 1.x APIs; mmcv-lite 2.x lacks runner/parrots/UPSAMPLE_LAYERS."""
    import mmcv.cnn as cnn
    class Registry:
        def register_module(self, *a, **k):
            def deco(cls):
                return cls
            return deco
    if not hasattr(cnn, 'UPSAMPLE_LAYERS'):
        cnn.UPSAMPLE_LAYERS = Registry()
    runner = types.ModuleType('mmcv.runner')
    runner.load_checkpoint = lambda *a, **k: None
    sys.modules['mmcv.runner'] = runner
    parrots = types.ModuleType('mmcv.utils.parrots_wrapper')
    parrots._BatchNorm = torch.nn.modules.batchnorm._BatchNorm
    sys.modules['mmcv.utils.parrots_wrapper'] = parrots


def _hstforu(out_path):
    _install_mmcv_shims()
    repo = CLONE / 'HSTforU/HSTforU'
    sys.path.insert(0, str(repo))
    from models.dec import dec_618_1_unet as dec
    if not getattr(dec, '_v2_shimmed', False):
        def _build(cfg, **kw):
            if cfg.get('type') == 'DeconvModule':
                return dec.DeconvModule(kw['in_channels'], kw['out_channels'],
                                        norm_cfg=kw.get('norm_cfg'),
                                        act_cfg=kw.get('act_cfg'))
            return dec.build_upsample_layer(cfg=cfg, **kw)
        dec.build_upsample_layer = _build
        dec._v2_shimmed = True
    from models.HSTforU_Avenue import AnomalyDetection
    base = AnomalyDetection(None, None, is_trained=False).cuda().eval()

    class HST(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, frames):
            out = self.m(frames)
            return F.mse_loss(out, torch.zeros_like(out))

    model = HST(base).cuda().eval()
    p = sum(p.numel() for p in model.parameters())
    meta = _meta(
        'HSTforU', 'Avenue', '256x256x4', 4, 1, 1, 1.0,
        p / 1e6, p / 1e6, 0.0, 0.0,
        git_commit(CLONE / 'HSTforU'),
        'random init (no checkpoint)',
        'user table / HSTforU paper',
        'Avenue/PVT2-b2 variant: PVT2 encoder + TimeSformer + UNet decoder; 4 input frames, '
        'one next-frame prediction, MSE scoring. Uses mmcv-lite 2.x with compatibility shims for '
        'mmcv 1.x runner/parrots/UPSAMPLE_LAYERS; no checkpoint.',
        out_path, adapted=True)
    x1 = [torch.randn(1, 3, 256, 256, device='cuda') for _ in range(4)]
    x16 = [torch.randn(16, 3, 256, 256, device='cuda') for _ in range(4)]
    return model, lambda frames: model(frames), (x1, x16), meta


BUILDERS = {
    'lightvit': (_lightvit, OUT / 'LightViT-AD/default'),
    'madvit': (_madvit, OUT / 'MADViT/Full'),
    'st': (_st_dissoc, OUT / 'ST Dissoc./default'),
    'astt': (_astt, OUT / 'ASTT/default'),
    'dadfsm': (_dadfsm, OUT / 'DAD-FSM/default'),
    'hstforu': (_hstforu, OUT / 'HSTforU/default'),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=sorted(BUILDERS))
    p.add_argument('--out-dir', type=Path, default=None)
    p.add_argument('--no-profiler', action='store_true')
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')

    builder, default_out = BUILDERS[args.method]
    out_dir = args.out_dir or default_out
    out_path = out_dir / 'benchmark_v2.json'
    model, fn, (x1, x16), meta = builder(out_path)
    result = measure(fn, model, x1, x16, meta, run_profiler=not args.no_profiler)
    write_result(out_path, result)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
