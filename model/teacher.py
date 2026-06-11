"""Appearance teacher for UAV-VAD — a FROZEN DINOv3 ViT-B/16 per-frame feature extractor.

Why DINOv3 (replacing the MADViT DeiT teacher):
  same size as DeiT-base (~86M params, embed_dim=768) — so any gain is NOT from capacity — but
  trained self-supervised on ~1.7B images. Two task-relevant consequences:
    (1) every PATCH token is explicitly discriminative (the axis our per-patch top-k discrepancy
        reads), vs DeiT's classification training where patch tokens are a by-product;
    (2) far smaller domain gap on out-of-ImageNet local structures (e.g. the small top-down
        railway obstacle) — directly targets the "teacher can't separate anomaly from look-alike
        normal" (R1) failure.

Design contract:
  * The teacher is the FIXED "normal-vs-everything" reference. Its weights are frozen ON PURPOSE:
    fine-tuning it on normal-only data would collapse the teacher-student discrepancy that IS the
    anomaly detector. This module therefore holds ZERO trainable parameters.
  * Dim-alignment to the student is a TRAINABLE projection on the STUDENT side — not here. This
    module is a pure extractor and is deliberately opinion-free (returns RAW features; the loss /
    readout decides normalization, temperature, top-k, etc.).

Token layout (DINOv3, timm 'Eva' impl): [CLS, reg0..reg3, patch...] -> num_prefix_tokens=5, so patch
tokens are tokens[:, 5:]. DINOv3 uses RoPE (no learned absolute pos-embed), so it is resolution-
tolerant. We feed the model's NATIVE 256x256 -> 16x16 = 256 patches (= 261 tokens; this is the
train.py default, matching DINOv3's training resolution). 224x224 -> 14x14 = 196 patches (= 201
tokens) is only an optional drop-in match with the old DeiT pipeline. Readout is patch-count-agnostic.
timm's DINOv3 has no get_intermediate_layers(), so per-block features are captured with forward hooks
(same approach as the MADViT repro).

NOTE: the "L5-L8 is the discriminative depth" finding was measured on the DeiT teacher; it must be
re-verified per-layer for DINOv3. Default here taps ALL 12 blocks so that analysis stays open.
"""
import torch
import torch.nn as nn
import timm

DINOV3_NAME = 'vit_base_patch16_dinov3'


class DINOv3Teacher(nn.Module):
    def __init__(self, layers=None, pretrained=True):
        super().__init__()
        self.vit = timm.create_model(DINOV3_NAME, pretrained=pretrained)
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad_(False)

        self.embed_dim = self.vit.embed_dim                  # 768
        self.num_blocks = len(self.vit.blocks)               # 12
        self.num_prefix = self.vit.num_prefix_tokens         # 5  (1 cls + 4 register)
        self.num_register = self.num_prefix - 1              # 4
        self.layers = list(range(self.num_blocks)) if layers is None else list(layers)

        # recommended input preprocessing (read from the pretrained config, do NOT hardcode)
        cfg = self.vit.pretrained_cfg
        self.img_mean = tuple(cfg.get('mean', (0.485, 0.456, 0.406)))
        self.img_std = tuple(cfg.get('std', (0.229, 0.224, 0.225)))
        self.input_size = tuple(cfg.get('input_size', (3, 224, 224)))

        self._feats = {}
        self._handles = [self.vit.blocks[li].register_forward_hook(self._mk_hook(li))
                         for li in self.layers]

    def _mk_hook(self, li):
        def hook(_m, _inp, out):
            self._feats[li] = out
        return hook

    def train(self, mode=True):
        # a frozen teacher must stay in eval regardless of the parent's train() call
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        """x: (B,3,H,W) already normalized with self.img_mean/std.

        Returns dict of RAW (un-normalized) features:
          'patches' : list over self.layers, each (B, 196, 768)  — per-block patch tokens
          'cls'     : (B, 768)                                   — final (post-norm) CLS token
          'tokens'  : list over self.layers, each (B, 201, 768)  — full raw block outputs
          'layers'  : the tapped block indices
        The loss/readout is responsible for L2-normalization, temperature, top-k, etc.
        """
        self._feats.clear()
        ff = self.vit.forward_features(x)                    # (B, 201, 768) post-final-norm
        tokens = [self._feats[li] for li in self.layers]     # raw per-block (B, 201, 768)
        patches = [t[:, self.num_prefix:] for t in tokens]   # (B, 196, 768)
        cls = ff[:, 0]                                        # (B, 768)
        return {'patches': patches, 'cls': cls, 'tokens': tokens, 'layers': list(self.layers)}


def build_appearance_teacher(layers=None, pretrained=True):
    """Factory: frozen DINOv3 appearance teacher. `layers` = block indices to tap (default all 12)."""
    return DINOv3Teacher(layers=layers, pretrained=pretrained)
