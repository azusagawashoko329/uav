"""Minimal 6-frame adapted ANDT model.

The released ANDT/ANDT/model.py Decoder.forward does:
    output = x.view(B, 256, 16, 16)
which only works for one frame (256 patch tokens). The paper's prediction model
takes 6 consecutive frames and predicts the next frame. This adapter uses the
already-defined-but-commented Decoder.de_dense path:

    output = self.decoder(self.de_dense(x[:, 0]))

It also removes the hard-coded .cuda() and pdb.set_trace() in the upstream
forward. This is an ADAPTED variant, not the authors' official 6-frame code;
label results accordingly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ANDT_ROOT = Path(os.environ.get(
    'ANDT_ROOT',
    '/data1/data8tb/liangyuxuan/Comparison method/ANDT/ANDT',
))
sys.path.insert(0, str(ANDT_ROOT))

import model as andt_model  # noqa: E402


class AdaptedDecoder(andt_model.Decoder):
    def forward(self, x):
        # x: (B, 1 + 256*num_frames, 768); use CLS to predict the next frame.
        out = self.de_dense(x[:, 0])
        out = out.view(x.size(0), 256, 16, 16)
        return self.decoder(out)


class AdaptedANDT(andt_model.VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder = AdaptedDecoder()

    def forward(self, x):
        # x: (B, num_frames, 3, H, W); no .cuda()/pdb in the adapted path.
        x = x.float()
        all_emb = None
        for i in range(x.shape[1]):
            emb = self.embedding(x[:, i])
            emb = emb.permute(0, 2, 3, 1)
            b, h, w, c = emb.shape
            emb = emb.reshape(b, h * w, c)
            all_emb = emb if all_emb is None else torch.cat((emb, all_emb), 1)
        b, n, c = all_emb.shape
        emb = torch.cat([self.cls_token.repeat(b, 1, 1), all_emb], dim=1)
        feat = self.transformer(emb)
        return self.decoder(feat)


def build_adapted_andt(num_frames=6, image_size=256, patch_size=16,
                       emb_dim=768, mlp_dim=4096, num_heads=6, num_layers=2,
                       num_classes=25, dropout_rate=0.1):
    return AdaptedANDT(
        image_size=(image_size, image_size),
        patch_size=(patch_size, patch_size),
        emb_dim=emb_dim,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        attn_dropout_rate=0.0,
        dropout_rate=dropout_rate,
        num_frames=num_frames,
    )
