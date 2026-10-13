"""Token-wise depth-rank-two OCBE for 12 frozen teacher taps.

For each adjacent group of three tap features, retain an orthonormal mean mode and
a first-to-last depth-change mode. The discarded mode is the second-order depth
curvature. The final code width stays 64, so the one-class information bottleneck
and the downstream student/loss contract remain unchanged.
"""
from math import sqrt

import torch
import torch.nn as nn


class DepthRank2OCBE(nn.Module):
    """Compress each three-tap group along depth before channel fusion."""

    def __init__(self, embed_dim=768, n_taps=12, code_dim=64,
                 fuse_hidden=128, use_tap_norm=True):
        super().__init__()
        if n_taps % 3:
            raise ValueError(f"n_taps must be divisible by 3, got {n_taps}")
        if code_dim >= embed_dim:
            raise ValueError("code_dim must be narrower than teacher features")
        self.embed_dim = embed_dim
        self.n_taps = n_taps
        self.code_dim = code_dim
        self.tap_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) if use_tap_norm else nn.Identity()
            for _ in range(n_taps)
        ])
        self.fuse = nn.Sequential(
            nn.Linear((n_taps // 3) * 2 * embed_dim, fuse_hidden),
            nn.GELU(),
            nn.Linear(fuse_hidden, code_dim),
        )

    def forward(self, taps):
        if len(taps) != self.n_taps:
            raise ValueError(f"expected {self.n_taps} taps, got {len(taps)}")
        normalized = [norm(tap) for norm, tap in zip(self.tap_norms, taps)]
        modes = []
        for start in range(0, self.n_taps, 3):
            first, middle, last = normalized[start:start + 3]
            modes.append((first + middle + last) / sqrt(3.0))
            modes.append((last - first) / sqrt(2.0))
        return self.fuse(torch.cat(modes, dim=-1))


def build_depthrank2_bottleneck(n_taps=12, code_dim=64, embed_dim=768,
                                fuse_hidden=128, use_tap_norm=True):
    return DepthRank2OCBE(embed_dim=embed_dim, n_taps=n_taps,
                          code_dim=code_dim, fuse_hidden=fuse_hidden,
                          use_tap_norm=use_tap_norm)
