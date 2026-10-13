"""Trainable shared two-direction depth projection for adjacent teacher triplets.

The six raw coefficients start at the fixed mean/trend basis. Gram-Schmidt in the
forward pass keeps the effective 2x3 matrix row-orthonormal, so this ablation
changes the retained depth subspace without changing the rank-two bottleneck.
"""
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck_depthrank import DepthRank2OCBE


class LearnableDepthRank2OCBE(DepthRank2OCBE):
    """One shared learnable depth basis for all four three-tap groups."""

    def __init__(self, embed_dim=768, n_taps=12, code_dim=64,
                 fuse_hidden=128, use_tap_norm=True):
        super().__init__(embed_dim=embed_dim, n_taps=n_taps, code_dim=code_dim,
                         fuse_hidden=fuse_hidden, use_tap_norm=use_tap_norm)
        q_fixed = torch.tensor([
            [1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)],
            [-1 / sqrt(2), 0.0, 1 / sqrt(2)],
        ], dtype=torch.float32)
        self.raw_depth_basis = nn.Parameter(q_fixed)

    def effective_depth_basis(self):
        """Return an orthonormal 2x3 basis without a new loss coefficient."""
        first = F.normalize(self.raw_depth_basis[0], dim=0, eps=1e-8)
        second = self.raw_depth_basis[1]
        second = second - torch.dot(second, first) * first
        second = F.normalize(second, dim=0, eps=1e-8)
        return torch.stack((first, second), dim=0)

    def forward(self, taps):
        if len(taps) != self.n_taps:
            raise ValueError(f"expected {self.n_taps} taps, got {len(taps)}")
        normalized = [norm(tap) for norm, tap in zip(self.tap_norms, taps)]
        q = self.effective_depth_basis()
        modes = []
        for start in range(0, self.n_taps, 3):
            first, middle, last = normalized[start:start + 3]
            modes.append(q[0, 0] * first + q[0, 1] * middle + q[0, 2] * last)
            modes.append(q[1, 0] * first + q[1, 1] * middle + q[1, 2] * last)
        return self.fuse(torch.cat(modes, dim=-1))


def build_learnable_depthrank2_bottleneck(n_taps=12, code_dim=64, embed_dim=768,
                                         fuse_hidden=128, use_tap_norm=True):
    return LearnableDepthRank2OCBE(embed_dim=embed_dim, n_taps=n_taps,
                                  code_dim=code_dim, fuse_hidden=fuse_hidden,
                                  use_tap_norm=use_tap_norm)
