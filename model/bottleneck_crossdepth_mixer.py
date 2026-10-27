"""Nonlinear cross-depth mixer followed by a learnable group-axis Rank2.

The local teacher-triplet Q and the group-axis mixer act on different tensor
axes. A zero-start 4x4 residual mixer learns group relations without adding
the many small value/gate/hop operators used by DGT.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck_depthrank_learnable import LearnableDepthRank2OCBE


class CrossDepthMixerBottleneck(LearnableDepthRank2OCBE):
    def __init__(self, embed_dim=768, n_taps=12, code_dim=64,
                 hidden=128, use_tap_norm=True,
                 basis_init="random_orthonormal", basis_seed=42):
        super().__init__(embed_dim=embed_dim, n_taps=n_taps, code_dim=code_dim,
                         fuse_hidden=128, use_tap_norm=use_tap_norm,
                         basis_init=basis_init, basis_seed=basis_seed)
        if n_taps != 12:
            raise ValueError("CrossDepthMixerBottleneck expects twelve taps")
        self.depth_mixer = nn.Parameter(torch.zeros(4, 4))
        mean = torch.full((4,), 0.5, dtype=torch.float32)
        trend = torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=torch.float32)
        trend = trend / math.sqrt(float((trend * trend).sum()))
        self.raw_group_basis = nn.Parameter(torch.stack((mean, trend)))
        self.fuse = nn.Sequential(
            nn.Linear(2 * 2 * embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, code_dim),
        )

    def effective_group_basis(self):
        first = F.normalize(self.raw_group_basis[0], dim=0, eps=1e-8)
        second = self.raw_group_basis[1] - torch.dot(self.raw_group_basis[1], first) * first
        second = F.normalize(second, dim=0, eps=1e-8)
        return torch.stack((first, second))

    def _group_tensor(self, normalized, q):
        groups = []
        for start in range(0, self.n_taps, 3):
            first, middle, last = normalized[start:start + 3]
            mode0 = q[0, 0] * first + q[0, 1] * middle + q[0, 2] * last
            mode1 = q[1, 0] * first + q[1, 1] * middle + q[1, 2] * last
            groups.append(torch.stack((mode0, mode1), dim=2))
        return torch.stack(groups, dim=2)

    def forward(self, taps):
        if len(taps) != self.n_taps:
            raise ValueError(f"expected {self.n_taps} taps, got {len(taps)}")
        normalized = [norm(tap) for norm, tap in zip(self.tap_norms, taps)]
        q = self.effective_depth_basis()
        if (self._inference_depth_projector is not None
                and not self.training and not torch.is_grad_enabled()):
            from .depthrank_projection import project_depth_rank2
            b, p, _ = normalized[0].shape
            group = project_depth_rank2(normalized, q).view(b, p, 4, 2, self.embed_dim)
        else:
            group = self._group_tensor(normalized, q)
        mixed = group + F.gelu(torch.einsum("gh,bphrc->bpgrc", self.depth_mixer, group))
        modes = torch.einsum("sg,bpgrc->bpsrc", self.effective_group_basis(), mixed)
        return self.fuse(modes.reshape(modes.shape[0], modes.shape[1], -1))


def build_cross_depth_mixer_bottleneck(n_taps=12, code_dim=64, embed_dim=768,
                                       hidden=128, use_tap_norm=True,
                                       basis_init="random_orthonormal", basis_seed=42):
    return CrossDepthMixerBottleneck(
        embed_dim=embed_dim, n_taps=n_taps, code_dim=code_dim,
        hidden=hidden, use_tap_norm=use_tap_norm,
        basis_init=basis_init, basis_seed=basis_seed)
