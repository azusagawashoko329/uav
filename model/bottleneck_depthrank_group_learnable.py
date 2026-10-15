"""Four independent orthonormal depth bases for adjacent teacher triplets."""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck_depthrank import DepthRank2OCBE


class GroupLearnableDepthRank2OCBE(DepthRank2OCBE):
    """Compress each three-tap group with its own trainable 2x3 depth basis."""

    def __init__(self, embed_dim=768, n_taps=12, code_dim=64,
                 fuse_hidden=128, use_tap_norm=True, basis_init="mean_trend",
                 basis_seed=None):
        super().__init__(embed_dim=embed_dim, n_taps=n_taps, code_dim=code_dim,
                         fuse_hidden=fuse_hidden, use_tap_norm=use_tap_norm)
        n_groups = n_taps // 3
        if basis_init == "mean_trend":
            one = torch.tensor([
                [1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)],
                [-1 / sqrt(2), 0.0, 1 / sqrt(2)],
            ], dtype=torch.float32)
            q_init = one.repeat(n_groups, 1, 1)
        elif basis_init == "random_orthonormal":
            if basis_seed is None:
                raise ValueError("basis_seed is required for random_orthonormal")
            generator = torch.Generator(device="cpu").manual_seed(basis_seed)
            q_init = torch.randn((n_groups, 2, 3), generator=generator,
                                 dtype=torch.float32)
            first = F.normalize(q_init[:, 0, :], dim=-1)
            second = q_init[:, 1, :] - (q_init[:, 1, :] * first).sum(-1, keepdim=True) * first
            q_init = torch.stack((first, F.normalize(second, dim=-1)), dim=1)
        else:
            raise ValueError(f"unknown basis_init: {basis_init}")
        self.raw_depth_basis = nn.Parameter(q_init)

    def effective_depth_basis(self):
        """Apply differentiable Gram-Schmidt separately to each group."""
        first = F.normalize(self.raw_depth_basis[:, 0, :], dim=-1, eps=1e-8)
        second = self.raw_depth_basis[:, 1, :]
        second = second - (second * first).sum(-1, keepdim=True) * first
        second = F.normalize(second, dim=-1, eps=1e-8)
        return torch.stack((first, second), dim=1)

    def forward(self, taps):
        if len(taps) != self.n_taps:
            raise ValueError(f"expected {self.n_taps} taps, got {len(taps)}")
        normalized = [norm(tap) for norm, tap in zip(self.tap_norms, taps)]
        q = self.effective_depth_basis()
        modes = []
        for group, start in enumerate(range(0, self.n_taps, 3)):
            first, middle, last = normalized[start:start + 3]
            for row in range(2):
                modes.append(q[group, row, 0] * first + q[group, row, 1] * middle
                             + q[group, row, 2] * last)
        return self.fuse(torch.cat(modes, dim=-1))


def build_group_learnable_depthrank2_bottleneck(n_taps=12, code_dim=64,
                                                embed_dim=768, fuse_hidden=128,
                                                use_tap_norm=True,
                                                basis_init="mean_trend",
                                                basis_seed=None):
    return GroupLearnableDepthRank2OCBE(
        embed_dim=embed_dim, n_taps=n_taps, code_dim=code_dim,
        fuse_hidden=fuse_hidden, use_tap_norm=use_tap_norm,
        basis_init=basis_init, basis_seed=basis_seed,
    )
