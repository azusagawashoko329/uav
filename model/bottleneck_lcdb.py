"""Low-rank cross-depth bottleneck for the four Rank2 teacher groups.

Unlike DGT, group interaction happens while each group still has 1536
features. A fixed mean/trend basis reduces the four group axis to two modes,
then one shared 3072->hidden->64 bottleneck mixes all groups before the final
code. The fixed basis avoids adding a new loss or another per-forward
orthogonalization path.
"""
import math

import torch
import torch.nn as nn

from .bottleneck_depthrank_learnable import LearnableDepthRank2OCBE


class LowRankCrossDepthBottleneck(LearnableDepthRank2OCBE):
    def __init__(self, embed_dim=768, n_taps=12, code_dim=64,
                 hidden=128, use_tap_norm=True,
                 basis_init="random_orthonormal", basis_seed=42):
        super().__init__(embed_dim=embed_dim, n_taps=n_taps, code_dim=code_dim,
                         fuse_hidden=128, use_tap_norm=use_tap_norm,
                         basis_init=basis_init, basis_seed=basis_seed)
        if n_taps != 12:
            raise ValueError("LCDB currently expects four groups of three taps")
        mean = torch.full((4,), 0.5, dtype=torch.float32)
        trend = torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=torch.float32)
        trend = trend / math.sqrt(float((trend * trend).sum()))
        self.register_buffer("depth_basis", torch.stack((mean, trend)), persistent=True)
        self.fuse = nn.Sequential(
            nn.Linear(2 * 2 * embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, code_dim),
        )

    def _group_tensor(self, normalized, q):
        groups = []
        for start in range(0, self.n_taps, 3):
            first, middle, last = normalized[start:start + 3]
            mode0 = q[0, 0] * first + q[0, 1] * middle + q[0, 2] * last
            mode1 = q[1, 0] * first + q[1, 1] * middle + q[1, 2] * last
            groups.append(torch.cat((mode0, mode1), dim=-1))
        return torch.stack(groups, dim=2)

    def forward(self, taps):
        if len(taps) != self.n_taps:
            raise ValueError(f"expected {self.n_taps} taps, got {len(taps)}")
        normalized = [norm(tap) for norm, tap in zip(self.tap_norms, taps)]
        q = self.effective_depth_basis()
        if (self._inference_depth_projector is not None
                and not self.training and not torch.is_grad_enabled()):
            from .depthrank_projection import project_depth_rank2
            group = project_depth_rank2(normalized, q).view(
                normalized[0].shape[0], normalized[0].shape[1], 4, 2 * self.embed_dim)
        else:
            group = self._group_tensor(normalized, q)
        # [B,P,2,1536]; the two modes mix all four teacher groups before MLP.
        modes = torch.einsum("rg,bpgd->bprd", self.depth_basis, group)
        return self.fuse(modes.reshape(modes.shape[0], modes.shape[1], -1))


def build_lcdb_bottleneck(n_taps=12, code_dim=64, embed_dim=768,
                          hidden=128, use_tap_norm=True,
                          basis_init="random_orthonormal", basis_seed=42):
    return LowRankCrossDepthBottleneck(
        embed_dim=embed_dim, n_taps=n_taps, code_dim=code_dim,
        hidden=hidden, use_tap_norm=use_tap_norm,
        basis_init=basis_init, basis_seed=basis_seed)
