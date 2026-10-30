"""Group-block low-rank fusion for the Rank2 OCBE bottleneck.

The shared depth Rank2 projection is unchanged.  The dense first fusion map is
factorized by teacher group: a shared input projection V maps each 1536-D group
to rank r, each group has an independent output map U_g -> hidden, and the four
hidden contributions are summed before GELU.  This preserves cross-group mixing
before the nonlinearity without a context/gate/hop path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck_depthrank_learnable import LearnableDepthRank2OCBE


class GroupBlockLowRankBottleneck(LearnableDepthRank2OCBE):
    """Rank2 OCBE with a group-block low-rank fusion map."""

    def __init__(self, embed_dim=768, n_taps=12, code_dim=64,
                 hidden=128, rank=128, use_tap_norm=True,
                 basis_init="random_orthonormal", basis_seed=42):
        super().__init__(embed_dim=embed_dim, n_taps=n_taps,
                         code_dim=code_dim, fuse_hidden=128,
                         use_tap_norm=use_tap_norm,
                         basis_init=basis_init, basis_seed=basis_seed)
        if n_taps != 12:
            raise ValueError("GroupBlockLowRankBottleneck expects twelve taps")
        if rank <= 0 or rank > 2 * embed_dim:
            raise ValueError(f"rank must be in [1, {2 * embed_dim}], got {rank}")
        self.group_rank = rank
        self.group_count = n_taps // 3
        self.shared_in = nn.Linear(2 * embed_dim, rank, bias=False)
        # U_g has shape [group, rank, hidden]. A single hidden bias is shared
        # after summing group contributions, matching the original dense layer.
        self.group_out = nn.Parameter(torch.empty(self.group_count, rank, hidden))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden))
        nn.init.xavier_uniform_(self.shared_in.weight)
        nn.init.xavier_uniform_(self.group_out)
        self.fuse = nn.Sequential(
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

    def _fuse_groups(self, group):
        # group: [B, patches, 4, 1536]
        b, p, g, _ = group.shape
        local = self.shared_in(group)
        # One batched matmul over the four group-specific U_g maps. This keeps
        # group mixing before GELU without Python loops or per-group kernels.
        local = local.permute(2, 0, 1, 3).reshape(g, b * p, self.group_rank)
        hidden = torch.bmm(local, self.group_out).sum(dim=0)
        hidden = hidden.reshape(b, p, -1) + self.hidden_bias
        return self.fuse(hidden)

    def forward(self, taps):
        if len(taps) != self.n_taps:
            raise ValueError(f"expected {self.n_taps} taps, got {len(taps)}")
        normalized = [norm(tap) for norm, tap in zip(self.tap_norms, taps)]
        q = self.effective_depth_basis()
        if (self._inference_depth_projector is not None
                and not self.training and not torch.is_grad_enabled()):
            from .depthrank_projection import project_depth_rank2
            b, p, c = normalized[0].shape
            projected = self._inference_depth_projector(normalized, q)
            group = projected.reshape(b, p, 4, 2 * c)
        else:
            group = self._group_tensor(normalized, q)
        return self._fuse_groups(group)


def build_group_block_lr_bottleneck(n_taps=12, code_dim=64,
                                    embed_dim=768, hidden=128, rank=128,
                                    use_tap_norm=True,
                                    basis_init="random_orthonormal",
                                    basis_seed=42):
    return GroupBlockLowRankBottleneck(
        n_taps=n_taps, code_dim=code_dim, embed_dim=embed_dim,
        hidden=hidden, rank=rank, use_tap_norm=use_tap_norm,
        basis_init=basis_init, basis_seed=basis_seed)
