"""Depth-group gated tensor fusion OCBE.

The input keeps the existing shared learnable rank-2 depth basis.  Instead of
flattening all four groups into one dense projection, each three-tap group is
encoded separately, gated multiplicatively, and mixed along the ordered depth
groups with a residual path graph.  The output contract remains ``(B, N, 64)``.
"""

from math import sqrt

import torch
import torch.nn as nn

from .bottleneck_depthrank_learnable import LearnableDepthRank2OCBE


class DepthGroupGatedTensorOCBE(LearnableDepthRank2OCBE):
    """Rank-2 depth compression followed by gated, ordered group fusion."""

    def __init__(self, embed_dim=768, n_taps=12, code_dim=64,
                 group_hidden=64, use_tap_norm=True,
                 basis_init="random_orthonormal", basis_seed=42):
        super().__init__(
            embed_dim=embed_dim, n_taps=n_taps, code_dim=code_dim,
            fuse_hidden=128, use_tap_norm=use_tap_norm,
            basis_init=basis_init, basis_seed=basis_seed,
        )
        self.group_hidden = group_hidden
        self.n_groups = n_taps // 3
        group_input = 2 * embed_dim

        # Shared group encoder and gate.  Sharing is deliberate: the ordered
        # group axis is mixed separately, rather than giving each group a full
        # dense 1536 -> hidden matrix.
        self.value_proj = nn.Linear(group_input, group_hidden)
        self.gate_proj = nn.Linear(group_input, group_hidden)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

        # A fixed path graph expresses teacher-depth adjacency.  The two
        # channel-wise residual coefficients start at zero, so the new paths
        # are introduced gradually and cannot destroy the original signal at
        # initialization.
        path = torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=torch.float32)
        self.register_buffer("depth_path", path, persistent=True)
        self.depth_alpha1 = nn.Parameter(torch.zeros(group_hidden))
        self.depth_alpha2 = nn.Parameter(torch.zeros(group_hidden))

        self.out_proj = nn.Linear(self.n_groups * group_hidden, code_dim)

        # Remove the parent dense 6144 -> 128 -> code module.  The forward
        # implementation below does not use it and therefore it must not stay
        # in the optimizer or parameter count.
        self.fuse = nn.Identity()

    def forward(self, taps):
        if len(taps) != self.n_taps:
            raise ValueError(f"expected {self.n_taps} taps, got {len(taps)}")
        normalized = [norm(tap) for norm, tap in zip(self.tap_norms, taps)]
        q = self.effective_depth_basis()
        prepared = self._inference_group_inputs(normalized, q)
        groups = []
        for start in range(0, self.n_taps, 3):
            if prepared is None:
                first, middle, last = normalized[start:start + 3]
                mode0 = q[0, 0] * first + q[0, 1] * middle + q[0, 2] * last
                mode1 = q[1, 0] * first + q[1, 1] * middle + q[1, 2] * last
                group = torch.cat((mode0, mode1), dim=-1)
            else:
                group = prepared[start // 3]
            value = torch.nn.functional.gelu(self.value_proj(group))
            gate = 1.0 + torch.tanh(self.gate_proj(group))
            groups.append(value * gate)

        # (B, N, G, d), then propagate only along the ordered teacher-depth
        # group axis.  No patch-to-patch operation is introduced here.
        hidden = torch.stack(groups, dim=2)
        one_hop = torch.einsum("ij,bnjd->bnid", self.depth_path, hidden)
        two_hop = torch.einsum("ij,bnjd->bnid", self.depth_path @ self.depth_path, hidden)
        hidden = hidden + one_hop * self.depth_alpha1 + two_hop * self.depth_alpha2
        return self.out_proj(hidden.reshape(hidden.shape[0], hidden.shape[1], -1))


def build_depth_group_gated_tensor_bottleneck(
        n_taps=12, code_dim=64, embed_dim=768, group_hidden=64,
        use_tap_norm=True, basis_init="random_orthonormal", basis_seed=42):
    return DepthGroupGatedTensorOCBE(
        embed_dim=embed_dim, n_taps=n_taps, code_dim=code_dim,
        group_hidden=group_hidden, use_tap_norm=use_tap_norm,
        basis_init=basis_init, basis_seed=basis_seed,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_depth_group_gated_tensor_bottleneck()
    x = [torch.randn(2, 256, 768) for _ in range(12)]
    y = model(x)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({"output": tuple(y.shape), "trainable": trainable})
