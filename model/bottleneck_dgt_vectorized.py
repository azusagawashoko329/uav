"""Vectorized inference form of DGT.

The mathematical DGT path is unchanged. Four group tensors are flattened into
one GEMM for value and one GEMM for gate, instead of launching eight small
Linear calls from a Python loop. Training remains differentiable and uses the
same modules and state_dict keys.
"""
import torch
import torch.nn.functional as F

from .bottleneck_dgt import DepthGroupGatedTensorOCBE


class VectorizedDepthGroupGatedTensorOCBE(DepthGroupGatedTensorOCBE):
    def forward(self, taps):
        if len(taps) != self.n_taps:
            raise ValueError(f"expected {self.n_taps} taps, got {len(taps)}")
        normalized = [norm(tap) for norm, tap in zip(self.tap_norms, taps)]
        q = self.effective_depth_basis()
        if (self._inference_depth_projector is not None
                and not self.training and not torch.is_grad_enabled()):
            from .depthrank_projection import project_depth_rank2
            group = project_depth_rank2(normalized, q).view(
                normalized[0].shape[0], normalized[0].shape[1], 4,
                2 * self.embed_dim)
        else:
            groups = []
            for start in range(0, self.n_taps, 3):
                first, middle, last = normalized[start:start + 3]
                mode0 = q[0, 0] * first + q[0, 1] * middle + q[0, 2] * last
                mode1 = q[1, 0] * first + q[1, 1] * middle + q[1, 2] * last
                groups.append(torch.cat((mode0, mode1), dim=-1))
            group = torch.stack(groups, dim=2)

        batch, patches, groups, width = group.shape
        flat = group.reshape(batch * patches * groups, width)
        value = F.gelu(self.value_proj(flat)).reshape(
            batch, patches, groups, self.group_hidden)
        gate = 1.0 + torch.tanh(self.gate_proj(flat)).reshape(
            batch, patches, groups, self.group_hidden)
        hidden = value * gate
        one_hop = torch.einsum("ij,bnjd->bnid", self.depth_path, hidden)
        two_hop = torch.einsum("ij,bnjd->bnid", self.depth_path @ self.depth_path, hidden)
        hidden = hidden + one_hop * self.depth_alpha1 + two_hop * self.depth_alpha2
        return self.out_proj(hidden.reshape(batch, patches, -1))


def build_vectorized_dgt_bottleneck(n_taps=12, code_dim=64, embed_dim=768,
                                    group_hidden=64, use_tap_norm=True,
                                    basis_init="random_orthonormal", basis_seed=42):
    return VectorizedDepthGroupGatedTensorOCBE(
        embed_dim=embed_dim, n_taps=n_taps, code_dim=code_dim,
        group_hidden=group_hidden, use_tap_norm=use_tap_norm,
        basis_init=basis_init, basis_seed=basis_seed)
