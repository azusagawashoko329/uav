"""OCBE extensions that preserve the Rank2 Learnable-Q bottleneck.

The side-channel variants keep the original 6144-dimensional Rank2 input and
add zero-initialized features before the existing 128-dimensional hidden
layer.  They therefore start with the same forward function as the baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck_depthrank_learnable import LearnableDepthRank2OCBE


class Rank2SideFusionBottleneck(LearnableDepthRank2OCBE):
    """Base class for a zero-initialized side feature before the OCBE GELU."""

    def __init__(self, embed_dim=768, n_taps=12, code_dim=64,
                 hidden=128, side_dim=0, use_tap_norm=True,
                 basis_init="random_orthonormal", basis_seed=42):
        super().__init__(embed_dim=embed_dim, n_taps=n_taps, code_dim=code_dim,
                         fuse_hidden=hidden, use_tap_norm=use_tap_norm,
                         basis_init=basis_init, basis_seed=basis_seed)
        if n_taps != 12:
            raise ValueError("Rank2 side fusion expects twelve taps")
        if side_dim <= 0:
            raise ValueError("side_dim must be positive")
        self.side_dim = side_dim
        # Replace the parent's dense Sequential rather than retaining a second
        # 6144->128->64 head in the parameter set.
        del self.fuse
        self.input_proj = nn.Linear(4 * 2 * embed_dim, hidden)
        self.side_proj = nn.Linear(side_dim, hidden, bias=False)
        nn.init.zeros_(self.side_proj.weight)
        self.hidden_act = nn.GELU()
        self.output_proj = nn.Linear(hidden, code_dim)

    def _project_groups(self, normalized, q):
        b, p, c = normalized[0].shape
        stacked = torch.stack(normalized, dim=2).reshape(b, p, 4, 3, c)
        projected = torch.einsum("ds,bpgsc->bpgdc", q, stacked)
        return projected.reshape(b, p, 4, 2 * c)

    def _side_features(self, group, q, normalized):
        raise NotImplementedError

    def _fuse(self, group, side):
        x = group.reshape(group.shape[0], group.shape[1], -1)
        hidden = self.input_proj(x) + self.side_proj(side)
        return self.output_proj(self.hidden_act(hidden))

    def forward(self, taps):
        if len(taps) != self.n_taps:
            raise ValueError(f"expected {self.n_taps} taps, got {len(taps)}")
        normalized = [norm(tap) for norm, tap in zip(self.tap_norms, taps)]
        q = self.effective_depth_basis()
        if (self._inference_depth_projector is not None
                and not self.training and not torch.is_grad_enabled()):
            from .depthrank_projection import project_depth_rank2
            b, p, _ = normalized[0].shape
            group = self._inference_depth_projector(normalized, q).view(
                b, p, 4, 2 * self.embed_dim)
        else:
            group = self._project_groups(normalized, q)
        side = self._side_features(group, q, normalized)
        return self._fuse(group, side)


class Rank2GramSideBottleneck(Rank2SideFusionBottleneck):
    """Rank2 plus four-group energy and cosine consistency statistics."""

    def __init__(self, **kwargs):
        super().__init__(side_dim=10, **kwargs)

    def _side_features(self, group, q, normalized):
        # Group vectors retain all 2x768 Rank2 information.  Only the side
        # statistic is normalized, so the main reconstruction path is intact.
        norm = torch.linalg.vector_norm(group, dim=-1).clamp_min(1e-6)
        rms = norm / (group.shape[-1] ** 0.5)
        unit = group / norm.unsqueeze(-1)
        gram = torch.matmul(unit, unit.transpose(-1, -2))
        upper = torch.triu_indices(4, 4, offset=1, device=group.device)
        cosine = gram[..., upper[0], upper[1]]
        return torch.cat((torch.log1p(rms), cosine), dim=-1)


class Rank2ResidualEnergyBottleneck(Rank2SideFusionBottleneck):
    """Rank2 plus energy in the one discarded orthogonal depth direction."""

    def __init__(self, **kwargs):
        super().__init__(side_dim=4, **kwargs)

    def _side_features(self, group, q, normalized):
        b, p, c = normalized[0].shape
        stacked = torch.stack(normalized, dim=2).reshape(b, p, 4, 3, c)
        complement = F.normalize(torch.linalg.cross(q[0], q[1]), dim=0, eps=1e-8)
        residual = torch.einsum("s,bpgsc->bpgc", complement, stacked)
        energy = residual.square().mean(dim=-1)
        return torch.log1p(energy)


def build_rank2_gram_side_bottleneck(n_taps=12, code_dim=64, embed_dim=768,
                                     hidden=128, use_tap_norm=True,
                                     basis_init="random_orthonormal", basis_seed=42):
    return Rank2GramSideBottleneck(
        embed_dim=embed_dim, n_taps=n_taps, code_dim=code_dim,
        hidden=hidden, use_tap_norm=use_tap_norm,
        basis_init=basis_init, basis_seed=basis_seed)


def build_rank2_residual_energy_bottleneck(
        n_taps=12, code_dim=64, embed_dim=768, hidden=128,
        use_tap_norm=True, basis_init="random_orthonormal", basis_seed=42):
    return Rank2ResidualEnergyBottleneck(
        embed_dim=embed_dim, n_taps=n_taps, code_dim=code_dim,
        hidden=hidden, use_tap_norm=use_tap_norm,
        basis_init=basis_init, basis_seed=basis_seed)
