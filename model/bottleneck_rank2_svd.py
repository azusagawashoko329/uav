"""Function-preserving low-rank factorization of the Rank2 OCBE head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck_depthrank_learnable import LearnableDepthRank2OCBE


class FactorizedLinear(nn.Module):
    """Two linear maps initialized from a truncated SVD of one dense map."""

    def __init__(self, weight, bias, rank):
        super().__init__()
        if rank <= 0 or rank > min(weight.shape):
            raise ValueError(f"rank {rank} is invalid for weight {tuple(weight.shape)}")
        u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
        self.down = nn.Parameter(vh[:rank].contiguous())
        self.up = nn.Parameter((u[:, :rank] * s[:rank].unsqueeze(0)).contiguous())
        self.bias = nn.Parameter(bias.detach().float().clone())

    def forward(self, x):
        return F.linear(F.linear(x, self.down), self.up, self.bias)


class SVDRank2OCBE(LearnableDepthRank2OCBE):
    """Rank2 OCBE whose 6144->128 head is initialized by baseline SVD."""

    def __init__(self, rank=64, baseline_state=None, embed_dim=768,
                 n_taps=12, code_dim=64, hidden=128, use_tap_norm=True,
                 basis_init="random_orthonormal", basis_seed=42):
        super().__init__(embed_dim=embed_dim, n_taps=n_taps,
                         code_dim=code_dim, fuse_hidden=hidden,
                         use_tap_norm=use_tap_norm,
                         basis_init=basis_init, basis_seed=basis_seed)
        if hidden != 128:
            raise ValueError("SVDRank2OCBE currently expects hidden=128")
        if baseline_state is None:
            raise ValueError("baseline_state is required for SVD initialization")
        weight = baseline_state["fuse.0.weight"]
        bias = baseline_state["fuse.0.bias"]
        output = nn.Linear(hidden, code_dim)
        output.weight.data.copy_(baseline_state["fuse.2.weight"])
        output.bias.data.copy_(baseline_state["fuse.2.bias"])
        self.fuse = nn.Sequential(
            FactorizedLinear(weight, bias, rank),
            nn.GELU(),
            output,
        )
        # Preserve the trained Rank2 baseline's Q, tap LayerNorms, and any
        # other non-head state.  Only the original dense fusion head is
        # replaced by its truncated-SVD factorization.
        preserved = {k: v for k, v in baseline_state.items()
                     if not k.startswith("fuse.")}
        self.load_state_dict(preserved, strict=False)


def build_svd_rank2_bottleneck(baseline_state, rank=64, n_taps=12,
                               code_dim=64, embed_dim=768, hidden=128,
                               use_tap_norm=True,
                               basis_init="random_orthonormal",
                               basis_seed=42):
    return SVDRank2OCBE(
        rank=rank, baseline_state=baseline_state, embed_dim=embed_dim,
        n_taps=n_taps, code_dim=code_dim, hidden=hidden,
        use_tap_norm=use_tap_norm, basis_init=basis_init,
        basis_seed=basis_seed)
