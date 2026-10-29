"""Function-preserving and end-to-end low-rank Rank2 OCBE heads."""

import math

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


class RandomFactorizedLinear(nn.Module):
    """Rank-factorized linear layer initialized without a dense checkpoint."""

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        if rank <= 0 or rank > min(in_features, out_features):
            raise ValueError(f"rank {rank} is invalid for {in_features}x{out_features}")
        self.down = nn.Parameter(torch.empty(rank, in_features))
        self.up = nn.Parameter(torch.empty(out_features, rank))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.normal_(self.down, mean=0.0, std=1.0 / math.sqrt(in_features))
        nn.init.normal_(self.up, mean=0.0, std=1.0 / math.sqrt(rank))
        nn.init.zeros_(self.bias)

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


class EndToEndFactorizedRank2OCBE(LearnableDepthRank2OCBE):
    """Rank2 OCBE with a randomly initialized factorized fusion head."""

    def __init__(self, rank=64, embed_dim=768, n_taps=12, code_dim=64,
                 hidden=128, use_tap_norm=True,
                 basis_init="random_orthonormal", basis_seed=42):
        super().__init__(embed_dim=embed_dim, n_taps=n_taps,
                         code_dim=code_dim, fuse_hidden=hidden,
                         use_tap_norm=use_tap_norm,
                         basis_init=basis_init, basis_seed=basis_seed)
        if hidden != 128:
            raise ValueError("EndToEndFactorizedRank2OCBE expects hidden=128")
        self.fuse = nn.Sequential(
            RandomFactorizedLinear(4 * 2 * embed_dim, hidden, rank),
            nn.GELU(),
            nn.Linear(hidden, code_dim),
        )


def build_end_to_end_factorized_rank2_bottleneck(
        rank=64, n_taps=12, code_dim=64, embed_dim=768, hidden=128,
        use_tap_norm=True, basis_init="random_orthonormal", basis_seed=42):
    return EndToEndFactorizedRank2OCBE(
        rank=rank, embed_dim=embed_dim, n_taps=n_taps,
        code_dim=code_dim, hidden=hidden, use_tap_norm=use_tap_norm,
        basis_init=basis_init, basis_seed=basis_seed)
