"""Separable group fusion bottlenecks for the Rank2 OCBE path.

The teacher triplet projection remains the shared learnable Rank2 Q.  SGF first
compresses each 2x768 group with one shared channel projection and only then
mixes the four teacher groups in the small latent axis.  This avoids running a
nonlinear group mixer over the 2x768 tensors used by DGT/CrossDepth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck_depthrank_learnable import LearnableDepthRank2OCBE


class DepthwiseGroupMix(nn.Module):
    """Depthwise 1-D convolution along the ordered four-group axis.

    Each latent channel has its own three-point group kernel.  The center tap
    starts at one and the two neighbours at zero, so the module is initially an
    identity map and does not disturb the shared-projection control.
    """

    def __init__(self, width: int):
        super().__init__()
        self.conv = nn.Conv1d(width, width, kernel_size=3, padding=1,
                              groups=width, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, 0, 1] = 1.0

    def forward(self, x):
        # x: [B, P, groups=4, width]
        b, p, groups, width = x.shape
        y = x.reshape(b * p, groups, width).transpose(1, 2)
        y = self.conv(y)
        return y.transpose(1, 2).reshape(b, p, groups, width)


class SeparableGroupFusionBottleneck(LearnableDepthRank2OCBE):
    """Shared local channel compression followed by optional group mixing."""

    def __init__(self, embed_dim=768, n_taps=12, code_dim=64,
                 hidden=128, latent_width=48, use_group_mix=False,
                 use_tap_norm=True, basis_init="random_orthonormal",
                 basis_seed=42):
        # The parent creates tap norms and the Rank2 parameter/cache contract.
        super().__init__(embed_dim=embed_dim, n_taps=n_taps,
                         code_dim=code_dim, fuse_hidden=128,
                         use_tap_norm=use_tap_norm,
                         basis_init=basis_init, basis_seed=basis_seed)
        if n_taps != 12:
            raise ValueError("SGF currently expects twelve taps")
        if latent_width <= 0:
            raise ValueError("latent_width must be positive")
        self.latent_width = latent_width
        self.use_group_mix = use_group_mix
        self.local_proj = nn.Linear(2 * embed_dim, latent_width)
        self.local_act = nn.GELU()
        self.group_mix = (DepthwiseGroupMix(latent_width)
                          if use_group_mix else nn.Identity())
        self.fuse = nn.Sequential(
            nn.Linear(4 * latent_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, code_dim),
        )

    def _project_groups(self, normalized, q):
        """Return [B,P,4,1536] without Python-side per-group matmuls."""
        b, p, c = normalized[0].shape
        stacked = torch.stack(normalized, dim=2).reshape(b, p, 4, 3, c)
        # q is [2,3], and the result is [B,P,4,2,768].
        projected = torch.einsum("ds,bpgsc->bpgdc", q, stacked)
        return projected.reshape(b, p, 4, 2 * c)

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
        latent = self.local_act(self.local_proj(group))
        latent = self.group_mix(latent)
        return self.fuse(latent.reshape(latent.shape[0], latent.shape[1], -1))


def build_sgf_bottleneck(n_taps=12, code_dim=64, embed_dim=768,
                         hidden=128, latent_width=48,
                         use_group_mix=False, use_tap_norm=True,
                         basis_init="random_orthonormal", basis_seed=42):
    return SeparableGroupFusionBottleneck(
        embed_dim=embed_dim, n_taps=n_taps, code_dim=code_dim,
        hidden=hidden, latent_width=latent_width,
        use_group_mix=use_group_mix, use_tap_norm=use_tap_norm,
        basis_init=basis_init, basis_seed=basis_seed)
