"""Group-context fusion for the Rank2 OCBE bottleneck.

The shared Rank2 Q still compresses each adjacent teacher triplet from three depth
views to two.  The right-hand fusion is structured as a shared local channel
projection, a four-group global context interaction, and a small output MLP.  The
context branch is zero-gated at initialization so the local path is stable at the
start of training; no auxiliary loss is required.
"""

import torch
import torch.nn as nn

from .bottleneck_depthrank_learnable import LearnableDepthRank2OCBE


class GroupContextFusionBottleneck(LearnableDepthRank2OCBE):
    """Rank2 OCBE with explicit low-cost cross-group context interaction."""

    def __init__(self, embed_dim=768, n_taps=12, code_dim=64,
                 hidden=128, latent_width=64, use_tap_norm=True,
                 basis_init="random_orthonormal", basis_seed=42):
        super().__init__(embed_dim=embed_dim, n_taps=n_taps,
                         code_dim=code_dim, fuse_hidden=128,
                         use_tap_norm=use_tap_norm,
                         basis_init=basis_init, basis_seed=basis_seed)
        if n_taps != 12:
            raise ValueError("GroupContextFusionBottleneck expects twelve taps")
        if latent_width <= 0:
            raise ValueError("latent_width must be positive")
        self.latent_width = latent_width
        self.local_proj = nn.Linear(2 * embed_dim, latent_width)
        self.local_act = nn.GELU()
        # Per-latent-channel residual gate.  Zero initialization keeps the
        # initial mapping on the local branch while allowing context gradients.
        self.context_scale = nn.Parameter(torch.zeros(1, 1, 1, latent_width))
        self.fuse = nn.Sequential(
            nn.Linear(4 * latent_width, hidden),
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
        # [B, patches, groups=4, 2*embed_dim]
        return torch.stack(groups, dim=2)

    def _context_fuse(self, group):
        latent = self.local_act(self.local_proj(group))
        context = torch.tanh(latent.mean(dim=2, keepdim=True))
        latent = latent + self.context_scale * latent * context
        return self.fuse(latent.reshape(latent.shape[0], latent.shape[1], -1))

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
        return self._context_fuse(group)


def build_group_context_bottleneck(n_taps=12, code_dim=64, embed_dim=768,
                                   hidden=128, latent_width=64,
                                   use_tap_norm=True,
                                   basis_init="random_orthonormal",
                                   basis_seed=42):
    return GroupContextFusionBottleneck(
        n_taps=n_taps, code_dim=code_dim, embed_dim=embed_dim,
        hidden=hidden, latent_width=latent_width,
        use_tap_norm=use_tap_norm, basis_init=basis_init,
        basis_seed=basis_seed)
