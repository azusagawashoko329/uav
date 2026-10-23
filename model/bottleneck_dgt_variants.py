"""Ablations for the depth-group fusion stage of OCBE.

All variants keep the shared learnable random-orthonormal Rank-2 Q and the
per-group value/gate encoder.  Only the interaction between the four ordered
teacher-depth groups changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck_depthrank_learnable import LearnableDepthRank2OCBE


class _DepthGroupBase(LearnableDepthRank2OCBE):
    def __init__(self, embed_dim=768, n_taps=12, code_dim=64,
                 group_hidden=64, use_tap_norm=True, basis_seed=42):
        super().__init__(embed_dim=embed_dim, n_taps=n_taps, code_dim=code_dim,
                         fuse_hidden=128, use_tap_norm=use_tap_norm,
                         basis_init="random_orthonormal", basis_seed=basis_seed)
        self.group_hidden = group_hidden
        self.n_groups = n_taps // 3
        group_input = 2 * embed_dim
        self.value_proj = nn.Linear(group_input, group_hidden)
        self.gate_proj = nn.Linear(group_input, group_hidden)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        self.fuse = nn.Identity()
        self._diagnostics = {}

    def encode_groups(self, taps):
        if len(taps) != self.n_taps:
            raise ValueError(f"expected {self.n_taps} taps, got {len(taps)}")
        normalized = [norm(tap) for norm, tap in zip(self.tap_norms, taps)]
        q = self.effective_depth_basis()
        groups = []
        gate_values = []
        for start in range(0, self.n_taps, 3):
            first, middle, last = normalized[start:start + 3]
            mode0 = q[0, 0] * first + q[0, 1] * middle + q[0, 2] * last
            mode1 = q[1, 0] * first + q[1, 1] * middle + q[1, 2] * last
            group = torch.cat((mode0, mode1), dim=-1)
            value = F.gelu(self.value_proj(group))
            raw_gate = self.gate_proj(group)
            gate = 1.0 + torch.tanh(raw_gate)
            groups.append(value * gate)
            gate_values.append(gate)
        hidden = torch.stack(groups, dim=2)  # B,N,G,D
        with torch.no_grad():
            self._diagnostics = {
                "gate_mean": float(torch.stack(gate_values).mean().detach().cpu()),
                "gate_std": float(torch.stack(gate_values).std().detach().cpu()),
                "hidden_rms": float(hidden.detach().pow(2).mean().sqrt().cpu()),
            }
        return hidden

    def diagnostics(self):
        return dict(self._diagnostics)


class DepthGroupNoGraphOCBE(_DepthGroupBase):
    """Group gate only; no cross-group propagation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out_proj = nn.Linear(self.n_groups * self.group_hidden, self.code_dim)

    def forward(self, taps):
        hidden = self.encode_groups(taps)
        return self.out_proj(hidden.reshape(hidden.shape[0], hidden.shape[1], -1))


class DepthGroupCrossAttnPoolOCBE(_DepthGroupBase):
    """Dynamic attention over four depth groups followed by two query pools."""

    def __init__(self, attention_dim=16, n_queries=2, **kwargs):
        super().__init__(**kwargs)
        d = self.group_hidden
        self.attention_dim = attention_dim
        self.n_queries = n_queries
        self.depth_pos = nn.Parameter(torch.zeros(self.n_groups, d))
        self.q_proj = nn.Linear(d, attention_dim)
        self.k_proj = nn.Linear(d, attention_dim)
        self.v_proj = nn.Linear(d, d)
        self.attn_out = nn.Linear(d, d)
        # The residual attention branch starts as zero, so the baseline group
        # representation is preserved while Q/K/V learn from a stable path.
        nn.init.zeros_(self.attn_out.weight)
        nn.init.zeros_(self.attn_out.bias)
        self.pool_queries = nn.Parameter(torch.zeros(n_queries, d))
        nn.init.normal_(self.pool_queries, mean=0.0, std=0.02)
        self.pool_out = nn.Linear(n_queries * d, self.code_dim)

    def forward(self, taps):
        hidden = self.encode_groups(taps)
        x = hidden + self.depth_pos.view(1, 1, self.n_groups, -1)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        logits = torch.einsum("bnid,bnjd->bnij", q, k) / (self.attention_dim ** 0.5)
        attn = logits.softmax(dim=-1)
        mixed = hidden + self.attn_out(torch.einsum("bnij,bnjd->bnid", attn, v))

        query_logits = torch.einsum("md,bngd->bnmg", self.pool_queries, mixed)
        pool_weights = query_logits.softmax(dim=-1)
        pooled = torch.einsum("bnmg,bngd->bnmd", pool_weights, mixed)
        with torch.no_grad():
            entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(-1).mean()
            pool_entropy = -(pool_weights.clamp_min(1e-8) * pool_weights.clamp_min(1e-8).log()).sum(-1).mean()
            self._diagnostics.update({
                "attention_entropy": float(entropy.detach().cpu()),
                "pool_entropy": float(pool_entropy.detach().cpu()),
                "attention_identity_mass": float(torch.diagonal(attn, dim1=-2, dim2=-1).mean().detach().cpu()),
            })
        return self.pool_out(pooled.reshape(pooled.shape[0], pooled.shape[1], -1))


class DepthGroupBilinearOCBE(_DepthGroupBase):
    """Low-rank multiplicative interactions for adjacent depth groups."""

    def __init__(self, interaction_rank=16, **kwargs):
        super().__init__(**kwargs)
        d = self.group_hidden
        r = interaction_rank
        self.interaction_rank = r
        self.pair_a = nn.Linear(d, r)
        self.pair_b = nn.Linear(d, r)
        self.pair_up = nn.Linear(r, d)
        # Zero output gives a stable residual start while retaining gradients
        # for the interaction output projection.
        nn.init.zeros_(self.pair_up.weight)
        nn.init.zeros_(self.pair_up.bias)
        self.out_proj = nn.Linear(self.n_groups * d, self.code_dim)

    def forward(self, taps):
        hidden = self.encode_groups(taps)
        deltas = []
        for g in range(self.n_groups - 1):
            left = F.gelu(self.pair_a(hidden[:, :, g, :]))
            right = F.gelu(self.pair_b(hidden[:, :, g + 1, :]))
            deltas.append(self.pair_up(left * right))
        mixed = hidden.clone()
        for g, delta in enumerate(deltas):
            mixed[:, :, g, :] = mixed[:, :, g, :] + delta
            mixed[:, :, g + 1, :] = mixed[:, :, g + 1, :] + delta
        with torch.no_grad():
            self._diagnostics["bilinear_delta_rms"] = float(
                torch.stack(deltas, dim=2).detach().pow(2).mean().sqrt().cpu())
        return self.out_proj(mixed.reshape(mixed.shape[0], mixed.shape[1], -1))


def build_dgt_variant(variant, n_taps=12, code_dim=64, embed_dim=768,
                      group_hidden=64, basis_seed=42):
    common = dict(n_taps=n_taps, code_dim=code_dim, embed_dim=embed_dim,
                  group_hidden=group_hidden, use_tap_norm=True,
                  basis_seed=basis_seed)
    if variant == "no_graph":
        return DepthGroupNoGraphOCBE(**common)
    if variant == "crossattn_pool":
        return DepthGroupCrossAttnPoolOCBE(**common)
    if variant == "bilinear":
        return DepthGroupBilinearOCBE(**common)
    raise ValueError(f"unknown DGT variant: {variant}")
