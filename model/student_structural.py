"""Four-stage student with an explicit global/local block topology.

The decoder remains 768-dimensional so the experiment isolates block topology rather than
changing the student input/output width. Its first block is global because reverse alignment
maps the first student stage to the deepest teacher group; later blocks use 2-D local mixing.
"""
import math

import torch
import torch.nn as nn


class GatedFFN(nn.Module):
    """Gated channel mixer with a 2x hidden ratio.

    A standard ratio-4 GELU MLP has two dense matrices with 8*d^2 parameters. This gated
    mixer uses a 2*d hidden size and a split input projection (6*d^2 total), while changing
    the operator from a plain pointwise GELU to multiplicative channel gating.
    """

    def __init__(self, dim, hidden_ratio=2.0):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.in_proj = nn.Linear(dim, 2 * hidden)
        self.out_proj = nn.Linear(hidden, dim)

    def forward(self, x):
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(torch.nn.functional.silu(gate) * value)


class GlobalMixerBlock(nn.Module):
    """Global patch mixer followed by gated channel mixing."""

    def __init__(self, dim=768, heads=8, ffn_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GatedFFN(dim, ffn_ratio)

    def forward(self, x):
        z = self.norm1(x)
        z, _ = self.attn(z, z, z, need_weights=False)
        x = x + z
        return x + self.ffn(self.norm2(x))


class LocalMixerBlock(nn.Module):
    """2-D local patch mixer followed by the same gated channel mixer.

    The token sequence is temporarily viewed as a square patch grid. There is no token
    permutation or pooling, so each output remains aligned with its original patch.
    """

    def __init__(self, dim=768, ffn_ratio=2.0, kernel_size=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2,
                                groups=dim, bias=True)
        self.pointwise = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GatedFFN(dim, ffn_ratio)

    def forward(self, x):
        b, patches, dim = x.shape
        side = math.isqrt(patches)
        if side * side != patches:
            raise ValueError(f"LocalMixerBlock requires a square patch grid, got {patches}")
        z = self.norm1(x).transpose(1, 2).reshape(b, dim, side, side)
        z = self.dwconv(z).flatten(2).transpose(1, 2)
        x = x + self.pointwise(z)
        return x + self.ffn(self.norm2(x))


class FourBlockGlobalLocalDecoder(nn.Module):
    """One global block followed by three local blocks and 12 reverse-aligned heads."""

    def __init__(self, dim=768, code_dim=64, heads=8, ffn_ratio=2.0):
        super().__init__()
        self.proj_in = nn.Linear(code_dim, dim)
        self.blocks = nn.ModuleList([
            GlobalMixerBlock(dim, heads, ffn_ratio),
            LocalMixerBlock(dim, ffn_ratio),
            LocalMixerBlock(dim, ffn_ratio),
            LocalMixerBlock(dim, ffn_ratio),
        ])
        self.heads = nn.ModuleList(nn.Linear(dim, dim) for _ in range(12))

    def forward(self, code):
        x = self.proj_in(code)
        out = [None] * 12
        for stage, block in enumerate(self.blocks):
            x = block(x)
            group = 3 - stage
            for layer in range(group * 3, group * 3 + 3):
                out[layer] = self.heads[layer](x)
        return out


def build_structural_student(dim=768, code_dim=64, heads=8, ffn_ratio=2.0):
    return FourBlockGlobalLocalDecoder(dim=dim, code_dim=code_dim, heads=heads,
                                       ffn_ratio=ffn_ratio)
