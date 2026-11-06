"""Four-stage student with an explicit global/local block topology.

The internal width can differ from the teacher width. Reconstruction heads project each
student stage back to the teacher width so the distillation target remains unchanged.
The block order is explicit and may be changed for controlled global/local ablations.
"""
import math

import torch
import torch.nn as nn


class GELUMLP(nn.Module):
    """Standard ratio-4 GELU MLP used by the plain transformer block."""

    def __init__(self, dim, hidden_ratio=4.0):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.mlp(x)


def build_ffn(dim, ffn_type='gate', gate_ratio=2.0):
    if ffn_type == 'gate':
        return GatedFFN(dim, gate_ratio)
    if ffn_type == 'gelumlp':
        return GELUMLP(dim, 4.0)
    raise ValueError(f"unknown ffn_type={ffn_type!r}; expected 'gate' or 'gelumlp'")


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
    """Global patch mixer followed by the configured channel mixer."""

    def __init__(self, dim=768, heads=8, ffn_ratio=2.0, ffn_type='gate'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = build_ffn(dim, ffn_type, ffn_ratio)

    def forward(self, x):
        z = self.norm1(x)
        z, _ = self.attn(z, z, z, need_weights=False)
        x = x + z
        return x + self.ffn(self.norm2(x))


class LocalMixerBlock(nn.Module):
    """2-D local patch mixer followed by the configured channel mixer.

    The token sequence is temporarily viewed as a square patch grid. There is no token
    permutation or pooling, so each output remains aligned with its original patch.
    """

    def __init__(self, dim=768, ffn_ratio=2.0, kernel_size=3, ffn_type='gate'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2,
                                groups=dim, bias=True)
        self.pointwise = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = build_ffn(dim, ffn_type, ffn_ratio)

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
    """Four configurable global/local blocks and 12 reverse-aligned heads."""

    def __init__(self, dim=768, code_dim=64, heads=8, ffn_ratio=2.0,
                 teacher_dim=768, block_order='GLLL', ffn_type='gate'):
        super().__init__()
        if len(block_order) != 4 or any(ch not in 'GL' for ch in block_order):
            raise ValueError(f"block_order must be four G/L characters, got {block_order!r}")
        if ffn_type not in {'gate', 'gelumlp'}:
            raise ValueError(f"unknown ffn_type={ffn_type!r}")
        self.block_order = block_order
        self.ffn_type = ffn_type
        self.proj_in = nn.Linear(code_dim, dim)
        blocks = []
        for kind in block_order:
            if kind == 'G':
                blocks.append(GlobalMixerBlock(dim, heads, ffn_ratio, ffn_type))
            else:
                blocks.append(LocalMixerBlock(dim, ffn_ratio, ffn_type=ffn_type))
        self.blocks = nn.ModuleList(blocks)
        self.heads = nn.ModuleList(nn.Linear(dim, teacher_dim) for _ in range(12))

    def forward(self, code):
        x = self.proj_in(code)
        out = [None] * 12
        for stage, block in enumerate(self.blocks):
            x = block(x)
            group = 3 - stage
            for layer in range(group * 3, group * 3 + 3):
                out[layer] = self.heads[layer](x)
        return out


def build_structural_student(dim=768, code_dim=64, heads=8, ffn_ratio=2.0,
                             teacher_dim=768, block_order='GLLL', ffn_type='gate'):
    return FourBlockGlobalLocalDecoder(dim=dim, code_dim=code_dim, heads=heads,
                                       ffn_ratio=ffn_ratio, teacher_dim=teacher_dim,
                                       block_order=block_order, ffn_type=ffn_type)
