"""Formal UIT HAP-RD-Lite with one learnable Q per depth group.

This module is intentionally separate from the shared-Q UIT entrypoint.  The
per-Q variant has four independent 2x3 depth bases, one for each adjacent
three-tap group, and keeps the formal LLLG+GATE decoder and PLDD contract.
"""
from __future__ import annotations

from copy import deepcopy
from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F


SPEC = {
    "label": "HAP-RD-Lite-Per-Q",
    "student_width": 192,
    "pdfb_hidden": 128,
    "podp": True,
    "q_groups": 4,
    "parameters": 3761240,
    "teacher_taps": list(range(12)),
    "block_order": "LLLG",
    "student_blocks": 4,
    "ffn": "gate",
    "ffn_hidden_ratio": 2,
    "attention_heads": 8,
    "reconstruction_heads": 12,
    "teacher_width": 768,
    "code_dim": 64,
    "tap_norm": True,
    "patches": 256,
    "topk": 5,
    "alpha": 0.5,
}


def model_spec(variant: str) -> dict:
    if variant != "hap_rd_lite_per_q":
        raise ValueError(f"unknown UIT per-Q variant {variant!r}")
    return deepcopy(SPEC)


class GatedFFN(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.in_proj = nn.Linear(width, 4 * width)
        self.out_proj = nn.Linear(2 * width, width)

    def forward(self, x):
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(F.silu(gate) * value)


class GLMB(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.dwconv = nn.Conv2d(width, width, 3, padding=1, groups=width, bias=True)
        self.pointwise = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = GatedFFN(width)

    def forward(self, x):
        batch, patches, width = x.shape
        if patches != 256:
            raise ValueError("formal UIT run requires a 16x16 patch grid")
        z = self.norm1(x).transpose(1, 2).reshape(batch, width, 16, 16)
        z = self.dwconv(z).flatten(2).transpose(1, 2)
        x = x + self.pointwise(z)
        return x + self.ffn(self.norm2(x))


class GGMB(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, 8, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = GatedFFN(width)

    def forward(self, x):
        z = self.norm1(x)
        z, _ = self.attn(z, z, z, need_weights=False)
        x = x + z
        return x + self.ffn(self.norm2(x))


class LGGD(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.proj_in = nn.Linear(64, width)
        self.blocks = nn.ModuleList([GLMB(width), GLMB(width), GLMB(width), GGMB(width)])
        self.heads = nn.ModuleList(nn.Linear(width, 768) for _ in range(12))
        self.routes = {stage: list(range((3 - stage) * 3, (4 - stage) * 3)) for stage in range(4)}

    def forward(self, code):
        x = self.proj_in(code)
        predictions = [None] * 12
        for stage, block in enumerate(self.blocks):
            x = block(x)
            for tap in self.routes[stage]:
                predictions[tap] = self.heads[tap](x)
        return predictions


class PerGroupPDFB(nn.Module):
    """PDFB with four independently trainable row-orthonormal Q matrices."""

    def __init__(self, hidden: int, seed: int):
        super().__init__()
        self.podp = True
        self.q_groups = 4
        self.tap_norms = nn.ModuleList(nn.LayerNorm(768) for _ in range(12))
        self.fuse = nn.Sequential(nn.Linear(8 * 768, hidden), nn.GELU(), nn.Linear(hidden, 64))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        sample = torch.randn((4, 2, 3), generator=generator)
        first = F.normalize(sample[:, 0, :], dim=-1)
        second = sample[:, 1, :] - (sample[:, 1, :] * first).sum(-1, keepdim=True) * first
        self.raw_depth_basis = nn.Parameter(torch.stack((first, F.normalize(second, dim=-1)), dim=1))

    def effective_depth_basis(self):
        first = F.normalize(self.raw_depth_basis[:, 0, :], dim=-1, eps=1e-8)
        second = self.raw_depth_basis[:, 1, :]
        second = second - (second * first).sum(-1, keepdim=True) * first
        return torch.stack((first, F.normalize(second, dim=-1, eps=1e-8)), dim=1)

    def forward(self, taps):
        if len(taps) != 12:
            raise ValueError("PDFB requires twelve teacher taps")
        normalized = [norm(tap) for norm, tap in zip(self.tap_norms, taps)]
        q = self.effective_depth_basis()
        modes = []
        for group, start in enumerate(range(0, 12, 3)):
            first, middle, last = normalized[start:start + 3]
            for row in range(2):
                modes.append(q[group, row, 0] * first + q[group, row, 1] * middle
                             + q[group, row, 2] * last)
        return self.fuse(torch.cat(modes, dim=-1))


def build_hap_rd_lite_per_q(seed: int):
    bottleneck = PerGroupPDFB(SPEC["pdfb_hidden"], seed)
    decoder = LGGD(SPEC["student_width"])
    actual_guard(bottleneck, decoder, "hap_rd_lite_per_q")
    return bottleneck, decoder


def actual_guard(bottleneck, decoder, variant: str):
    spec = model_spec(variant)
    if [type(block) for block in decoder.blocks] != [GLMB, GLMB, GLMB, GGMB]:
        raise ValueError("actual decoder is not exactly LLLG")
    width = spec["student_width"]
    for block in decoder.blocks:
        if type(block.ffn) is not GatedFFN or block.ffn.in_proj.out_features != 4 * width \
                or block.ffn.out_proj.in_features != 2 * width:
            raise ValueError("actual FFN is not gated ratio2")
        if isinstance(block, GLMB) and (block.dwconv.kernel_size != (3, 3)
                                        or block.dwconv.groups != width):
            raise ValueError("wrong local mixer")
    if decoder.blocks[3].attn.num_heads != 8 or decoder.proj_in.out_features != width:
        raise ValueError("wrong attention or student width")
    if len(decoder.heads) != 12 or any(head.in_features != width or head.out_features != 768
                                       for head in decoder.heads):
        raise ValueError("twelve independent width-to-768 heads required")
    if decoder.routes != {0: [9, 10, 11], 1: [6, 7, 8], 2: [3, 4, 5], 3: [0, 1, 2]}:
        raise ValueError("wrong reverse teacher pairing")
    if len(bottleneck.tap_norms) != 12 or not all(isinstance(n, nn.LayerNorm)
                                                  for n in bottleneck.tap_norms):
        raise ValueError("per-tap normalization missing")
    if bottleneck.fuse[0].in_features != 8 * 768 or bottleneck.fuse[0].out_features != 128 \
            or bottleneck.fuse[2].out_features != 64:
        raise ValueError("wrong per-Q PDFB dimensions")
    q = bottleneck.effective_depth_basis().detach().cpu()
    if q.shape != (4, 2, 3) or not torch.allclose(q @ q.transpose(1, 2),
                                                   torch.eye(2).expand(4, -1, -1), atol=1e-5):
        raise ValueError("per-group Q must be row-orthonormal")
    count = sum(p.numel() for p in bottleneck.parameters()) + sum(p.numel() for p in decoder.parameters())
    if count != spec["parameters"]:
        raise ValueError(f"actual parameter count {count} != {spec['parameters']}")
    return {**spec, "variant": variant, "observed_block_types": [type(b).__name__ for b in decoder.blocks],
            "routes": decoder.routes, "q_shape": list(q.shape),
            "q_groups": 4, "q_shared_across_groups": False,
            "q_init": "seeded_random_orthonormal",
            "trainable_parameters": count}


def pldd_match(taps, predictions):
    if len(taps) != 12 or len(predictions) != 12:
        raise ValueError("twelve matched layer predictions required")
    errors = []
    for teacher, student in zip(taps, predictions):
        if teacher.shape != student.shape or teacher.ndim != 3 or teacher.shape[1:] != (256, 768):
            raise ValueError("expected matched Bx256x768 features")
        errors.append((1 - F.cosine_similarity(teacher, student, dim=-1)).topk(5, dim=-1).values.mean(-1))
    groups = torch.stack(errors, dim=1).reshape(taps[0].shape[0], 4, 3)
    return (0.5 * groups.mean(-1) + 0.5 * groups.max(-1).values).mean(1)
