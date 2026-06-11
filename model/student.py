"""Reverse-distillation student DECODER for UAV-VAD (appearance axis).

Takes the OCBE bottleneck code (B,196,code_dim) and REGENERATES the frozen teacher's tapped patch
features (B,196,embed_dim) at each tap layer. It never sees the image — only the squeezed, normal-
trained code — so it can only rebuild "normal-vocabulary" features; an anomaly (whose info the
bottleneck dropped) is rebuilt as its nearest-normal -> mismatch with the teacher = the score.

Capacity is NOT the anomaly mechanism here (the bottleneck is), so the decoder can be full-width
(embed_dim) and as deep as you like without leaking — depth is a reconstruction-quality / compute
knob, not an R2 knob. Cross-patch self-attention IS used (unlike the per-patch bottleneck): each
patch's code is already anomaly-stripped, so sharing context across patches only helps normal recon.

Reverse alignment: teacher taps are given shallow->deep; the decoder generates abstract->concrete,
so the DEEPEST teacher tap is reconstructed at an EARLY decoder stage and the SHALLOWEST at a LATE
stage. CLS and the 4 register tokens are dropped — this pipeline is purely the 196 patch tokens
(the strong, discriminative leg for DINOv3). A global CLS head is a possible later add-on, not core.
"""
import torch
import torch.nn as nn


class Block(nn.Module):
    """Minimal pre-norm transformer block: self-attention over the 196 patches + MLP."""
    def __init__(self, dim, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x):
        xn = self.norm1(x)
        a, _ = self.attn(xn, xn, xn, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class StudentDecoder(nn.Module):
    def __init__(self, n_taps=3, code_dim=64, embed_dim=768, depth=9, heads=8):
        super().__init__()
        self.n_taps = n_taps
        self.embed_dim = embed_dim
        self.proj_in = nn.Linear(code_dim, embed_dim)        # expand the squeezed code
        self.blocks = nn.ModuleList([Block(embed_dim, heads) for _ in range(depth)])
        # one reconstruction head per tap, at evenly spaced decoder depths (last is always block depth-1)
        self.recon_at = [round((i + 1) * depth / n_taps) - 1 for i in range(n_taps)]   # ascending
        self.heads = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_taps)])

    def forward(self, code):
        """code: (B,196,code_dim). Returns list of n_taps reconstructions (B,196,embed_dim),
        ordered to match the teacher taps list (shallow->deep)."""
        x = self.proj_in(code)
        stage_outs, hi = [], 0
        for bi, blk in enumerate(self.blocks):
            x = blk(x)
            if bi in self.recon_at:
                stage_outs.append(self.heads[hi](x))         # early -> late decoder stages
                hi += 1
        return stage_outs[::-1]                              # reverse-align -> teacher-tap order (shallow->deep)


def build_student(n_taps=3, code_dim=64, embed_dim=768, depth=9, heads=8):
    return StudentDecoder(n_taps=n_taps, code_dim=code_dim, embed_dim=embed_dim, depth=depth, heads=heads)


if __name__ == '__main__':
    n_taps, code_dim, depth = 3, 64, 9
    dec = build_student(n_taps=n_taps, code_dim=code_dim, depth=depth)
    code = torch.randn(2, 196, code_dim)
    recon = dec(code)
    n_train = sum(p.numel() for p in dec.parameters() if p.requires_grad) / 1e6
    print(f'StudentDecoder | trainable={n_train:.1f}M  depth={depth}  recon_at(blocks)={dec.recon_at}')
    print(f'in code {tuple(code.shape)} -> {len(recon)} reconstructions, each {tuple(recon[0].shape)} (teacher-tap order)')
