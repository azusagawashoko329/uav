"""One-Class Bottleneck Embedding (OCBE) for reverse-distillation UAV-VAD.

Sits between the FROZEN DINOv3 teacher and the (trainable) student decoder. It fuses the teacher's
tapped patch features and SQUEEZES them through a narrow `code_dim` channel.

Why this is the whole mechanism (the collapse we stress-tested):
  A token-wise path  embed_dim -> code_dim -> ...  has rank <= code_dim. With code_dim << embed_dim
  it CANNOT be the identity on embed_dim (identity has rank embed_dim), so the teacher features
  cannot be passed through unchanged -> the "constant/identity gives zero loss" collapse is
  physically impossible. Trained on NORMAL only, the limited code_dim capacity is spent fitting the
  normal-feature manifold. At test, an anomalous feature (off that manifold) is projected to its
  nearest-normal code -> the decoder reconstructs a "normal version" -> mismatch with the teacher's
  real (anomalous) feature = the anomaly signal.

TRAINABLE (not frozen). `code_dim` is THE critical knob: too wide -> identity collapse (no gap);
too narrow -> even normal reconstructs poorly. The sweet spot exists *because* training is one-class.

Kept token-wise (no cross-patch mixing) on purpose, so the rank/identity argument above stays exact
and so anomaly information can't route sideways into neighbouring patches. A cross-patch transformer
block before the squeeze is a deliberate later option (helps global/logical anomalies, but muddies
the clean per-patch chokepoint) — left out for now.
"""
import torch
import torch.nn as nn


class OCBE(nn.Module):
    def __init__(self, embed_dim=768, n_taps=3, code_dim=64, fuse_hidden=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_taps = n_taps
        self.code_dim = code_dim
        assert code_dim < embed_dim, "code_dim must be << embed_dim or it is not a bottleneck"
        # different teacher depths have different feature scales/stats; LayerNorm each tap BEFORE
        # concat so no single layer dominates the squeeze (the concat just hands all the same-patch
        # depth-views to the learnable fuse below — it doesn't assert they share semantics).
        self.tap_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_taps)])
        # fuse the n normalized taps (each B,196,embed_dim) -> mix -> squeeze to code_dim
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * n_taps, fuse_hidden),
            nn.GELU(),
            nn.Linear(fuse_hidden, code_dim),   # <-- the chokepoint: code_dim << embed_dim
        )

    def forward(self, taps):
        """taps: list of `n_taps` tensors, each (B, 196, embed_dim) — the teacher's tapped patches.
        Returns the compressed code (B, 196, code_dim)."""
        assert len(taps) == self.n_taps, f"expected {self.n_taps} taps, got {len(taps)}"
        x = torch.cat([nrm(t) for nrm, t in zip(self.tap_norms, taps)], dim=-1)   # (B,196, embed_dim*n_taps)
        code = self.fuse(x)                    # (B, 196, code_dim)  ← squeezed
        return code


def build_bottleneck(n_taps=3, code_dim=64, embed_dim=768, fuse_hidden=512):
    return OCBE(embed_dim=embed_dim, n_taps=n_taps, code_dim=code_dim, fuse_hidden=fuse_hidden)


if __name__ == '__main__':
    n_taps, code_dim = 3, 64
    ocbe = build_bottleneck(n_taps=n_taps, code_dim=code_dim)
    taps = [torch.randn(2, 196, 768) for _ in range(n_taps)]
    code = ocbe(taps)
    n_train = sum(p.numel() for p in ocbe.parameters() if p.requires_grad)
    print(f'OCBE | trainable={n_train/1e3:.1f}K  in: {n_taps}x(2,196,768)  ->  code {tuple(code.shape)}')
    print(f'chokepoint: 768 -> {code_dim}  (rank <= {code_dim} < 768, identity impossible)')
