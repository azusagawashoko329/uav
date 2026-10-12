"""Small, auditable module screen for student internal width/mixer choices.

This is intentionally not a training script. It reports parameter count, CUDA forward and
backward timing, and a small PCA effective-rank probe on normal DINOv3 patch features.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from model import build_appearance_teacher


class MLP(nn.Module):
    def __init__(self, d: int, ratio: float):
        super().__init__()
        h = int(d * ratio)
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, x):
        return self.net(x)


class MHSA(nn.Module):
    def __init__(self, d: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = MLP(d, mlp_ratio)

    def forward(self, x):
        y = self.n1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        return x + self.mlp(self.n2(x))


class GQA(nn.Module):
    def __init__(self, d: int, q_heads: int, kv_heads: int, mlp_ratio: float):
        super().__init__()
        assert q_heads % kv_heads == 0 and d % q_heads == 0
        self.d, self.qh, self.kvh = d, q_heads, kv_heads
        self.hd = d // q_heads
        self.n1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, kv_heads * self.hd)
        self.v = nn.Linear(d, kv_heads * self.hd)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = MLP(d, mlp_ratio)

    def forward(self, x):
        b, n, _ = x.shape
        z = self.n1(x)
        q = self.q(z).view(b, n, self.qh, self.hd).transpose(1, 2)
        k = self.k(z).view(b, n, self.kvh, self.hd).transpose(1, 2)
        v = self.v(z).view(b, n, self.kvh, self.hd).transpose(1, 2)
        repeat = self.qh // self.kvh
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).reshape(b, n, self.d)
        x = x + self.o(y)
        return x + self.mlp(self.n2(x))


class LinearAttention(nn.Module):
    """Naive positive-feature linear attention; included as a complexity sanity check."""
    def __init__(self, d: int, heads: int, mlp_ratio: float):
        super().__init__()
        assert d % heads == 0
        self.d, self.h, self.hd = d, heads, d // heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = MLP(d, mlp_ratio)

    def phi(self, x):
        return F.elu(x) + 1.0

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q = self.phi(q.view(b, n, self.h, self.hd).transpose(1, 2))
        k = self.phi(k.view(b, n, self.h, self.hd).transpose(1, 2))
        v = v.view(b, n, self.h, self.hd).transpose(1, 2)
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        denom = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + 1e-6)
        y = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, denom)
        y = y.transpose(1, 2).reshape(b, n, self.d)
        x = x + self.o(y)
        return x + self.mlp(self.n2(x))


class LocalDWConv(nn.Module):
    def __init__(self, d: int, mlp_ratio: float, side: int = 16):
        super().__init__()
        self.d, self.side = d, side
        self.n1 = nn.LayerNorm(d)
        self.dw = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.pw = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = MLP(d, mlp_ratio)

    def forward(self, x):
        b, n, d = x.shape
        z = self.n1(x).transpose(1, 2).reshape(b, d, self.side, self.side)
        z = self.dw(z).flatten(2).transpose(1, 2)
        x = x + self.pw(z)
        return x + self.mlp(self.n2(x))


def make_variant(name: str):
    if name == "mhsa_mlp4_d768":
        return MHSA(768, 8, 4.0)
    if name == "mhsa_mlp2_d768":
        return MHSA(768, 8, 2.0)
    if name == "mhsa_mlp4_d512":
        return MHSA(512, 8, 4.0)
    if name == "mhsa_mlp4_d384":
        return MHSA(384, 8, 4.0)
    if name == "gqa2_mlp2_d768":
        return GQA(768, 8, 2, 2.0)
    if name == "linear_mlp2_d768":
        return LinearAttention(768, 8, 2.0)
    if name == "local_dwconv_mlp2_d768":
        return LocalDWConv(768, 2.0)
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cuda_ms(fn, iters: int):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def benchmark(device, warmup: int, iters: int, backward_iters: int):
    out = []
    for name in [
        "mhsa_mlp4_d768", "mhsa_mlp2_d768", "mhsa_mlp4_d512", "mhsa_mlp4_d384",
        "gqa2_mlp2_d768", "linear_mlp2_d768", "local_dwconv_mlp2_d768",
    ]:
        torch.manual_seed(42)
        model = make_variant(name).to(device).train()
        d = 768 if "d768" in name else (512 if "d512" in name else 384)
        x = torch.randn(1, 256, d, device=device)
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
        fwd_ms = cuda_ms(lambda: model(x), iters)
        x = x.detach().requires_grad_(True)
        for _ in range(5):
            model.zero_grad(set_to_none=True)
            model(x).square().mean().backward()
        def train_step():
            model.zero_grad(set_to_none=True)
            model(x).square().mean().backward()
        bwd_ms = cuda_ms(train_step, backward_iters)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        out.append({"variant": name, "parameters": count_params(model),
                    "forward_ms": fwd_ms, "backward_ms": bwd_ms, "peak_allocated_mb": peak_mb})
        del model, x
        torch.cuda.empty_cache()
    return out


def rank_probe(device, data_root: Path, max_images: int):
    paths = sorted(glob.glob(str(data_root / "Highway" / "sequence*" / "train" / "*" / "*.jpg")))[:max_images]
    if not paths:
        return {"status": "skipped", "reason": "no highway train frames found", "data_root": str(data_root)}
    teacher = build_appearance_teacher(layers=list(range(12))).to(device).eval()
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize(teacher.img_mean, teacher.img_std)])
    layer_values = [[] for _ in range(12)]
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as im:
                xb = tf(im.convert("RGB")).unsqueeze(0).to(device)
            taps = teacher(xb)["patches"]
            for i, tap in enumerate(taps):
                layer_values[i].append(tap.squeeze(0).float().cpu())
    result = {"status": "completed", "images": len(paths), "tokens_per_image": 256, "dim": 768, "layers": []}
    for i, values in enumerate(layer_values):
        x = torch.cat(values, dim=0)
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / max(1, x.shape[0] - 1)
        eig = torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)
        energy = eig.cumsum(0) / eig.sum().clamp_min(1e-12)
        ranks = {str(p): int(torch.searchsorted(energy, p).item() + 1) for p in (0.9, 0.95, 0.99)}
        result["layers"].append({"layer": i, "r90": ranks["0.9"], "r95": ranks["0.95"], "r99": ranks["0.99"]})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("UAV_DA_ROOT", "/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly")))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=16)
    ap.add_argument("--skip-rank", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    result = {
        "seed": args.seed,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "benchmark": benchmark(device, 30, 100, 20),
        "rank_probe": None if args.skip_rank else rank_probe(device, args.data_root, args.max_images),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
