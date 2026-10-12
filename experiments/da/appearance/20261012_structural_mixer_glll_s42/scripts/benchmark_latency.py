"""Inference latency comparison for the direct four-block baseline and structural student."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from model import build_appearance_teacher, build_bottleneck
from model.student_adapter import FourBlockTwelveHeadDecoder
from model.student_structural import FourBlockGlobalLocalDecoder


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def timed(fn, device, warmup, iterations):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        for _ in range(iterations):
            fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def run(device_id, warmup, iterations):
    device = torch.device(f"cuda:{device_id}")
    torch.manual_seed(42)
    code = torch.randn(1, 256, 64, device=device)
    taps = [torch.randn(1, 256, 768, device=device) for _ in range(12)]
    rows = []
    variants = {
        "four_block_mhsa": FourBlockTwelveHeadDecoder("none").to(device).eval(),
        "global_local_glll": FourBlockGlobalLocalDecoder().to(device).eval(),
    }
    for name, student in variants.items():
        ocbe = build_bottleneck(n_taps=12, code_dim=64, use_tap_norm=True).to(device).eval()
        student_ms = timed(lambda: student(code), device, warmup, iterations)
        pipeline_ms = timed(lambda: student(ocbe(taps)), device, warmup, iterations)
        rows.append({"variant": name, "student_parameters": count(student),
                     "ocbe_plus_student_parameters": count(student) + count(ocbe),
                     "student_forward_ms": student_ms,
                     "ocbe_plus_student_forward_ms": pipeline_ms})
        del ocbe, student
        torch.cuda.empty_cache()

    # The frozen teacher is common to both variants. Measuring it shows the likely lower bound
    # for end-to-end latency, while the comparison is decided by the OCBE+student delta above.
    teacher = build_appearance_teacher(layers=list(range(12))).to(device).eval()
    image = torch.randn(1, 3, 256, 256, device=device)
    teacher_ms = timed(lambda: teacher(image)["patches"], device, warmup, iterations)
    return {"device": torch.cuda.get_device_name(device), "gpu": device_id,
            "batch": 1, "patch_tokens": 256, "warmup": warmup,
            "iterations": iterations, "teacher_forward_ms": teacher_ms,
            "variants": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    result = run(args.gpu, args.warmup, args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
