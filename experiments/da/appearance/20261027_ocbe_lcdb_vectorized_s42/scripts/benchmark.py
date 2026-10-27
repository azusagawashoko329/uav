"""Synthetic seven-scene speed audit for Vectorized-DGT and LCDB."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.bottleneck import build_bottleneck
from model.bottleneck_depthrank_learnable import build_learnable_depthrank2_bottleneck
from model.bottleneck_dgt import build_depth_group_gated_tensor_bottleneck
from model.bottleneck_dgt_vectorized import build_vectorized_dgt_bottleneck
from model.bottleneck_lcdb import build_lcdb_bottleneck
from model.loss import anomaly_score
from model.student_structural import FourBlockGlobalLocalDecoder

SCENES = ["bike", "crossroads", "farmland", "highway", "railway", "solar", "vehicle"]
Q_ROOT = ROOT / "experiments/da/appearance/20261013_ocbe_depthrank2_randomorth_h128_s42/results"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_q(scene):
    path = Q_ROOT / f"{scene}_basis.json"
    data = json.loads(path.read_text())
    return torch.tensor(data["history"][-1]["raw_q"]), digest(path)


def build(kind, device, q):
    torch.manual_seed(42)
    if kind == "dense128":
        model = build_bottleneck(n_taps=12, fuse_hidden=128)
    elif kind == "rank2":
        model = build_learnable_depthrank2_bottleneck(
            basis_init="random_orthonormal", basis_seed=42)
    elif kind == "dgt":
        model = build_depth_group_gated_tensor_bottleneck()
    elif kind == "vectorized_dgt":
        model = build_vectorized_dgt_bottleneck()
    elif kind == "lcdb":
        model = build_lcdb_bottleneck(hidden=128)
    else:
        raise ValueError(kind)
    if hasattr(model, "raw_depth_basis"):
        with torch.no_grad():
            model.raw_depth_basis.copy_(q)
    model = model.to(device).eval()
    if hasattr(model, "set_diagnostics_enabled"):
        model.set_diagnostics_enabled(False)
    return model


def prepare(model, fused):
    if hasattr(model, "cache_depth_basis_for_inference"):
        model.cache_depth_basis_for_inference(fuse_projection=fused)
    return model


def measure(fn, device, warmup, samples, repeats, batch):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        values, rounds = [], []
        for _ in range(repeats):
            current = []
            for _ in range(samples):
                start = time.perf_counter(); fn(); torch.cuda.synchronize(device)
                current.append((time.perf_counter() - start) * 1000)
            values.extend(current); rounds.append(current)
        torch.cuda.synchronize(device); start = time.perf_counter()
        for _ in range(samples): fn()
        torch.cuda.synchronize(device)
        queued = batch * samples / (time.perf_counter() - start)
    values.sort(); p50 = values[len(values) // 2]
    p95 = values[min(len(values) - 1, int(len(values) * .95))]
    return dict(latency_p50_ms=p50, latency_p95_ms=p95,
                serial_fps=1000 * batch / p50, queued_fps=queued,
                round_p50_ms=[sorted(x)[len(x) // 2] for x in rounds],
                batch=batch, samples=samples, repeats=repeats)


def taps(batch, device, scene):
    g = torch.Generator(device=device).manual_seed(4200 + SCENES.index(scene) * 17 + batch)
    return [torch.randn(batch, 256, 768, device=device, generator=g) for _ in range(12)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--variant", choices=("dense128", "rank2", "dgt", "vectorized_dgt", "lcdb"), required=True)
    args = ap.parse_args(); config = json.loads((EXP / "config.json").read_text())
    device = torch.device(f"cuda:{args.gpu}"); torch.cuda.set_device(device)
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    env = dict(started_at=stamp(), source_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        gpu=args.gpu, cuda_device_name=torch.cuda.get_device_name(device),
        torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
        dtype="float32", tf32=False, cpu_threads=torch.get_num_threads(),
        affinity=sorted(os.sched_getaffinity(0)))
    result = dict(variant=args.variant, environment=env, scenes=[], q_sha256={})
    student = FourBlockGlobalLocalDecoder(dim=192, teacher_dim=768).to(device).eval()
    for scene in SCENES:
        q, qsha = load_q(scene); result["q_sha256"][scene] = qsha
        base = build(args.variant, device, q.to(device))
        original = copy.deepcopy(base).eval()
        cached = copy.deepcopy(base).eval(); fused = copy.deepcopy(base).eval()
        prepare(cached, False); prepare(fused, True)
        for m in (original, cached, fused):
            if hasattr(m, "set_diagnostics_enabled"): m.set_diagnostics_enabled(False)
        scene_result = dict(scene=scene, timings={}, verification={})
        for batch in config["batches"]:
            x = taps(batch, device, scene)
            with torch.inference_mode():
                y0, y1, y2 = original(x), cached(x), fused(x)
                s0, s1, s2 = student(y0), student(y1), student(y2)
                a0, a1, a2 = anomaly_score(x, s0), anomaly_score(x, s1), anomaly_score(x, s2)
                scene_result["verification"][str(batch)] = dict(
                    cached_code_max_abs_diff=float((y0-y1).abs().max().cpu()),
                    fused_code_max_abs_diff=float((y0-y2).abs().max().cpu()),
                    cached_score_max_abs_diff=float((a0-a1).abs().max().cpu()),
                    fused_score_max_abs_diff=float((a0-a2).abs().max().cpu()))
            funcs = {"original": original, "cached_loop": cached, "fused": fused}
            scene_result["timings"][str(batch)] = {
                scope: {name: measure((lambda m=m: student(m(x))) if scope == "subnet"
                                       else (lambda m=m: anomaly_score(x, student(m(x)))),
                                       device, config["warmup"],
                                       config["samples"] if batch == 1 else config["batch16_samples"],
                                       config["repeats"], batch)
                        for name, m in funcs.items()}
                for scope in ("subnet", "score")}
            del x, y0, y1, y2, s0, s1, s2, a0, a1, a2
        result["scenes"].append(scene_result)
        del base, original, cached, fused
        torch.cuda.empty_cache()
        print(json.dumps({"event": "measured", "variant": args.variant, "scene": scene,
                          "fps": {k: scene_result["timings"]["1"]["subnet"][k]["serial_fps"]
                                  for k in ("original", "cached_loop", "fused")}}), flush=True)
    result["finished_at"] = stamp()
    dump(EXP / "results" / f"{args.variant}_gpu{args.gpu}.json", result)
    print(json.dumps({"event": "completed", "variant": args.variant, "gpu": args.gpu}), flush=True)


if __name__ == "__main__": main()
