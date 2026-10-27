"""Re-measure DGT-family inference after fusing the shared Rank2 projection."""
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
from model.bottleneck_dgt_variants import build_dgt_variant
from model.loss import anomaly_score
from model.student_structural import FourBlockGlobalLocalDecoder


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_config():
    return json.loads((EXP / "config.json").read_text())


def scene_q(spec, scene):
    info = spec["weight_files"].get(scene)
    if not info or info["kind"] != "final_Q_json":
        return None
    data = json.loads((ROOT / info["path"]).read_text())
    return data["history"][-1]["raw_q"]


def make_models(spec, scene, device):
    torch.manual_seed(42)
    kind = spec["ocbe"]
    if kind == "dense":
        ocbe = build_bottleneck(n_taps=12, fuse_hidden=128)
    elif kind == "rank2_random":
        ocbe = build_learnable_depthrank2_bottleneck(
            basis_init="random_orthonormal", basis_seed=42)
    elif kind == "dgt_graph":
        ocbe = build_depth_group_gated_tensor_bottleneck()
    else:
        ocbe = build_dgt_variant(kind)
    student = FourBlockGlobalLocalDecoder(dim=192, teacher_dim=768)

    checkpoint = ROOT / spec["experiment"] / "ckpt" / f"{scene}_final.pth"
    loaded_checkpoint = False
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        ocbe.load_state_dict(state["ocbe_state_dict"], strict=True)
        student.load_state_dict(state["student_state_dict"], strict=True)
        loaded_checkpoint = True
    elif scene_q(spec, scene) is not None and hasattr(ocbe, "raw_depth_basis"):
        with torch.no_grad():
            ocbe.raw_depth_basis.copy_(torch.tensor(scene_q(spec, scene)))

    models = {"original": ocbe.to(device).eval()}
    if hasattr(ocbe, "set_diagnostics_enabled"):
        models["original"].set_diagnostics_enabled(False)
    models["cached_loop"] = copy.deepcopy(models["original"])
    models["fused"] = copy.deepcopy(models["original"])
    for name in ("cached_loop", "fused"):
        if hasattr(models[name], "cache_depth_basis_for_inference"):
            models[name].cache_depth_basis_for_inference(
                fuse_projection=(name == "fused"))
        if hasattr(models[name], "set_diagnostics_enabled"):
            models[name].set_diagnostics_enabled(False)
    student = student.to(device).eval()
    return models, student, loaded_checkpoint


def measure(fn, device, warmup, samples, repeats, batch):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        values = []
        rounds = []
        for _ in range(repeats):
            current = []
            for _ in range(samples):
                start = time.perf_counter()
                fn()
                torch.cuda.synchronize(device)
                current.append((time.perf_counter() - start) * 1000)
            values.extend(current)
            rounds.append(current)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(samples):
            fn()
        torch.cuda.synchronize(device)
        queued = batch * samples / (time.perf_counter() - start)
    values.sort()
    p50 = values[len(values) // 2]
    p95 = values[min(len(values) - 1, int(len(values) * .95))]
    return dict(latency_p50_ms=p50, latency_p95_ms=p95,
                latency_mean_ms=sum(values) / len(values),
                serial_fps=1000 * batch / p50, queued_fps=queued,
                round_p50_ms=[sorted(x)[len(x) // 2] for x in rounds],
                batch=batch, samples=samples, repeats=repeats)


def input_taps(batch, device):
    # Five prefix tokens mirror the teacher output used by the training code.
    g = torch.Generator(device=device).manual_seed(4200 + batch)
    return [torch.randn(batch, 256, 768, device=device, generator=g)
            for _ in range(12)]


def run_variant(spec, config, device, gpu):
    started = stamp()
    env = dict(started_at=started, source_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               gpu=gpu, cuda_device_name=torch.cuda.get_device_name(device),
               cuda=torch.version.cuda, torch=torch.__version__,
               python=platform.python_version(), dtype="float32", tf32=False,
               cpu_threads=torch.get_num_threads(), pid=os.getpid(),
               affinity=sorted(os.sched_getaffinity(0)))
    result = dict(id=spec["id"], label=spec["label"], environment=env,
                  experiment=spec["experiment"], scenes=[], loaded_checkpoints={})
    student = None
    for scene in config["scenes"]:
        device_models, student, loaded = make_models(spec, scene, device)
        result["loaded_checkpoints"][scene] = loaded
        scene_result = dict(scene=scene, verification={}, timings={})
        for batch in config["batches"]:
            taps = input_taps(batch, device)
            dense_input = torch.cat(taps, dim=-1)
            with torch.inference_mode():
                outputs = {k: model(taps) for k, model in device_models.items()}
                student_outputs = {k: student(v) for k, v in outputs.items()}
                scores = {k: anomaly_score(taps, student_outputs[k])
                          for k in outputs}
                reference = outputs["original"]
                scene_result["verification"][str(batch)] = {
                    "cached_code_max_abs_diff": float((reference - outputs["cached_loop"]).abs().max().cpu()),
                    "fused_code_max_abs_diff": float((reference - outputs["fused"]).abs().max().cpu()),
                    "cached_student_max_abs_diff": max(float((a - b).abs().max().cpu())
                                                        for a, b in zip(student_outputs["original"], student_outputs["cached_loop"])),
                    "fused_student_max_abs_diff": max(float((a - b).abs().max().cpu())
                                                       for a, b in zip(student_outputs["original"], student_outputs["fused"])),
                    "cached_score_max_abs_diff": float((scores["original"] - scores["cached_loop"]).abs().max().cpu()),
                    "fused_score_max_abs_diff": float((scores["original"] - scores["fused"]).abs().max().cpu()),
                }
            scopes = {
                "ocbe": {k: (lambda m=m: m(taps)) for k, m in device_models.items()},
                "subnet": {k: (lambda m=m: student(m(taps))) for k, m in device_models.items()},
                "score": {k: (lambda m=m: anomaly_score(taps, student(m(taps))))
                          for k, m in device_models.items()},
            }
            # The original dense control has no Q path. Use it only in the
            # separate control process/row to keep architecture timing visible.
            scene_result["timings"][str(batch)] = {
                scope: {name: measure(fn, device, config["warmup"],
                                      config["samples"] if batch == 1 else config["auxiliary"]["samples"],
                                      config["repeats"] if batch == 1 else config["auxiliary"]["repeats"], batch)
                        for name, fn in functions.items()}
                for scope, functions in scopes.items()
            }
            del taps, dense_input, outputs, student_outputs, scores
        result["scenes"].append(scene_result)
        del device_models
        torch.cuda.empty_cache()
        print(json.dumps({"event": "measured", "variant": spec["id"],
                          "scene": scene,
                          "fps": {k: scene_result["timings"]["1"]["subnet"][k]["serial_fps"]
                                  for k in ["original", "cached_loop", "fused"]}}), flush=True)
    result["finished_at"] = stamp()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--variant", required=True)
    args = ap.parse_args()
    config = load_config()
    spec = next(x for x in config["variants"] if x["id"] == args.variant)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(json.dumps({"event": "started", "variant": args.variant,
                      "gpu": args.gpu, "source_commit": subprocess.check_output(
                          ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}), flush=True)
    result = run_variant(spec, config, device, args.gpu)
    dump(EXP / "results" / f"{args.variant}_gpu{args.gpu}.json", result)
    print(json.dumps({"event": "completed", "variant": args.variant,
                      "gpu": args.gpu}), flush=True)


if __name__ == "__main__":
    main()
