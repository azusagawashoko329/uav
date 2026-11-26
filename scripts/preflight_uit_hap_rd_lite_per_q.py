#!/usr/bin/env python3
"""Preflight the dedicated UIT HAP-RD-Lite-Per-Q entrypoint."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.hap_lllg_per_q import actual_guard, build_hap_rd_lite_per_q, pldd_match
from model.teacher import build_appearance_teacher
from scripts.uit_hap_rd_lite_per_q_runner import data_manifest, read, save_json, sha, state_hash, checked_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.manifest)
    owner = args.manifest.resolve().parent
    if len(manifest["experiments"]) != 3:
        raise ValueError("expected three UIT per-Q experiments")

    torch.set_num_threads(2)
    _, _, data = data_manifest()
    teacher = build_appearance_teacher(layers=list(range(12))).eval()
    teacher_hash = state_hash(teacher)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError("teacher must be frozen")
    report = {
        "schema_version": 2,
        "status": "running",
        "variant": "hap_rd_lite_per_q",
        "seeds": [],
        "data": data,
        "teacher": {"frozen": True, "taps": list(range(12)), "state_sha256": teacher_hash},
        "torch": torch.__version__,
        "checks": [],
    }

    for relative in manifest["experiments"]:
        experiment = ROOT / relative
        config = checked_config(experiment)
        if config["variant"] != "hap_rd_lite_per_q":
            raise ValueError(f"wrong variant in {experiment}")
        seed = config["seed"]
        torch.manual_seed(seed)
        bottleneck, decoder = build_hap_rd_lite_per_q(seed)
        guard = actual_guard(bottleneck, decoder, config["variant"])
        if guard["q_shape"] != [4, 2, 3] or guard["q_shared_across_groups"]:
            raise ValueError("per-Q guard does not describe four independent Q matrices")
        if guard["trainable_parameters"] != 3761240:
            raise ValueError("unexpected per-Q parameter count")

        initial_q = bottleneck.effective_depth_basis().detach()
        if all(torch.allclose(initial_q[0], initial_q[index], atol=1e-6) for index in range(1, 4)):
            raise ValueError("all per-group Q matrices are identical")
        taps = [torch.randn(2, 256, 768) for _ in range(12)]
        predictions = decoder(bottleneck(taps))
        score = pldd_match(taps, predictions)
        score.mean().backward()
        if bottleneck.raw_depth_basis.grad is None or not torch.isfinite(bottleneck.raw_depth_basis.grad).all():
            raise ValueError("per-group Q has no finite gradient")
        optimizer = torch.optim.AdamW(list(bottleneck.parameters()) + list(decoder.parameters()), lr=5e-4)
        optimizer.step()
        q_after = bottleneck.effective_depth_basis().detach()
        if not torch.allclose(q_after @ q_after.transpose(1, 2), torch.eye(2).expand(4, -1, -1), atol=1e-5):
            raise ValueError("per-group Q lost orthogonality after update")
        report["seeds"].append(seed)
        report["checks"].append({
            "experiment": experiment.name,
            "seed": seed,
            "guard": guard,
            "forward_backward": "passed",
            "independent_initialization": "passed",
            "post_step_orthogonality": "passed",
            "pldd_entrypoint": "passed",
        })
        del bottleneck, decoder, taps, predictions, score, optimizer
        gc.collect()
        print(f"preflight passed {experiment.name}", flush=True)

    if sorted(report["seeds"]) != [42, 43, 44]:
        raise ValueError(f"unexpected seeds {report['seeds']}")
    code_paths = [
        "model/hap_lllg_per_q.py",
        "model/teacher.py",
        "scripts/uit_hap_rd_lite_per_q_runner.py",
        "scripts/uit_hap_rd_lite_per_q_scheduler.py",
        "scripts/preflight_uit_hap_rd_lite_per_q.py",
    ]
    code_paths += [str(Path(relative) / name) for relative in manifest["experiments"]
                   for name in ["config.json", "scripts/run_uit.py"]]
    report["code_hashes"] = {relative: sha(ROOT / relative) for relative in code_paths}
    report["status"] = "passed"
    save_json(owner / "preflight.json", report)
    print("PREFLIGHT PASSED: three UIT per-Q configs, actual model, gradients, orthogonality and PLDD verified", flush=True)


if __name__ == "__main__":
    main()
