#!/usr/bin/env python3
"""Audit the DA HAP-RD-Lite per-group-Q plan before training."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from model.publication_rd import architecture_guard, build_publication_model
from scripts.da_publication_scene import config


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    experiments = [(ROOT / path).resolve() for path in args.experiments]
    if len(experiments) != 3:
        raise ValueError("expected seed42/43/44 experiment directories")
    expected = {42, 43, 44}
    observed = set()
    model_checks = []
    for experiment in experiments:
        cfg = config(experiment)
        observed.add(cfg["seed"])
        if cfg["variant"] != "hap_rd_lite_q4":
            raise ValueError(f"unexpected variant in {experiment}")
        if cfg["model"].get("q_groups") != 4:
            raise ValueError(f"per-group Q missing in {experiment}")
        torch.manual_seed(cfg["seed"])
        bottleneck, decoder = build_publication_model(cfg["variant"], cfg["seed"])
        guard = architecture_guard(bottleneck, decoder, cfg["variant"])
        if guard["trainable_parameters"] != 3_761_240:
            raise RuntimeError("unexpected parameter count")
        taps = [torch.randn(2, 256, 768) for _ in range(12)]
        reconstructions = decoder(bottleneck(taps))
        loss = sum(item.square().mean() for item in reconstructions)
        loss.backward()
        if bottleneck.raw_depth_basis.grad is None:
            raise RuntimeError("per-group Q has no gradient")
        optimizer = torch.optim.AdamW(list(bottleneck.parameters()) + list(decoder.parameters()), lr=1e-3)
        optimizer.step()
        q = bottleneck.effective_depth_basis().detach()
        identity = torch.eye(2).expand(4, -1, -1)
        if not torch.allclose(q @ q.transpose(1, 2), identity, atol=1e-5):
            raise RuntimeError("per-group Q lost orthogonality after update")
        if torch.allclose(q[0], q[1], atol=1e-6) and torch.allclose(q[0], q[2], atol=1e-6):
            raise RuntimeError("all group Q matrices are identical")
        model_checks.append({"experiment": experiment.name, "guard": guard,
                             "forward_backward": "passed",
                             "post_step_orthogonality": "passed",
                             "independent_initialization": "passed"})

    if observed != expected:
        raise ValueError(f"expected seeds {sorted(expected)}, got {sorted(observed)}")
    code_paths = [
        "model/publication_rd.py",
        "model/bottleneck.py",
        "model/bottleneck_depthrank.py",
        "model/bottleneck_depthrank_learnable.py",
        "model/bottleneck_depthrank_group_learnable.py",
        "model/student_structural.py",
        "model/teacher.py",
        "scripts/da_publication_scene.py",
        "scripts/da_publication_scheduler.py",
        "scripts/preflight_da_hap_lite_q4.py",
    ]
    code_paths += [str((experiment / "config.json").relative_to(ROOT)) for experiment in experiments]
    code_paths += [str((experiment / "scripts/run_scene.py").relative_to(ROOT)) for experiment in experiments]
    report = {
        "schema_version": 2,
        "status": "passed",
        "variant": "hap_rd_lite_q4",
        "seeds": sorted(observed),
        "model_checks": model_checks,
        "code_hashes": {path: sha(ROOT / path) for path in code_paths},
        "torch": torch.__version__,
        "parameter_delta_vs_shared_q": 18,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "passed", "experiments": len(experiments),
                      "parameter_delta_vs_shared_q": 18}, ensure_ascii=False))


if __name__ == "__main__":
    main()
