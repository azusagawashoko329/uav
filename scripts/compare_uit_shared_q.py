#!/usr/bin/env python3
"""Compare completed UIT shared-Q and per-Q HAP-RD-Lite results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
SHARED = "experiments/uit/appearance/20261120_formal_lllg_hap_rd_lite_s{seed}/results/uit.json"
PER_Q = "experiments/uit/appearance/20261127_formal_lllg_hap_rd_lite_per_q_s{seed}/results/uit.json"
OUT = ROOT / "experiments/uit/appearance/20261127_formal_lllg_hap_rd_lite_per_q_s42/results/compare_shared_q.json"
MD = OUT.with_suffix(".md")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    shared = [read(ROOT / SHARED.format(seed=seed)) for seed in SEEDS]
    per_q = [read(ROOT / PER_Q.format(seed=seed)) for seed in SEEDS]
    rows = []
    for seed, shared_result, per_q_result in zip(SEEDS, shared, per_q):
        rows.append({
            "seed": seed,
            "shared_q_auc_percent": shared_result["auc_percent"],
            "per_q_auc_percent": per_q_result["auc_percent"],
            "delta_auc_percent": per_q_result["auc_percent"] - shared_result["auc_percent"],
            "shared_q_eer": shared_result["eer"],
            "per_q_eer": per_q_result["eer"],
            "delta_eer": per_q_result["eer"] - shared_result["eer"],
        })
    auc_shared = np.array([row["shared_q_auc_percent"] for row in rows])
    auc_per_q = np.array([row["per_q_auc_percent"] for row in rows])
    eer_shared = np.array([row["shared_q_eer"] for row in rows])
    eer_per_q = np.array([row["per_q_eer"] for row in rows])
    summary = {
        "seeds": list(SEEDS),
        "shared_q_parameters": 3761222,
        "per_q_parameters": 3761240,
        "parameter_delta": 18,
        "shared_q_auc_mean_percent": float(auc_shared.mean()),
        "shared_q_auc_sd_percent": float(auc_shared.std(ddof=1)),
        "per_q_auc_mean_percent": float(auc_per_q.mean()),
        "per_q_auc_sd_percent": float(auc_per_q.std(ddof=1)),
        "delta_auc_mean_percent": float((auc_per_q - auc_shared).mean()),
        "shared_q_eer_mean": float(eer_shared.mean()),
        "shared_q_eer_sd": float(eer_shared.std(ddof=1)),
        "per_q_eer_mean": float(eer_per_q.mean()),
        "per_q_eer_sd": float(eer_per_q.std(ddof=1)),
        "delta_eer_mean": float((eer_per_q - eer_shared).mean()),
        "rows": rows,
        "scope": "UIT global AUC/EER over all 51 test videos and 147005 frames per seed",
        "caveat": "Three paired seeds support descriptive comparison only, not significance testing.",
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    MD.write_text(
        "# UIT HAP-RD-Lite shared-Q vs per-Q\n\n"
        "The two variants use the same UIT split, 45 epochs, LLLG+GATE decoder, PLDD scoring, and global test-frame aggregation. Per-Q adds 18 trainable scalars.\n\n"
        "| seed | shared-Q AUC | per-Q AUC | ΔAUC (pp) | shared-Q EER | per-Q EER | ΔEER |\n"
        "|---:|---:|---:|---:|---:|---:|---:|\n" +
        "\n".join(
            f"| {row['seed']} | {row['shared_q_auc_percent']:.4f} | {row['per_q_auc_percent']:.4f} | "
            f"{row['delta_auc_percent']:+.4f} | {row['shared_q_eer']:.6f} | {row['per_q_eer']:.6f} | "
            f"{row['delta_eer']:+.6f} |" for row in rows
        ) +
        f"\n\n| mean ± SD | {auc_shared.mean():.4f} ± {auc_shared.std(ddof=1):.4f} | "
        f"{auc_per_q.mean():.4f} ± {auc_per_q.std(ddof=1):.4f} | "
        f"{(auc_per_q - auc_shared).mean():+.4f} | {eer_shared.mean():.6f} ± {eer_shared.std(ddof=1):.6f} | "
        f"{eer_per_q.mean():.6f} ± {eer_per_q.std(ddof=1):.6f} | "
        f"{(eer_per_q - eer_shared).mean():+.6f} |\n\n"
        "AUC is higher-is-better and EER is lower-is-better. The paired mean difference is small relative to the seed spread, so this comparison does not show a material per-Q gain.\n\n"
        "Three paired seeds support descriptive comparison only, not significance testing.\n"
    )


if __name__ == "__main__":
    main()
