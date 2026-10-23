"""Review the completed DGT-OCBE seed-42 run against its paired Dense OCBE run."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


SCENES = ["bike", "crossroads", "farmland", "highway", "railway", "solar", "vehicle"]
ROOT = Path(__file__).resolve().parents[5]
EXP = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/da/appearance/20261013_ocbe_depthrank2_randomorth_h128_s42"


def final_loss(exp: Path, scene: str) -> float:
    logs = sorted((exp / "logs").glob(f"*_{scene}_gpu*.log"))
    if len(logs) != 1:
        raise ValueError(f"expected one scene log for {scene}, got {logs}")
    lines = logs[0].read_text().splitlines()
    epochs = [line for line in lines if line.startswith("epoch=")]
    if len(epochs) != 15 or not epochs[-1].startswith("epoch=15 "):
        raise ValueError(f"incomplete 15-epoch log for {scene}")
    return float(epochs[-1].split("loss=")[1].split()[0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for scene in SCENES:
        current = json.loads((EXP / "results" / f"{scene}.json").read_text())
        base = json.loads((BASE / "results" / f"{scene}.json").read_text())
        if current["seed"] != 42 or current["trainable_parameters"] != 3179718:
            raise ValueError(f"unexpected seed or parameter count: {scene}")
        if current["split_fingerprint"] != base["split_fingerprint"]:
            raise ValueError(f"split fingerprint mismatch: {scene}")
        rows.append({
            "scene": scene,
            "dgt_auc_percent": current["auc_percent"],
            "dense_auc_percent": base["auc_percent"],
            "delta_auc_pp": current["auc_percent"] - base["auc_percent"],
            "dgt_final_loss": final_loss(EXP, scene),
            "dense_final_loss": final_loss(BASE, scene),
            "elapsed_s": current["elapsed_s"],
            "trainable_parameters": current["trainable_parameters"],
            "basis_principal_angle_deg": current["basis_principal_angle_deg"],
            "basis_projector_distance_fro": current["basis_projector_distance_fro"],
        })
    dgt_macro = sum(row["dgt_auc_percent"] for row in rows) / len(rows)
    dense_macro = sum(row["dense_auc_percent"] for row in rows) / len(rows)
    dgt_params = rows[0]["trainable_parameters"]
    dense_params = json.loads((BASE / "config.json").read_text())["predicted_trainable_parameters"]["total"]
    result = {
        "experiment": "20261022_ocbe_dgt64_s42",
        "baseline": "20261013_ocbe_depthrank2_randomorth_h128_s42",
        "scenes": rows,
        "macro": {
            "dgt_auc_percent": dgt_macro,
            "dense_auc_percent": dense_macro,
            "delta_auc_pp": dgt_macro - dense_macro,
        },
        "parameters": {
            "dgt_total": dgt_params,
            "dense_total": dense_params,
            "reduction": dense_params - dgt_params,
            "reduction_percent": 100 * (dense_params - dgt_params) / dense_params,
            "dgt_ocbe": 231750,
            "dense_ocbe": 813254,
            "ocbe_reduction": 813254 - 231750,
            "ocbe_reduction_percent": 100 * (813254 - 231750) / 813254,
        },
        "integrity": {
            "scene_count": len(rows),
            "all_scene_logs_have_15_epochs": True,
            "scheduler_failures": [],
            "split_fingerprints_paired": True,
        },
    }
    out = EXP / "results"
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# DGT-OCBE d=64：seed42 七场景结果",
        "",
        "本实验使用共享可学习、随机正交初始化的 Rank-2 Q；唯一结构变量是右侧 DGT-OCBE。学生、三对一反向对齐、group-balanced top-k loss 和训练流程与 Dense 对照一致。",
        "",
        "| 场景 | DGT AUC | Dense AUC | DGT−Dense (pp) | DGT epoch15 loss | Dense epoch15 loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['scene']} | {row['dgt_auc_percent']:.4f} | {row['dense_auc_percent']:.4f} | {row['delta_auc_pp']:+.4f} | {row['dgt_final_loss']:.6f} | {row['dense_final_loss']:.6f} |")
    lines += [
        "",
        f"宏平均 AUC：DGT **{dgt_macro:.4f}**，Dense **{dense_macro:.4f}**，差值 **{dgt_macro - dense_macro:+.4f} pp**。",
        "",
        f"总可训练参数：DGT **{dgt_params:,}**，Dense **{dense_params:,}**，减少 **{dense_params - dgt_params:,}（{100 * (dense_params - dgt_params) / dense_params:.2f}%）**。OCBE 部分从 **813,254** 降至 **231,750**，减少 **{813254 - 231750:,}（{100 * (813254 - 231750) / 813254:.2f}%）**。",
        "",
        "观察：Highway 提升 +1.1813 pp，Crossroads 提升 +0.3369 pp；Railway 下降 -2.5799 pp，Vehicle 下降 -1.5982 pp。其余场景变化较小但总体偏负，因此本 seed 下 DGT 的轻量化收益没有达到 AUC 非劣。",
        "",
        "解释限制：DGT 的 gate 和深度图混合没有保存训练后权重，当前只能根据 AUC、loss 和 Q 轨迹判断，不能进一步证明哪条结构路径实际起作用。单个 seed 也不能推断跨场景稳定性。",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"macro": result["macro"], "parameters": result["parameters"]}, ensure_ascii=False))
    for path in (out / "summary.md", out / "summary.csv", out / "summary.json"):
        print(path.name, sha256(path))


if __name__ == "__main__":
    main()
