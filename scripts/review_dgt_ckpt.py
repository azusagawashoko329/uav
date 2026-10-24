"""Review one checkpoint-enabled DGT fusion ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch


SCENES = ["bike", "crossroads", "farmland", "highway", "railway", "solar", "vehicle"]
ROOT = Path(__file__).resolve().parents[1]
DGT_BASE = ROOT / "experiments/da/appearance/20261022_ocbe_dgt64_s42"
DENSE_BASE = ROOT / "experiments/da/appearance/20261013_ocbe_depthrank2_randomorth_h128_s42"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def final_loss(exp: Path, scene: str) -> float:
    logs = sorted((exp / "logs").glob(f"*_{scene}_gpu*.log"))
    if len(logs) != 1:
        raise ValueError(f"expected one scene log for {scene}, got {logs}")
    lines = [line for line in logs[0].read_text().splitlines() if line.startswith("epoch=")]
    if len(lines) != 15 or not lines[-1].startswith("epoch=15 "):
        raise ValueError(f"incomplete 15-epoch log for {scene}")
    return float(lines[-1].split("loss=")[1].split()[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=Path, required=True)
    args = ap.parse_args()
    exp = args.experiment.resolve()
    rows = []
    artifacts = []
    for scene in SCENES:
        current = json.loads((exp / "results" / f"{scene}.json").read_text())
        dgt = json.loads((DGT_BASE / "results" / f"{scene}.json").read_text())
        dense = json.loads((DENSE_BASE / "results" / f"{scene}.json").read_text())
        ckpt = exp / current["checkpoint"]
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        if sha256(ckpt) != current["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {scene}")
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        if payload["scene"] != scene or payload["split_fingerprint"] != current["split_fingerprint"]:
            raise ValueError(f"checkpoint metadata mismatch: {scene}")
        rows.append({
            "scene": scene,
            "auc_percent": current["auc_percent"],
            "vs_dgt_pp": current["auc_percent"] - dgt["auc_percent"],
            "vs_dense_pp": current["auc_percent"] - dense["auc_percent"],
            "final_loss": final_loss(exp, scene),
            "dgt_auc_percent": dgt["auc_percent"],
            "dense_auc_percent": dense["auc_percent"],
            "trainable_parameters": current["trainable_parameters"],
            "checkpoint": current["checkpoint"],
            "checkpoint_sha256": current["checkpoint_sha256"],
            **{f"fusion_{k}": v for k, v in current.get("fusion_diagnostics", {}).items()},
        })
        artifacts.append({
            "path": current["checkpoint"], "kind": "checkpoint",
            "producer_script": "scripts/run_ocbe.py", "run_id": None,
            "inputs": "config.json; Drone-Anomaly scene split",
            "status": "completed", "size_bytes": ckpt.stat().st_size,
            "sha256": current["checkpoint_sha256"], "storage": "local ignored artifact",
        })
    variant = json.loads((exp / "config.json").read_text())["variant"]
    auc = sum(r["auc_percent"] for r in rows) / len(rows)
    dgt_auc = sum(r["dgt_auc_percent"] for r in rows) / len(rows)
    dense_auc = sum(r["dense_auc_percent"] for r in rows) / len(rows)
    params = rows[0]["trainable_parameters"]
    result = {
        "experiment": exp.name, "variant": variant,
        "baseline_dgt": DGT_BASE.name, "baseline_dense": DENSE_BASE.name,
        "scenes": rows,
        "macro": {"auc_percent": auc, "vs_dgt_pp": auc - dgt_auc, "vs_dense_pp": auc - dense_auc,
                  "dgt_auc_percent": dgt_auc, "dense_auc_percent": dense_auc},
        "parameters": {"total": params, "dgt_total": 3179718,
                       "vs_dgt_reduction": 3179718 - params,
                       "vs_dgt_reduction_percent": 100 * (3179718 - params) / 3179718},
        "integrity": {"scene_count": 7, "all_scene_logs_have_15_epochs": True,
                      "scheduler_failures": [], "checkpoint_count": len(artifacts),
                      "checkpoint_hashes_verified": True},
    }
    record = json.loads((exp / "record.json").read_text())
    run_id = record["runs"][-1]["run_id"]
    for p in sorted((exp / "logs").glob("*")):
        if p.is_file():
            artifacts.append({"path": str(p.relative_to(exp)), "kind": "log",
                              "producer_script": "scripts/dynamic_scheduler.py", "run_id": run_id,
                              "inputs": "config.json; Drone-Anomaly scene split", "status": "completed",
                              "size_bytes": p.stat().st_size, "sha256": sha256(p), "storage": "git"})
    for p in sorted((exp / "results").glob("*")):
        if p.is_file():
            artifacts.append({"path": str(p.relative_to(exp)), "kind": "result",
                              "producer_script": "scripts/run_ocbe.py", "run_id": run_id,
                              "inputs": "config.json; Drone-Anomaly scene split", "status": "completed",
                              "size_bytes": p.stat().st_size, "sha256": sha256(p), "storage": "git"})
    for item in artifacts:
        if item["run_id"] is None: item["run_id"] = run_id
    (exp / "artifacts.json").write_text(json.dumps({"schema_version": 2, "artifacts": artifacts}, ensure_ascii=False, indent=2) + "\n")
    out = exp / "results"
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    fields = list(rows[0])
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = [
        f"# {variant}：seed42 七场景结果", "",
        "三组三层教师 Rank-2 Q、组内 value/gate、192 维 Global-Local 学生、三对一对齐和 group-balanced top-k loss 保持不变；本实验只改变四组深度融合算子。每个场景均保存最终 OCBE 和 student 权重。", "",
        "| 场景 | AUC | 相对 DGT pp | 相对 Dense pp | epoch15 loss |", "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['scene']} | {r['auc_percent']:.4f} | {r['vs_dgt_pp']:+.4f} | {r['vs_dense_pp']:+.4f} | {r['final_loss']:.6f} |")
    lines += ["", f"宏平均 AUC：**{auc:.4f}**；相对 DGT **{auc-dgt_auc:+.4f} pp**；相对 Dense **{auc-dense_auc:+.4f} pp**。", "", f"总可训练参数：**{params:,}**；相对 DGT 变化 **{params-3179718:+,}**。", "", "权重：每个场景的 `ckpt/<scene>_final.pth` 均已生成，哈希已在 `artifacts.json` 登记。", "", "该结果为 seed42 单种子观察，不能代表跨种子稳定性；本轮没有加入额外融合损失。"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"experiment": exp.name, "macro": result["macro"], "parameters": result["parameters"],
                      "checkpoint_count": len([x for x in artifacts if x["kind"] == "checkpoint"])}, ensure_ascii=False))
    return result, artifacts


if __name__ == "__main__":
    main()
