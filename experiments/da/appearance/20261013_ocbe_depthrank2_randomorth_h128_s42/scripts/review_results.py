#!/usr/bin/env python3
"""Audit random Q initialization against fixed- and learned-Q controls."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from scripts.summarize_ocbe import read_scheduler, read_scores  # noqa: E402

SCENES = {
    "bike": "Bike Roundabout", "crossroads": "Crossroads",
    "farmland": "Farmland Inspection", "highway": "Highway",
    "railway": "Railway Inspection", "solar": "Solar Panel Inspection",
    "vehicle": "Vehicle Roundabout",
}
FIXED = REPO / "experiments/da/appearance/20261013_ocbe_depthrank2_hidden128_s42"
LEARNED_FIXED_INIT = REPO / "experiments/da/appearance/20261013_ocbe_depthrank2_learnableorth_h128_s42"
Q_FIXED = np.array([[1 / math.sqrt(3)] * 3, [-1 / math.sqrt(2), 0, 1 / math.sqrt(2)]])


def subspace_angle(left: np.ndarray, right: np.ndarray) -> float:
    left_normal = np.cross(left[0], left[1]); left_normal /= np.linalg.norm(left_normal)
    right_normal = np.cross(right[0], right[1]); right_normal /= np.linalg.norm(right_normal)
    return math.degrees(math.acos(float(np.clip(abs(left_normal @ right_normal), 0, 1))))


def manifest_fingerprint(scene_dir: Path) -> str:
    train = sorted(scene_dir.glob("sequence*/train/*/*.jpg"))
    videos = sorted(p for p in scene_dir.glob("sequence*/test/*") if p.is_dir())
    assert train and videos
    lines = [f"{p.relative_to(scene_dir)}:{p.stat().st_size}" for p in train]
    for video in videos:
        frames = sorted(video.glob("*.jpg"), key=lambda p: int(p.stem))
        label = Path(str(video) + ".npy")
        assert frames and label.is_file()
        lines.extend(f"{p.relative_to(scene_dir)}:{p.stat().st_size}" for p in frames)
        lines.append(f"{label.relative_to(scene_dir)}:{hashlib.sha256(label.read_bytes()).hexdigest()}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def check_dataset_rows(scene_dir: Path, keys: list[tuple[str, str, str]]) -> None:
    labels_by_video = {}
    for video, frame, label in keys:
        path = scene_dir / video / frame
        if not path.is_file():
            raise ValueError(f"missing test frame: {path}")
        if video not in labels_by_video:
            labels_by_video[video] = np.asarray(np.load(str(scene_dir / video) + ".npy", allow_pickle=True)).astype(int).ravel()
        if int(labels_by_video[video][int(Path(frame).stem)]) != int(label):
            raise ValueError(f"test label mismatch: {path}")


def basis_measure(snapshot: dict) -> dict:
    raw, q = np.asarray(snapshot["raw_q"], dtype=float), np.asarray(snapshot["effective_q"], dtype=float)
    assert raw.shape == q.shape == (2, 3)
    angle = subspace_angle(q, Q_FIXED)
    distance = float(np.linalg.norm(q.T @ q - Q_FIXED.T @ Q_FIXED))
    orth_error = float(np.linalg.norm(q @ q.T - np.eye(2)))
    if abs(angle - snapshot["principal_angle_deg"]) > 1e-4:
        raise ValueError("stored basis angle differs from recomputation")
    if abs(distance - snapshot["projector_distance_fro"]) > 1e-6:
        raise ValueError("stored projector distance differs from recomputation")
    if abs(orth_error - snapshot["effective_orthogonality_error_fro"]) > 1e-6 or orth_error > 1e-5:
        raise ValueError("effective depth basis is not orthonormal")
    if abs(distance - math.sqrt(2) * math.sin(math.radians(angle))) > 1e-5:
        raise ValueError("angle/projector geometry mismatch")
    return {"principal_angle_deg": angle, "projector_distance_fro": distance,
            "orthogonality_error_fro": orth_error, "raw_q": raw.tolist(), "effective_q": q.tolist()}


def main() -> None:
    record = json.loads((EXPERIMENT / "record.json").read_text())
    run = record["runs"][0]
    launches, finishes, events = read_scheduler(EXPERIMENT / run["scheduler_log"])
    fixed_run = json.loads((FIXED / "record.json").read_text())["runs"][-1]
    fixed_launches, _, _ = read_scheduler(FIXED / fixed_run["scheduler_log"])
    config = json.loads((EXPERIMENT / "config.json").read_text())
    expected_params = config["predicted_trainable_parameters"]["total"]
    expected_init = np.asarray(config["ocbe"]["initial_q_effective"], dtype=float)
    if expected_init.shape != (2, 3) or subspace_angle(expected_init, Q_FIXED) < 10:
        raise ValueError("random initialization is absent or too close to fixed Q")
    data_root = Path(run["data_root"])
    rows = []
    for scene, scene_name in SCENES.items():
        metric = json.loads((EXPERIMENT / f"results/{scene}.json").read_text())
        old_metric = json.loads((FIXED / f"results/{scene}.json").read_text())
        learned_metric = json.loads((LEARNED_FIXED_INIT / f"results/{scene}.json").read_text())
        keys, labels, scores = read_scores(EXPERIMENT / f"results/{scene}_scores.csv")
        old_keys, old_labels, old_scores = read_scores(FIXED / f"results/{scene}_scores.csv")
        learned_keys, learned_labels, learned_scores = read_scores(LEARNED_FIXED_INIT / f"results/{scene}_scores.csv")
        if keys != old_keys or labels != old_labels or keys != learned_keys or labels != learned_labels:
            raise ValueError(f"paired frame keys/labels differ: {scene}")
        scene_dir = data_root / scene_name
        check_dataset_rows(scene_dir, keys)
        fingerprint = manifest_fingerprint(scene_dir)
        if any(fingerprint != item["split_fingerprint"] for item in (metric, old_metric, learned_metric)):
            raise ValueError(f"split fingerprint mismatch: {scene}")
        auc = roc_auc_score(labels, scores) * 100
        fixed_auc = roc_auc_score(old_labels, old_scores) * 100
        learned_auc = roc_auc_score(learned_labels, learned_scores) * 100
        if (abs(auc - metric["auc_percent"]) > 1e-8 or abs(fixed_auc - old_metric["auc_percent"]) > 1e-8
                or abs(learned_auc - learned_metric["auc_percent"]) > 1e-8):
            raise ValueError(f"stored AUC differs from recomputation: {scene}")
        if len(keys) != metric["frames"] or sum(labels) != metric["positive_frames"]:
            raise ValueError(f"frame counts differ: {scene}")
        if metric["seed"] != 42 or metric["trainable_parameters"] != expected_params:
            raise ValueError(f"seed or parameter count differs: {scene}")
        log = Path(launches[scene]["log"]).read_text().splitlines()
        epoch_lines = [line for line in log if line.startswith("epoch=")]
        if len(epoch_lines) != 15 or [int(line.split()[0].split("=")[1]) for line in epoch_lines] != list(range(1, 16)):
            raise ValueError(f"missing training epochs: {scene}")
        if abs(json.loads(log[-1])["auc_percent"] - auc) > 1e-8:
            raise ValueError(f"scene log final AUC differs: {scene}")
        fixed_log = Path(fixed_launches[scene]["log"]).read_text().splitlines()
        fixed_epoch15 = [line for line in fixed_log if line.startswith("epoch=15 ")]
        if len(fixed_epoch15) != 1:
            raise ValueError(f"fixed rank2 final epoch missing: {scene}")
        fixed_train_loss = float(fixed_epoch15[0].split("loss=")[1].split()[0])
        history = json.loads((EXPERIMENT / f"results/{scene}_basis.json").read_text())["history"]
        if len(history) != 16 or [step["epoch"] for step in history] != list(range(16)):
            raise ValueError(f"basis trajectory is incomplete: {scene}")
        measures = [basis_measure(step) for step in history]
        initial_q = np.asarray(history[0]["effective_q"], dtype=float)
        if np.linalg.norm(initial_q - expected_init) > 1e-6:
            raise ValueError(f"seeded random Q initialization differs: {scene}")
        learned_q = np.asarray(json.loads((LEARNED_FIXED_INIT / f"results/{scene}_basis.json").read_text())["history"][-1]["effective_q"])
        final = measures[-1]
        final_q = np.asarray(final["effective_q"])
        if abs(final["principal_angle_deg"] - metric["basis_principal_angle_deg"]) > 1e-4:
            raise ValueError(f"metric/basis angle differs: {scene}")
        if abs(final["projector_distance_fro"] - metric["basis_projector_distance_fro"]) > 1e-6:
            raise ValueError(f"metric/basis projector distance differs: {scene}")
        normal = [v for y, v in zip(labels, scores) if y == 0]
        abnormal = [v for y, v in zip(labels, scores) if y == 1]
        rows.append({"scene": scene, "auc_percent": auc, "fixed_auc_percent": fixed_auc,
                     "delta_vs_fixed_pp": auc - fixed_auc,
                     "learned_fixed_init_auc_percent": learned_auc,
                     "delta_vs_learned_fixed_init_pp": auc - learned_auc,
                     "frames": len(labels),
                     "positive_frames": sum(labels), "score_gap": mean(abnormal) - mean(normal),
                     "final_train_loss": history[-1]["train_loss"],
                     "fixed_final_train_loss": fixed_train_loss,
                     "delta_train_loss": history[-1]["train_loss"] - fixed_train_loss,
                     "principal_angle_deg": final["principal_angle_deg"],
                     "initial_angle_vs_fixed_deg": subspace_angle(initial_q, Q_FIXED),
                     "movement_from_random_init_deg": subspace_angle(final_q, initial_q),
                     "final_angle_vs_learned_fixed_init_deg": subspace_angle(final_q, learned_q),
                     "projector_distance_fro": final["projector_distance_fro"],
                     "orthogonality_error_fro": final["orthogonality_error_fro"],
                     "final_raw_q": final["raw_q"], "final_effective_q": final["effective_q"],
                     "split_fingerprint": fingerprint, "gpu": launches[scene]["gpu"],
                     "launch_time_utc": launches[scene]["time"], "finish_time_utc": finishes[scene]["time"]})
    macro = {"auc_percent": mean(row["auc_percent"] for row in rows),
             "fixed_auc_percent": mean(row["fixed_auc_percent"] for row in rows),
             "delta_vs_fixed_pp": mean(row["delta_vs_fixed_pp"] for row in rows),
             "learned_fixed_init_auc_percent": mean(row["learned_fixed_init_auc_percent"] for row in rows),
             "delta_vs_learned_fixed_init_pp": mean(row["delta_vs_learned_fixed_init_pp"] for row in rows),
             "mean_principal_angle_deg": mean(row["principal_angle_deg"] for row in rows),
             "mean_movement_from_random_init_deg": mean(row["movement_from_random_init_deg"] for row in rows),
             "max_orthogonality_error_fro": max(row["orthogonality_error_fro"] for row in rows),
             "trainable_parameters": expected_params, "fixed_trainable_parameters": expected_params - 6}
    output = EXPERIMENT / "results"
    with (output / "summary.csv").open("w", newline="") as file:
        columns = ["scene", "auc_percent", "fixed_auc_percent", "delta_vs_fixed_pp",
                   "learned_fixed_init_auc_percent", "delta_vs_learned_fixed_init_pp", "frames",
                   "positive_frames", "score_gap", "final_train_loss", "fixed_final_train_loss",
                   "delta_train_loss", "initial_angle_vs_fixed_deg", "principal_angle_deg",
                   "movement_from_random_init_deg", "final_angle_vs_learned_fixed_init_deg",
                   "projector_distance_fro", "orthogonality_error_fro", "gpu"]
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps({"run_id": run["run_id"],
        "method": "AUC recomputed from frame scores; test labels and manifest fingerprints checked against dataset; paired fixed-rank2 keys/labels checked; 15 epochs and scheduler checked; depth basis trajectory independently recomputed",
        "scenes": rows, "macro": macro,
        "scheduler": {"events": len(events), "launches": 7, "zero_return_finishes": 7,
                      "gpu_assignments": {scene: launches[scene]["gpu"] for scene in SCENES}}},
        ensure_ascii=False, indent=2) + "\n")
    lines = ["# 随机正交初始化 Q：七场景复核", "",
             "逐帧重算 AUC；核对测试帧标签和划分指纹，与固定 Q、固定初始化后可学习 Q 两个 seed42 对照逐帧配对。", "",
             "| 场景 | 随机初始化 AUC | 固定初始化可学习 AUC | Δ随机−固定初始化 | 固定 Q AUC | 最终偏离固定 Q | 从随机初值移动 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f'| {row["scene"]} | {row["auc_percent"]:.4f} | {row["learned_fixed_init_auc_percent"]:.4f} | {row["delta_vs_learned_fixed_init_pp"]:+.4f} | {row["fixed_auc_percent"]:.4f} | {row["principal_angle_deg"]:.3f}° | {row["movement_from_random_init_deg"]:.3f}° |')
    lines += [f'| **宏平均** | **{macro["auc_percent"]:.4f}** | **{macro["learned_fixed_init_auc_percent"]:.4f}** | **{macro["delta_vs_learned_fixed_init_pp"]:+.4f}** | **{macro["fixed_auc_percent"]:.4f}** | **{macro["mean_principal_angle_deg"]:.3f}°** | **{macro["mean_movement_from_random_init_deg"]:.3f}°** |',
              "", "## 观察与判读", "",
              f'相对固定初始化后可学习 Q，七场景宏平均 AUC 变化 **{macro["delta_vs_learned_fixed_init_pp"]:+.4f} 个百分点**；相对固定 Q 变化 **{macro["delta_vs_fixed_pp"]:+.4f} 个百分点**。初始随机子空间相对 Mean+Trend 的主夹角为 {rows[0]["initial_angle_vs_fixed_deg"]:.3f}°。最终相对固定 Q 平均主夹角 {macro["mean_principal_angle_deg"]:.3f}°，从随机初值平均移动 {macro["mean_movement_from_random_init_deg"]:.3f}°。',
              f'最终有效矩阵的最大正交误差为 {macro["max_orthogonality_error_fro"]:.2e}，这是每次前向 Gram-Schmidt 硬约束的结果，不能当作网络自行学会正交的证据。',
              "各场景完整原始/有效 Q、最终相对两个对照的子空间角见 [summary.json](summary.json)，逐 epoch 轨迹见 `*_basis.json`。", "",
              "## 限制", "",
              "仅一个随机初值与一个训练种子，不能判断随机初始化的方差。六个原始系数经过正交化且与下游融合层共同训练，无法把 AUC 变化因果归于深度方向本身。比较的是外观分支逐帧 AUC；参数数目不含冻结教师、行为分支和 KNN。", ""]
    (output / "summary.md").write_text("\n".join(lines))
    print(json.dumps(macro, ensure_ascii=False))


if __name__ == "__main__":
    main()
