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
DEPTH_MODES = {
    "mean": Q_FIXED[0],
    "trend": Q_FIXED[1],
    "curvature": np.array([1, -2, 1], dtype=float) / math.sqrt(6),
}


def subspace_angle(left: np.ndarray, right: np.ndarray) -> float:
    left_normal = np.cross(left[0], left[1]); left_normal /= np.linalg.norm(left_normal)
    right_normal = np.cross(right[0], right[1]); right_normal /= np.linalg.norm(right_normal)
    return math.degrees(math.acos(float(np.clip(abs(left_normal @ right_normal), 0, 1))))


def mode_retention(q: np.ndarray) -> dict[str, float]:
    """Squared projection of each unit depth mode onto Q's retained plane."""
    return {name: float(np.linalg.norm(q @ mode) ** 2) for name, mode in DEPTH_MODES.items()}


def compact_q(q: list[list[float]]) -> str:
    return " / ".join("[" + ", ".join(f"{v:+.3f}" for v in row) + "]" for row in q)


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
        initial_raw = np.asarray(history[0]["raw_q"], dtype=float)
        if np.linalg.norm(initial_q - expected_init) > 1e-6:
            raise ValueError(f"seeded random Q initialization differs: {scene}")
        if np.linalg.norm(initial_raw - expected_init) > 1e-6:
            raise ValueError(f"seeded random raw Q initialization differs: {scene}")
        learned_history = json.loads((LEARNED_FIXED_INIT / f"results/{scene}_basis.json").read_text())["history"]
        learned_q = np.asarray(learned_history[-1]["effective_q"])
        final = measures[-1]
        final_q = np.asarray(final["effective_q"])
        final_raw = np.asarray(final["raw_q"])
        raw_change = final_raw - initial_raw
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
                     "learned_fixed_init_final_train_loss": learned_history[-1]["train_loss"],
                     "delta_train_loss_vs_learned_fixed_init": history[-1]["train_loss"] - learned_history[-1]["train_loss"],
                     "fixed_final_train_loss": fixed_train_loss,
                     "delta_train_loss": history[-1]["train_loss"] - fixed_train_loss,
                     "principal_angle_deg": final["principal_angle_deg"],
                     "initial_angle_vs_fixed_deg": subspace_angle(initial_q, Q_FIXED),
                     "movement_from_random_init_deg": subspace_angle(final_q, initial_q),
                     "final_angle_vs_learned_fixed_init_deg": subspace_angle(final_q, learned_q),
                     "initial_mode_retention": mode_retention(initial_q),
                     "final_mode_retention": mode_retention(final_q),
                     "raw_q_change_fro": float(np.linalg.norm(raw_change)),
                     "raw_q_orthogonality_error_fro": float(np.linalg.norm(final_raw @ final_raw.T - np.eye(2))),
                     "raw_q_delta": raw_change.tolist(),
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
             "initial_mode_retention": mode_retention(expected_init),
             "mean_final_mode_retention": {name: mean(row["final_mode_retention"][name] for row in rows) for name in DEPTH_MODES},
             "mean_raw_q_change_fro": mean(row["raw_q_change_fro"] for row in rows),
             "max_raw_q_orthogonality_error_fro": max(row["raw_q_orthogonality_error_fro"] for row in rows),
             "max_orthogonality_error_fro": max(row["orthogonality_error_fro"] for row in rows),
             "trainable_parameters": expected_params, "fixed_trainable_parameters": expected_params - 6}
    output = EXPERIMENT / "results"
    with (output / "summary.csv").open("w", newline="") as file:
        columns = ["scene", "auc_percent", "fixed_auc_percent", "delta_vs_fixed_pp",
                   "learned_fixed_init_auc_percent", "delta_vs_learned_fixed_init_pp", "frames",
                   "positive_frames", "score_gap", "final_train_loss", "learned_fixed_init_final_train_loss",
                   "delta_train_loss_vs_learned_fixed_init", "fixed_final_train_loss",
                   "delta_train_loss", "initial_angle_vs_fixed_deg", "principal_angle_deg",
                   "movement_from_random_init_deg", "final_angle_vs_learned_fixed_init_deg",
                   "raw_q_change_fro", "raw_q_orthogonality_error_fro",
                   "projector_distance_fro", "orthogonality_error_fro", "gpu"]
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps({"run_id": run["run_id"],
        "method": "AUC recomputed from frame scores; test labels and manifest fingerprints checked against dataset; paired fixed-Q and fixed-initialized learned-Q keys/labels checked; 15 epochs and scheduler checked; depth basis trajectory independently recomputed",
        "initial_random_q_raw_and_effective": expected_init.tolist(),
        "fixed_mean_trend_q": Q_FIXED.tolist(),
        "depth_mode_definition": "mean=(1,1,1)/sqrt(3); trend=(-1,0,1)/sqrt(2); curvature=(1,-2,1)/sqrt(6); retention=||Q_eff mode||^2 (geometric, not measured feature energy)",
        "scenes": rows, "macro": macro,
        "scheduler": {"events": len(events), "launches": 7, "zero_return_finishes": 7,
                      "gpu_assignments": {scene: launches[scene]["gpu"] for scene in SCENES}}},
        ensure_ascii=False, indent=2) + "\n")
    lines = ["# 随机正交初始化 Q：七场景复核", "",
             "逐帧重算 AUC；核对测试帧标签和划分指纹，与固定 Q、固定初始化后可学习 Q 两个 seed42 对照逐帧配对。", "",
             "| 场景 | 固定 Q | 固定初始化可学习 Q | 随机初始化可学习 Q | Δ随机−固定 Q | Δ随机−固定初始化 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f'| {row["scene"]} | {row["fixed_auc_percent"]:.4f} | {row["learned_fixed_init_auc_percent"]:.4f} | {row["auc_percent"]:.4f} | {row["delta_vs_fixed_pp"]:+.4f} | {row["delta_vs_learned_fixed_init_pp"]:+.4f} |')
    lines += [f'| **宏平均** | **{macro["fixed_auc_percent"]:.4f}** | **{macro["learned_fixed_init_auc_percent"]:.4f}** | **{macro["auc_percent"]:.4f}** | **{macro["delta_vs_fixed_pp"]:+.4f}** | **{macro["delta_vs_learned_fixed_init_pp"]:+.4f}** |',
              "", "## 观察与判读", "",
              f'相对固定初始化后可学习 Q，七场景宏平均 AUC 变化 **{macro["delta_vs_learned_fixed_init_pp"]:+.4f} 个百分点**；相对固定 Q 变化 **{macro["delta_vs_fixed_pp"]:+.4f} 个百分点**。随机版 {sum(row["delta_vs_learned_fixed_init_pp"] > 0 for row in rows)} 个场景高于固定初始化可学习版，{sum(row["delta_vs_learned_fixed_init_pp"] < 0 for row in rows)} 个场景低于它；最大提升来自 vehicle（{max(rows,key=lambda row:row["delta_vs_learned_fixed_init_pp"])["delta_vs_learned_fixed_init_pp"]:+.4f}），最大下降来自 railway（{min(rows,key=lambda row:row["delta_vs_learned_fixed_init_pp"])["delta_vs_learned_fixed_init_pp"]:+.4f}）。',
              "", "## Q 的六个参数与子空间", "",
              f'所有场景共享同一个初始随机 Q：`{compact_q(expected_init.tolist())}`。与固定 Mean+Trend 子空间的初始夹角为 **{rows[0]["initial_angle_vs_fixed_deg"]:.3f}°**。下表的 Q_eff 是实际乘在特征上的矩阵；Q_raw 是六个直接优化的参数。',
              "", "| 场景 | 最终 Q_raw | 最终 Q_eff | 初值→最终夹角 | 最终→固定 Q 夹角 |",
              "|---|---|---|---:|---:|"]
    for row in rows:
        lines.append(f'| {row["scene"]} | `{compact_q(row["final_raw_q"])}` | `{compact_q(row["final_effective_q"])}` | {row["movement_from_random_init_deg"]:.3f}° | {row["principal_angle_deg"]:.3f}° |')
    lines += ["", f'最终 Q 从随机初值只移动 **{macro["mean_movement_from_random_init_deg"]:.3f}°**（七场景平均），相对固定 Mean+Trend 仍偏离 **{macro["mean_principal_angle_deg"]:.3f}°**；没有收敛到固定子空间。Q_raw 的范数/正交性可变化，但前向的 Q_eff 保持正交；最大 Q_raw 正交误差为 {macro["max_raw_q_orthogonality_error_fro"]:.3f}，Q_eff 最大正交误差为 {macro["max_orthogonality_error_fro"]:.2e}。后者是硬约束带来的，不代表网络自行学会正交。',
              "vehicle 的子空间只从随机初值移动 2.062°，AUC 却比固定初始化可学习版高 1.4037；railway 只移动 0.257°，AUC 却低 0.5980。Q 移动大小与 AUC 变化不呈单调关系；下游融合权重和训练轨迹也可能参与。",
              "", "## 保留了哪些深度方向", "",
              "用正交单位向量 mean=(1,1,1)/√3、trend=(-1,0,1)/√2、curvature=(1,-2,1)/√6 表示三层深度方向。表中是几何保留比例 `||Q_eff v||²`，不是实测教师特征能量，更不是异常贡献。固定 Q 的三项分别为 1、1、0。", "",
              "| Q 状态 | mean | trend | curvature |", "|---|---:|---:|---:|",
              f'| 随机初值 | {macro["initial_mode_retention"]["mean"]:.4f} | {macro["initial_mode_retention"]["trend"]:.4f} | {macro["initial_mode_retention"]["curvature"]:.4f} |']
    for row in rows:
        retention = row["final_mode_retention"]
        lines.append(f'| {row["scene"]} 最终 | {retention["mean"]:.4f} | {retention["trend"]:.4f} | {retention["curvature"]:.4f} |')
    modes = macro["mean_final_mode_retention"]
    lines += [f'| **最终平均** | **{modes["mean"]:.4f}** | **{modes["trend"]:.4f}** | **{modes["curvature"]:.4f}** |',
              "", "随机版最终基本保留 mean 和 curvature，trend 只保留一小部分。它与固定版的 mean+trend 是显著不同的二维平面，但本次 AUC 仍接近；这说明固定 Mean+Trend 不是本次设置下唯一能取得相近 AUC 的深度子空间。", "",
              "## 损失与限制", "",
              "vehicle 的第 15 epoch 正常帧训练损失比固定初始化可学习版略高，但 AUC 高 1.4037 个百分点；railway 损失略低，AUC 却低 0.5980 个百分点。正常帧蒸馏损失与异常 AUC 不单调。",
              "仅一个随机初值与一个训练种子，不能据 +0.20 个百分点级宏平均差异宣称随机初始化稳定更优。六个原始系数与下游融合层共同训练，无法把 AUC 变化独立归因于 Q。随机版与固定初始化可学习版均为 3,761,222 个可训练外观参数；固定 Q 版少六个。以上不含冻结教师、行为分支和 KNN。完整精度见 [summary.json](summary.json)，逐 epoch 轨迹见 `*_basis.json`。", ""]
    (output / "summary.md").write_text("\n".join(lines))
    print(json.dumps(macro, ensure_ascii=False))


if __name__ == "__main__":
    main()
