#!/usr/bin/env python3
"""Audit the paired Q-retention runs and write seven-scene reports."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_ocbe import read_scores  # noqa: E402


BASE = ROOT / "experiments/da/appearance"
EXPERIMENTS = {
    "fixed_init": BASE / "20261014_depthretain_fixedinit_h128_s42",
    "random_init": BASE / "20261014_depthretain_randominit_h128_s42",
}
CONTROLS = {
    "old_fixed_q": BASE / "20261013_ocbe_depthrank2_hidden128_s42",
    "old_random_q": BASE / "20261013_ocbe_depthrank2_randomorth_h128_s42",
}
SCENES = {
    "bike": "Bike Roundabout", "crossroads": "Crossroads",
    "farmland": "Farmland Inspection", "highway": "Highway",
    "railway": "Railway Inspection", "solar": "Solar Panel Inspection",
    "vehicle": "Vehicle Roundabout",
}
SOURCE = "25fa78ea503f9bfc4ae80f92c5dff0663b73165e"
PROBE_SHA = "c0a652ac36b70b05d716e7138c2bfedb600315de7ca6457b0d8b6543d7a5257a"


def load(path: Path):
    return json.loads(path.read_text())


def angle(left: np.ndarray, right: np.ndarray) -> float:
    a = np.cross(left[0], left[1]); a /= np.linalg.norm(a)
    b = np.cross(right[0], right[1]); b /= np.linalg.norm(b)
    return math.degrees(math.acos(float(np.clip(abs(a @ b), 0, 1))))


def energy(q: np.ndarray, gram: np.ndarray) -> float:
    return float(np.trace(q @ gram @ q.T) / np.trace(gram))


def target_basis(gram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, vectors = np.linalg.eigh(gram)
    return eigenvalues, vectors[:, 1:].T


def manifest_fingerprint(scene_dir: Path) -> str:
    train = sorted(scene_dir.glob("sequence*/train/*/*.jpg"))
    videos = sorted(p for p in scene_dir.glob("sequence*/test/*") if p.is_dir())
    if not train or not videos:
        raise ValueError(f"missing train/test data: {scene_dir}")
    lines = [f"{p.relative_to(scene_dir)}:{p.stat().st_size}" for p in train]
    for video in videos:
        frames = sorted(video.glob("*.jpg"), key=lambda p: int(p.stem))
        label = Path(str(video) + ".npy")
        if not frames or not label.is_file():
            raise ValueError(f"missing frames/labels: {video}")
        lines.extend(f"{p.relative_to(scene_dir)}:{p.stat().st_size}" for p in frames)
        lines.append(f"{label.relative_to(scene_dir)}:{hashlib.sha256(label.read_bytes()).hexdigest()}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def check_frame_labels(scene_dir: Path, keys: list[tuple[str, str, str]]) -> None:
    by_video: dict[str, list[tuple[str, str]]] = {}
    for video, frame, label in keys:
        by_video.setdefault(video, []).append((frame, label))
    for video, entries in by_video.items():
        directory = scene_dir / video
        frames = sorted(directory.glob("*.jpg"), key=lambda p: int(p.stem))
        labels = np.asarray(np.load(str(directory) + ".npy", allow_pickle=True)).astype(int).ravel()
        if len(frames) != len(entries) or len(labels) != len(entries):
            raise ValueError(f"frame/label count mismatch: {directory}")
        for path, (frame, label), expected in zip(frames, entries, labels):
            if path.name != frame or int(label) != int(expected):
                raise ValueError(f"frame/label mismatch: {path}")


def check_scheduler() -> tuple[dict, dict, list]:
    records = {kind: load(exp / "record.json") for kind, exp in EXPERIMENTS.items()}
    for kind, record in records.items():
        if record["source_commit"] != SOURCE or record["runs"][0]["source_commit"] != SOURCE:
            raise ValueError(f"wrong source snapshot: {kind}")
    paths = [exp / records[kind]["runs"][0]["scheduler_log"] for kind, exp in EXPERIMENTS.items()]
    if paths[0].read_bytes() != paths[1].read_bytes():
        raise ValueError("paired scheduler logs differ")
    events = [json.loads(line) for line in paths[0].read_text().splitlines()]
    if events[-1]["kind"] != "all_finished" or events[-1]["failures"]:
        raise ValueError("paired scheduler did not finish cleanly")
    launches = {(e["scene"], e["variant"]): e for e in events if e["kind"] == "launch"}
    finishes = {(e["scene"], e["variant"]): e for e in events if e["kind"] == "finish"}
    expected = {(scene, variant) for scene in SCENES for variant in EXPERIMENTS}
    if set(launches) != set(finishes) or set(launches) != expected or len(launches) != 14:
        raise ValueError("scheduler scene/variant set is incomplete")
    active = {}
    for event in events:
        if event["kind"] == "launch":
            if event["gpu"] in active:
                raise ValueError("GPU overlap in scheduler")
            active[event["gpu"]] = (event["scene"], event["variant"])
        elif event["kind"] == "finish":
            if active.pop(event["gpu"], None) != (event["scene"], event["variant"]):
                raise ValueError("scheduler finish does not match launch")
            key = (event["scene"], event["variant"])
            if event["returncode"] != 0 or event["pid"] != launches[key]["pid"]:
                raise ValueError(f"failed/mismatched process: {key}")
    if active:
        raise ValueError("scheduler still has active jobs")
    for scene in SCENES:
        fixed = launches[(scene, "fixed_init")]
        random = launches[(scene, "random_init")]
        if fixed["gpu"] != random["gpu"] or finishes[(scene, "fixed_init")]["time"] > random["time"]:
            raise ValueError(f"scene pair did not run sequentially on same GPU: {scene}")
    return launches, finishes, events


def check_basis(path: Path, metric: dict, pretrain: dict, variant: str) -> dict:
    basis = load(path)
    history = basis["history"]
    if len(history) != 16 or [x["epoch"] for x in history] != list(range(16)):
        raise ValueError(f"incomplete Q trajectory: {path}")
    steps = pretrain["history"]
    if [x["epoch"] for x in steps] != [0, 25, 50, 100, 200]:
        raise ValueError(f"incomplete Q pretraining: {path}")
    if pretrain["probe_sha256"] != PROBE_SHA or basis["retention_probe_sha256"] != PROBE_SHA:
        raise ValueError(f"probe hash mismatch: {path}")
    q0 = np.asarray(steps[0]["effective_q"], dtype=float)
    q_pre = np.asarray(steps[-1]["effective_q"], dtype=float)
    q_start = np.asarray(history[0]["effective_q"], dtype=float)
    q_final = np.asarray(history[-1]["effective_q"], dtype=float)
    if not np.allclose(q_pre, q_start, atol=1e-6):
        raise ValueError(f"Q changed between pretraining and joint training: {path}")
    if variant == "fixed_init":
        expected = np.array([[1 / math.sqrt(3)] * 3,
                             [-1 / math.sqrt(2), 0, 1 / math.sqrt(2)]])
        if not np.allclose(q0, expected, atol=1e-6):
            raise ValueError(f"fixed Q initialization mismatch: {path}")
    elif np.linalg.norm(q0 - np.asarray(load(CONTROLS["old_random_q"] / "results" / (metric["scene"] + "_basis.json"))["history"][0]["effective_q"])) > 1e-6:
        raise ValueError(f"random Q initialization mismatch: {path}")
    probe_gram = np.asarray(pretrain["depth_gram_mean_four_groups"], dtype=float)
    full_gram = np.asarray(basis["final_train_gram"], dtype=float)
    eigenvalues, optimum = target_basis(full_gram)
    if eigenvalues[1] <= eigenvalues[0] or eigenvalues[0] < 0:
        raise ValueError(f"invalid training Gram spectrum: {path}")
    for snapshot in history:
        q = np.asarray(snapshot["effective_q"], dtype=float)
        if q.shape != (2, 3) or np.linalg.norm(q @ q.T - np.eye(2)) > 1e-5:
            raise ValueError(f"Q is not row-orthonormal: {path}")
    final_angle = angle(q_final, optimum)
    final_energy = energy(q_final, full_gram)
    if abs(final_angle - metric["basis_angle_to_retention_optimum_deg"]) > 1e-4:
        raise ValueError(f"Q optimum angle mismatch: {path}")
    if abs(final_energy - metric["retained_normal_energy_fraction"]) > 1e-6:
        raise ValueError(f"Q retained energy mismatch: {path}")
    if abs(energy(q_pre, probe_gram) - steps[-1]["retained_normal_energy_fraction"]) > 1e-6:
        raise ValueError(f"pretrained energy mismatch: {path}")
    if abs(angle(q0, target_basis(probe_gram)[1]) - steps[0]["angle_to_retention_optimum_deg"]) > 1e-4:
        raise ValueError(f"initial probe angle mismatch: {path}")
    return {
        "initial_q": q0.tolist(), "pretrained_q": q_pre.tolist(), "final_q": q_final.tolist(),
        "final_raw_q": history[-1]["raw_q"], "final_train_gram": full_gram.tolist(),
        "initial_to_probe_optimum_deg": steps[0]["angle_to_retention_optimum_deg"],
        "pretrained_to_probe_optimum_deg": steps[-1]["angle_to_retention_optimum_deg"],
        "joint_start_to_final_deg": angle(q_pre, q_final),
        "final_to_full_train_optimum_deg": final_angle,
        "initial_probe_energy": steps[0]["retained_normal_energy_fraction"],
        "pretrained_probe_energy": steps[-1]["retained_normal_energy_fraction"],
        "final_train_energy": final_energy,
        "optimal_train_energy": float((eigenvalues[1] + eigenvalues[2]) / sum(eigenvalues)),
        "full_train_eigenvalues_ascending": eigenvalues.tolist(),
        "final_orthogonality_error_fro": float(np.linalg.norm(q_final @ q_final.T - np.eye(2))),
        "final_distill_train_loss": history[-1]["distill_train_loss"],
        "final_retention_train_loss": history[-1]["retention_loss"],
    }


def main() -> None:
    launches, finishes, events = check_scheduler()
    records = {variant: load(exp / "record.json") for variant, exp in EXPERIMENTS.items()}
    data_root = Path(records["fixed_init"]["runs"][0]["data_root"])
    rows = []
    for scene, scene_name in SCENES.items():
        fingerprint = manifest_fingerprint(data_root / scene_name)
        metrics, bases, score_sets = {}, {}, {}
        for name, exp in {**CONTROLS, **EXPERIMENTS}.items():
            metric = load(exp / "results" / f"{scene}.json")
            keys, labels, scores = read_scores(exp / "results" / f"{scene}_scores.csv")
            auc = roc_auc_score(labels, scores) * 100
            if abs(auc - metric["auc_percent"]) > 1e-8 or fingerprint != metric["split_fingerprint"]:
                raise ValueError(f"AUC or split fingerprint mismatch: {scene}/{name}")
            if len(labels) != metric["frames"] or sum(labels) != metric["positive_frames"]:
                raise ValueError(f"frame/label count mismatch: {scene}/{name}")
            metrics[name] = metric
            score_sets[name] = (keys, labels, scores)
        keys, labels, _ = score_sets["fixed_init"]
        check_frame_labels(data_root / scene_name, keys)
        for name, (other_keys, other_labels, _) in score_sets.items():
            if keys != other_keys or labels != other_labels:
                raise ValueError(f"paired test keys/labels mismatch: {scene}/{name}")
        for variant, exp in EXPERIMENTS.items():
            metric = metrics[variant]
            if metric["seed"] != 42 or metric["trainable_parameters"] != 3761222 or metric["retention_probe_sha256"] != PROBE_SHA:
                raise ValueError(f"seed/parameters/probe mismatch: {scene}/{variant}")
            log = Path(launches[(scene, variant)]["log"]).read_text().splitlines()
            epoch_lines = [line for line in log if line.startswith("epoch=")]
            if [int(line.split()[0].split("=")[1]) for line in epoch_lines] != list(range(1, 16)):
                raise ValueError(f"epoch log incomplete: {scene}/{variant}")
            if len([line for line in log if '"phase": "q_pretrain_done"' in line]) != 1:
                raise ValueError(f"pretrain log incomplete: {scene}/{variant}")
            if abs(json.loads(log[-1])["auc_percent"] - metric["auc_percent"]) > 1e-8:
                raise ValueError(f"final log metric mismatch: {scene}/{variant}")
            bases[variant] = check_basis(exp / "results" / f"{scene}_basis.json", metric,
                                         load(exp / "results" / f"{scene}_pretrain.json"), variant)
        q_fixed = np.asarray(bases["fixed_init"]["final_q"])
        q_random = np.asarray(bases["random_init"]["final_q"])
        overlap = q_fixed @ q_random.T
        projector_distance = float(np.linalg.norm(q_fixed.T @ q_fixed - q_random.T @ q_random))
        pair_angle = angle(q_fixed, q_random)
        if abs(projector_distance - math.sqrt(2) * math.sin(math.radians(pair_angle))) > 1e-5:
            raise ValueError(f"Q angle/projector geometry mismatch: {scene}")
        row = {
            "scene": scene, "frames": len(labels), "positive_frames": sum(labels),
            "split_fingerprint": fingerprint,
            "old_fixed_q_auc_percent": metrics["old_fixed_q"]["auc_percent"],
            "old_random_q_auc_percent": metrics["old_random_q"]["auc_percent"],
            "fixed_init_auc_percent": metrics["fixed_init"]["auc_percent"],
            "random_init_auc_percent": metrics["random_init"]["auc_percent"],
            "fixed_delta_vs_old_fixed_pp": metrics["fixed_init"]["auc_percent"] - metrics["old_fixed_q"]["auc_percent"],
            "random_delta_vs_old_random_pp": metrics["random_init"]["auc_percent"] - metrics["old_random_q"]["auc_percent"],
            "random_minus_fixed_auc_pp": metrics["random_init"]["auc_percent"] - metrics["fixed_init"]["auc_percent"],
            "final_plane_angle_deg": pair_angle, "final_projector_distance_fro": projector_distance,
            "final_effective_q_row_difference_fro": float(np.linalg.norm(q_fixed - q_random)),
            "final_effective_q_row_overlap": overlap.tolist(),
            "same_gpu": launches[(scene, "fixed_init")]["gpu"],
            "fixed_init_basis": bases["fixed_init"], "random_init_basis": bases["random_init"],
            "fixed_launch_utc": launches[(scene, "fixed_init")]["time"],
            "random_finish_utc": finishes[(scene, "random_init")]["time"],
        }
        rows.append(row)
    macro = {name: mean(row[f"{name}_auc_percent"] for row in rows)
             for name in ("old_fixed_q", "old_random_q", "fixed_init", "random_init")}
    macro.update({
        "fixed_delta_vs_old_fixed_pp": macro["fixed_init"] - macro["old_fixed_q"],
        "random_delta_vs_old_random_pp": macro["random_init"] - macro["old_random_q"],
        "random_minus_fixed_auc_pp": macro["random_init"] - macro["fixed_init"],
        "mean_abs_pair_auc_gap_pp": mean(abs(row["random_minus_fixed_auc_pp"]) for row in rows),
        "mean_final_plane_angle_deg": mean(row["final_plane_angle_deg"] for row in rows),
        "max_final_plane_angle_deg": max(row["final_plane_angle_deg"] for row in rows),
        "all_pair_planes_within_5_deg": all(row["final_plane_angle_deg"] <= 5 for row in rows),
        "mean_final_train_energy_fixed": mean(row["fixed_init_basis"]["final_train_energy"] for row in rows),
        "mean_final_train_energy_random": mean(row["random_init_basis"]["final_train_energy"] for row in rows),
        "mean_pretrain_initial_angle_fixed_deg": mean(row["fixed_init_basis"]["initial_to_probe_optimum_deg"] for row in rows),
        "mean_pretrain_initial_angle_random_deg": mean(row["random_init_basis"]["initial_to_probe_optimum_deg"] for row in rows),
        "max_pretrained_to_probe_optimum_deg": max(row[f"{variant}_basis"]["pretrained_to_probe_optimum_deg"]
                                             for row in rows for variant in EXPERIMENTS),
        "mean_joint_q_movement_fixed_deg": mean(row["fixed_init_basis"]["joint_start_to_final_deg"] for row in rows),
        "mean_joint_q_movement_random_deg": mean(row["random_init_basis"]["joint_start_to_final_deg"] for row in rows),
        "random_delta_vs_old_random_excluding_vehicle_pp": mean(
            row["random_delta_vs_old_random_pp"] for row in rows if row["scene"] != "vehicle"),
        "trainable_parameters": 3761222,
    })
    report = {
        "run_ids": {variant: records[variant]["runs"][0]["run_id"] for variant in EXPERIMENTS},
        "source_commit": SOURCE,
        "method": "14/14 scheduler and 15-epoch logs checked; frame labels and split manifests verified against dataset; four AUC variants recomputed from paired per-frame CSVs; Q geometry and retained teacher energy independently recalculated",
        "scenes": rows, "macro": macro,
        "scheduler": {"events": len(events), "launches": 14, "zero_return_finishes": 14,
                      "same_gpu_scene_pairs": 7, "final_time_utc": events[-1]["time"]},
    }
    columns = ["scene", "frames", "positive_frames", "old_fixed_q_auc_percent", "old_random_q_auc_percent",
               "fixed_init_auc_percent", "random_init_auc_percent", "fixed_delta_vs_old_fixed_pp",
               "random_delta_vs_old_random_pp", "random_minus_fixed_auc_pp", "final_plane_angle_deg",
               "final_projector_distance_fro", "final_effective_q_row_difference_fro", "same_gpu"]
    lines = ["# Q 深度信息保留：固定与随机初始化的七场景配对复核", "",
             "14/14 次训练返回码为 0；每次均完成 15 个 epoch。逐帧重算 ROC AUC×100，核对测试帧键、原始标签、数据划分指纹、参数数目和 Q 正交性。七对实验在相同逻辑 GPU 上先后运行。", "",
             "| 场景 | 旧固定 Q | 旧随机 Q | 新固定初值 | 新随机初值 | 新 Q 平面夹角(°) | Δ新固定−旧固定 | Δ新随机−旧随机 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f'| {row["scene"]} | {row["old_fixed_q_auc_percent"]:.4f} | {row["old_random_q_auc_percent"]:.4f} | {row["fixed_init_auc_percent"]:.4f} | {row["random_init_auc_percent"]:.4f} | {row["final_plane_angle_deg"]:.4f} | {row["fixed_delta_vs_old_fixed_pp"]:+.4f} | {row["random_delta_vs_old_random_pp"]:+.4f} |')
    lines.extend([
        f'| **宏平均** | **{macro["old_fixed_q"]:.4f}** | **{macro["old_random_q"]:.4f}** | **{macro["fixed_init"]:.4f}** | **{macro["random_init"]:.4f}** | **{macro["mean_final_plane_angle_deg"]:.4f}** | **{macro["fixed_delta_vs_old_fixed_pp"]:+.4f}** | **{macro["random_delta_vs_old_random_pp"]:+.4f}** |',
        "", "## Q 的方向与可辨识性", "",
        f'Q 单独预训练前，固定初值离本场景正常教师特征的最优二维平面平均 {macro["mean_pretrain_initial_angle_fixed_deg"]:.3f}°，随机初值平均 {macro["mean_pretrain_initial_angle_random_deg"]:.3f}°；200 步后 14 次运行中最远仅 {macro["max_pretrained_to_probe_optimum_deg"]:.4f}°。联合训练后，七对 Q 投影平面平均夹角 {macro["mean_final_plane_angle_deg"]:.4f}°，最大 {macro["max_final_plane_angle_deg"]:.4f}°，均低于预设 5° 判据。',
        f'最终固定/随机初值 Q 对全训练 batch 教师特征的平均能量保留率分别为 {macro["mean_final_train_energy_fixed"]*100:.3f}% / {macro["mean_final_train_energy_random"]*100:.3f}%。联合蒸馏阶段 Q 相对预训练终点仅移动 {macro["mean_joint_q_movement_fixed_deg"]:.3f}° / {macro["mean_joint_q_movement_random_deg"]:.3f}°；方向主要由正常特征能量目标决定。这说明两个起点学到了同一保留**子空间**；但同一平面有无穷多个正交行基，六个 Q 系数不必逐个相等。七场景两套有效 Q 的逐行 Frobenius 距离均约为 2，主要是平面内的旋转/反射。',
        "", "## AUC 结论与解释边界", "",
        f'新固定初值宏平均 {macro["fixed_init"]:.4f}，比旧固定 Q {macro["fixed_delta_vs_old_fixed_pp"]:+.4f} 个百分点；新随机初值宏平均 {macro["random_init"]:.4f}，比旧随机 Q {macro["random_delta_vs_old_random_pp"]:+.4f} 个百分点。随机版下降主要来自 vehicle（{next(row for row in rows if row["scene"] == "vehicle")["random_delta_vs_old_random_pp"]:+.4f} 个百分点）；其余六场景对旧随机 Q 的平均变化仅 {macro["random_delta_vs_old_random_excluding_vehicle_pp"]:+.4f} 个百分点。两套新模型虽保留同一个 Q 平面，逐场景 AUC 仍有差异，平均绝对差 {macro["mean_abs_pair_auc_gap_pp"]:.4f} 个百分点。',
        "正常教师特征能量提供了可辨识的 Q 方向，但不是异常 AUC 的代理。相同行空间的两套 Q 采用不同的行坐标；OCBE 融合层以相同初始权重接收不同坐标，随后与学生共同优化，可能形成不同轨迹。这是机制推断，当前实验没有单独隔离行坐标与下游优化的作用。",
        "", "## 限制与后续判据", "",
        "仅 seed42 和一个随机 Q 起点；Q 单独预训练用每场景 16 张正常帧，联合训练每 batch 使用实际正常训练帧。能量保留目标和蒸馏 loss 都不能保证异常 AUC。该正交能量目标的最优子空间可直接由 Gram 特征分解得到，因此 Q 的收敛不等于发现了 AUC 最优方向。若继续研究 Q，应先固定行基规范或同步旋转 OCBE 第一层权重，再做多种子对照；若目标直接是 AUC，本次结果不支持以该保留能量约束替代原方案。参数量均为 OCBE＋学生 3,761,222，不含冻结教师、行为分支或 KNN。", "",
    ])
    for exp in EXPERIMENTS.values():
        out = exp / "results"
        (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        with (out / "summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows({name: row[name] for name in columns} for row in rows)
        (out / "summary.md").write_text("\n".join(lines))
    print(json.dumps(macro, ensure_ascii=False))


if __name__ == "__main__":
    main()
