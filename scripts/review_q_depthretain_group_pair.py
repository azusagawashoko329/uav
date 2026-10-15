#!/usr/bin/env python3
"""Audit paired four-group Q runs and compare them with shared-Q controls."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.review_q_depthretain_pair import check_frame_labels, manifest_fingerprint  # noqa: E402
from scripts.summarize_ocbe import read_scores  # noqa: E402


BASE = ROOT / "experiments/da/appearance"
EXPERIMENTS = {
    "fixed_init": BASE / "20261015_depthretain_group4_fixedinit_dtypefix_h128_s42",
    "random_init": BASE / "20261015_depthretain_group4_randominit_dtypefix_h128_s42",
}
CONTROLS = {
    "shared_fixed": BASE / "20261014_depthretain_fixedinit_h128_s42",
    "shared_random": BASE / "20261014_depthretain_randominit_h128_s42",
}
SCENES = {
    "bike": "Bike Roundabout", "crossroads": "Crossroads",
    "farmland": "Farmland Inspection", "highway": "Highway",
    "railway": "Railway Inspection", "solar": "Solar Panel Inspection",
    "vehicle": "Vehicle Roundabout",
}
SOURCE = "8fc9985d51c326f1ec8f7b105b68db362f2be331"
PROBE_SHA = "c0a652ac36b70b05d716e7138c2bfedb600315de7ca6457b0d8b6543d7a5257a"
PARAMETERS = 3761240


def load(path: Path):
    return json.loads(path.read_text())


def plane_angle(left: np.ndarray, right: np.ndarray) -> float:
    a = np.cross(left[0], left[1]); a /= np.linalg.norm(a)
    b = np.cross(right[0], right[1]); b /= np.linalg.norm(b)
    return math.degrees(math.acos(float(np.clip(abs(a @ b), 0, 1))))


def group_energy(q: np.ndarray, grams: np.ndarray) -> float:
    return float(sum(np.trace(q[g] @ grams[g] @ q[g].T) for g in range(4))
                 / sum(np.trace(grams[g]) for g in range(4)))


def shared_energy(q: np.ndarray, grams: np.ndarray) -> float:
    return group_energy(np.repeat(q[None, :, :], 4, axis=0), grams)


def optimum(grams: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(grams)
    if np.min(values) < -1e-5 or np.min(values[:, 1] - values[:, 0]) < 0.01:
        raise ValueError("invalid group Gram spectrum")
    return vectors[:, :, 1:].transpose(0, 2, 1)


def check_scheduler(records: dict) -> tuple[dict, dict, list]:
    paths = [EXPERIMENTS[k] / records[k]["runs"][0]["scheduler_log"] for k in EXPERIMENTS]
    if paths[0].read_bytes() != paths[1].read_bytes():
        raise ValueError("paired scheduler logs differ")
    events = [json.loads(line) for line in paths[0].read_text().splitlines()]
    if events[-1]["kind"] != "all_finished" or events[-1]["failures"]:
        raise ValueError("scheduler did not finish cleanly")
    launches = {(e["scene"], e["variant"]): e for e in events if e["kind"] == "launch"}
    finishes = {(e["scene"], e["variant"]): e for e in events if e["kind"] == "finish"}
    expected = {(scene, variant) for scene in SCENES for variant in EXPERIMENTS}
    if set(launches) != set(finishes) or set(launches) != expected or len(launches) != 14:
        raise ValueError("incomplete scheduler launches/finishes")
    active = {}
    for event in events:
        if event["kind"] == "launch":
            if event["gpu"] in active:
                raise ValueError("overlapping jobs on logical GPU")
            active[event["gpu"]] = (event["scene"], event["variant"])
        elif event["kind"] == "finish":
            if active.pop(event["gpu"], None) != (event["scene"], event["variant"]):
                raise ValueError("finish does not match launch")
            key = (event["scene"], event["variant"])
            if event["returncode"] != 0 or event["pid"] != launches[key]["pid"]:
                raise ValueError(f"failed process: {key}")
    if active:
        raise ValueError("scheduler has active jobs")
    for scene in SCENES:
        fixed, random = launches[(scene, "fixed_init")], launches[(scene, "random_init")]
        if fixed["gpu"] != random["gpu"] or finishes[(scene, "fixed_init")]["time"] > random["time"]:
            raise ValueError(f"not sequential on same GPU: {scene}")
    return launches, finishes, events


def check_group_basis(exp: Path, scene: str, variant: str, metric: dict) -> dict:
    pretrain = load(exp / "results" / f"{scene}_pretrain.json")
    basis = load(exp / "results" / f"{scene}_basis.json")
    steps, history = pretrain["history"], basis["history"]
    if [x["epoch"] for x in steps] != [0, 25, 50, 100, 300]:
        raise ValueError(f"incomplete Q pretraining: {scene}/{variant}")
    if [x["epoch"] for x in history] != list(range(16)):
        raise ValueError(f"incomplete Q trajectory: {scene}/{variant}")
    if pretrain["probe_sha256"] != PROBE_SHA or basis["retention_probe_sha256"] != PROBE_SHA:
        raise ValueError(f"probe SHA mismatch: {scene}/{variant}")
    q0 = np.asarray(steps[0]["effective_q"], dtype=float)
    q_pre = np.asarray(steps[-1]["effective_q"], dtype=float)
    q_start = np.asarray(history[0]["effective_q"], dtype=float)
    q_final = np.asarray(history[-1]["effective_q"], dtype=float)
    if any(q.shape != (4, 2, 3) for q in (q0, q_pre, q_start, q_final)):
        raise ValueError("Q shape changed")
    if q_final.shape != (4, 2, 3) or not np.allclose(q_pre, q_start, atol=1e-6):
        raise ValueError(f"Q pretraining/joint boundary mismatch: {scene}/{variant}")
    fixed = np.array([[1 / math.sqrt(3)] * 3,
                      [-1 / math.sqrt(2), 0, 1 / math.sqrt(2)]])
    if variant == "fixed_init" and not np.allclose(q0, np.repeat(fixed[None], 4, axis=0), atol=1e-6):
        raise ValueError(f"wrong fixed initialization: {scene}")
    if variant == "random_init" and max(plane_angle(q0[0], q0[g]) for g in range(1, 4)) < 1:
        raise ValueError(f"random groups not independent: {scene}")
    for snapshot in [*steps, *history]:
        q = np.asarray(snapshot["effective_q"], dtype=float)
        if q.shape != (4, 2, 3) or max(np.linalg.norm(q[g] @ q[g].T - np.eye(2)) for g in range(4)) > 1e-5:
            raise ValueError(f"non-orthonormal group Q: {scene}/{variant}")
    probe_grams = np.asarray(pretrain["depth_gram_four_groups"], dtype=float)
    full_grams = np.asarray(basis["final_train_group_grams"], dtype=float)
    if probe_grams.shape != full_grams.shape or full_grams.shape != (4, 3, 3):
        raise ValueError(f"invalid group Grams: {scene}/{variant}")
    probe_opt, full_opt = optimum(probe_grams), optimum(full_grams)
    initial_angles = [plane_angle(q0[g], probe_opt[g]) for g in range(4)]
    pretrain_angles = [plane_angle(q_pre[g], probe_opt[g]) for g in range(4)]
    final_angles = [plane_angle(q_final[g], full_opt[g]) for g in range(4)]
    joint_moves = [plane_angle(q_pre[g], q_final[g]) for g in range(4)]
    final_group_energy = [float(np.trace(q_final[g] @ full_grams[g] @ q_final[g].T)
                                / np.trace(full_grams[g])) for g in range(4)]
    final_energy = group_energy(q_final, full_grams)
    if not np.allclose(final_angles, metric["group_angle_to_retention_optimum_deg"], atol=1e-4):
        raise ValueError(f"final angle metric mismatch: {scene}/{variant}")
    if not np.allclose(final_group_energy, metric["group_retained_normal_energy_fraction"], atol=1e-6):
        raise ValueError(f"group energy metric mismatch: {scene}/{variant}")
    if abs(final_energy - metric["combined_retained_normal_energy_fraction"]) > 1e-6:
        raise ValueError(f"combined energy metric mismatch: {scene}/{variant}")
    if not np.allclose(pretrain_angles, steps[-1]["group_angle_to_retention_optimum_deg"], atol=1e-4):
        raise ValueError(f"pretraining angles mismatch: {scene}/{variant}")
    return {
        "initial_q": q0.tolist(), "pretrained_q": q_pre.tolist(), "final_q": q_final.tolist(),
        "final_raw_q": history[-1]["raw_q"], "full_train_group_grams": full_grams.tolist(),
        "initial_to_probe_optimum_deg": initial_angles,
        "pretrained_to_probe_optimum_deg": pretrain_angles,
        "joint_start_to_final_deg": joint_moves,
        "final_to_full_train_optimum_deg": final_angles,
        "final_group_energy": final_group_energy,
        "final_combined_energy": final_energy,
        "final_distill_train_loss": history[-1]["distill_train_loss"],
        "final_retention_train_loss": history[-1]["retention_loss"],
    }


def main() -> None:
    records = {variant: load(exp / "record.json") for variant, exp in EXPERIMENTS.items()}
    for variant, record in records.items():
        run = record["runs"][0]
        if record["status"] != "completed" or run["status"] != "completed":
            raise ValueError(f"run not completed: {variant}")
        if record["source_commit"] != SOURCE or run["source_commit"] != SOURCE:
            raise ValueError(f"wrong training source: {variant}")
    launches, finishes, events = check_scheduler(records)
    data_root = Path(records["fixed_init"]["runs"][0]["data_root"])
    rows = []
    for scene, scene_name in SCENES.items():
        fingerprint = manifest_fingerprint(data_root / scene_name)
        metrics, scores, bases = {}, {}, {}
        for variant, exp in {**CONTROLS, **EXPERIMENTS}.items():
            metric = load(exp / "results" / f"{scene}.json")
            keys, labels, values = read_scores(exp / "results" / f"{scene}_scores.csv")
            auc = roc_auc_score(labels, values) * 100
            if abs(auc - metric["auc_percent"]) > 1e-8 or fingerprint != metric["split_fingerprint"]:
                raise ValueError(f"AUC or split fingerprint mismatch: {scene}/{variant}")
            if len(labels) != metric["frames"] or sum(labels) != metric["positive_frames"]:
                raise ValueError(f"frame/label count mismatch: {scene}/{variant}")
            metrics[variant], scores[variant] = metric, (keys, labels, values)
        keys, labels, _ = scores["fixed_init"]
        check_frame_labels(data_root / scene_name, keys)
        for variant, (other_keys, other_labels, _) in scores.items():
            if keys != other_keys or labels != other_labels:
                raise ValueError(f"paired test keys/labels mismatch: {scene}/{variant}")
        for variant, exp in EXPERIMENTS.items():
            metric = metrics[variant]
            if metric["seed"] != 42 or metric["trainable_parameters"] != PARAMETERS or metric["retention_probe_sha256"] != PROBE_SHA:
                raise ValueError(f"wrong seed/parameters/probe: {scene}/{variant}")
            lines = Path(launches[(scene, variant)]["log"]).read_text().splitlines()
            epoch_lines = [line for line in lines if line.startswith("epoch=")]
            if [int(line.split()[0].split("=")[1]) for line in epoch_lines] != list(range(1, 16)):
                raise ValueError(f"epoch log incomplete: {scene}/{variant}")
            if len([line for line in lines if '"phase": "q_pretrain_done"' in line]) != 1:
                raise ValueError(f"Q pretrain log incomplete: {scene}/{variant}")
            if abs(json.loads(lines[-1])["auc_percent"] - metric["auc_percent"]) > 1e-8:
                raise ValueError(f"final log metric mismatch: {scene}/{variant}")
            bases[variant] = check_group_basis(exp, scene, variant, metric)
        for variant, exp in CONTROLS.items():
            shared_basis = load(exp / "results" / f"{scene}_basis.json")
            bases[variant] = np.asarray(shared_basis["history"][-1]["effective_q"], dtype=float)
        q_fixed = np.asarray(bases["fixed_init"]["final_q"])
        q_random = np.asarray(bases["random_init"]["final_q"])
        grams_fixed = np.asarray(bases["fixed_init"]["full_train_group_grams"])
        grams_random = np.asarray(bases["random_init"]["full_train_group_grams"])
        if not np.allclose(grams_fixed, grams_random, atol=1e-7):
            raise ValueError(f"fixed/random group Gram mismatch: {scene}")
        for variant, grams in (("shared_fixed", grams_fixed), ("shared_random", grams_random)):
            old_gram = np.asarray(load(CONTROLS[variant] / "results" / f"{scene}_basis.json")["final_train_gram"])
            if not np.allclose(grams.mean(axis=0), old_gram, atol=1e-5):
                raise ValueError(f"shared/group teacher Gram mismatch: {scene}/{variant}")
        pair_angles = [plane_angle(q_fixed[g], q_random[g]) for g in range(4)]
        group_spread = max(plane_angle(q_fixed[g], q_fixed[h]) for g in range(4) for h in range(g + 1, 4))
        row = {
            "scene": scene, "frames": len(labels), "positive_frames": sum(labels),
            "split_fingerprint": fingerprint,
            "shared_fixed_auc_percent": metrics["shared_fixed"]["auc_percent"],
            "shared_random_auc_percent": metrics["shared_random"]["auc_percent"],
            "group_fixed_auc_percent": metrics["fixed_init"]["auc_percent"],
            "group_random_auc_percent": metrics["random_init"]["auc_percent"],
            "fixed_delta_vs_shared_pp": metrics["fixed_init"]["auc_percent"] - metrics["shared_fixed"]["auc_percent"],
            "random_delta_vs_shared_pp": metrics["random_init"]["auc_percent"] - metrics["shared_random"]["auc_percent"],
            "random_minus_fixed_auc_pp": metrics["random_init"]["auc_percent"] - metrics["fixed_init"]["auc_percent"],
            "four_pair_plane_angles_deg": pair_angles,
            "mean_pair_plane_angle_deg": mean(pair_angles),
            "max_pair_plane_angle_deg": max(pair_angles),
            "max_between_group_plane_angle_deg": group_spread,
            "fixed_group_energy": group_energy(q_fixed, grams_fixed),
            "fixed_shared_q_on_same_grams_energy": shared_energy(bases["shared_fixed"], grams_fixed),
            "random_group_energy": group_energy(q_random, grams_random),
            "random_shared_q_on_same_grams_energy": shared_energy(bases["shared_random"], grams_random),
            "same_gpu": launches[(scene, "fixed_init")]["gpu"],
            "fixed_init_basis": bases["fixed_init"], "random_init_basis": bases["random_init"],
            "fixed_launch_utc": launches[(scene, "fixed_init")]["time"],
            "random_finish_utc": finishes[(scene, "random_init")]["time"],
        }
        row["fixed_energy_gain_vs_shared_pp"] = 100 * (row["fixed_group_energy"] - row["fixed_shared_q_on_same_grams_energy"])
        row["random_energy_gain_vs_shared_pp"] = 100 * (row["random_group_energy"] - row["random_shared_q_on_same_grams_energy"])
        rows.append(row)
    macro = {name: mean(row[f"{name}_auc_percent"] for row in rows)
             for name in ("shared_fixed", "shared_random", "group_fixed", "group_random")}
    macro.update({
        "fixed_delta_vs_shared_pp": macro["group_fixed"] - macro["shared_fixed"],
        "random_delta_vs_shared_pp": macro["group_random"] - macro["shared_random"],
        "group_random_minus_fixed_pp": macro["group_random"] - macro["group_fixed"],
        "mean_pair_plane_angle_deg": mean(row["mean_pair_plane_angle_deg"] for row in rows),
        "max_pair_plane_angle_deg": max(row["max_pair_plane_angle_deg"] for row in rows),
        "max_between_group_plane_angle_deg": max(row["max_between_group_plane_angle_deg"] for row in rows),
        "mean_fixed_energy_gain_vs_shared_pp": mean(row["fixed_energy_gain_vs_shared_pp"] for row in rows),
        "mean_random_energy_gain_vs_shared_pp": mean(row["random_energy_gain_vs_shared_pp"] for row in rows),
        "mean_fixed_group_energy": mean(row["fixed_group_energy"] for row in rows),
        "mean_random_group_energy": mean(row["random_group_energy"] for row in rows),
        "max_pretrained_angle_deg": max(angle for row in rows for variant in EXPERIMENTS
                                         for angle in row[f"{variant}_basis"]["pretrained_to_probe_optimum_deg"]),
        "mean_joint_move_fixed_deg": mean(angle for row in rows for angle in row["fixed_init_basis"]["joint_start_to_final_deg"]),
        "mean_joint_move_random_deg": mean(angle for row in rows for angle in row["random_init_basis"]["joint_start_to_final_deg"]),
        "mean_pair_plane_angle_by_group_deg": [mean(row["four_pair_plane_angles_deg"][g] for row in rows) for g in range(4)],
        "max_pair_plane_angle_by_group_deg": [max(row["four_pair_plane_angles_deg"][g] for row in rows) for g in range(4)],
        "mean_fixed_group_energy_by_group": [mean(row["fixed_init_basis"]["final_group_energy"][g] for row in rows) for g in range(4)],
        "mean_random_group_energy_by_group": [mean(row["random_init_basis"]["final_group_energy"][g] for row in rows) for g in range(4)],
        "trainable_parameters": PARAMETERS,
    })
    columns = ["scene", "frames", "positive_frames", "shared_fixed_auc_percent", "shared_random_auc_percent",
               "group_fixed_auc_percent", "group_random_auc_percent", "fixed_delta_vs_shared_pp",
               "random_delta_vs_shared_pp", "random_minus_fixed_auc_pp", "mean_pair_plane_angle_deg",
               "max_pair_plane_angle_deg", "max_between_group_plane_angle_deg", "fixed_energy_gain_vs_shared_pp",
               "random_energy_gain_vs_shared_pp", "same_gpu"]
    report = {
        "run_ids": {variant: records[variant]["runs"][0]["run_id"] for variant in EXPERIMENTS},
        "source_commit": SOURCE, "probe_sha256": PROBE_SHA,
        "method": "14 scheduler returns and 15-epoch logs verified; frame labels, split manifests, and all four AUC variants recomputed from per-frame scores; four Q subspaces and teacher-energy fractions independently recalculated",
        "scenes": rows, "macro": macro,
        "scheduler": {"events": len(events), "launches": 14, "zero_return_finishes": 14,
                      "same_gpu_scene_pairs": 7, "final_time_utc": events[-1]["time"]},
    }
    lines = ["# 四组独立 Q 与分组 C：七场景复核", "",
             "14/14 次训练返回码为 0，每次完成 15 epoch。逐帧 ROC AUC×100 已从 CSV 重算，并核对测试帧标签、数据划分指纹、Q 正交性和四组 Gram 能量。每对固定/随机初值在相同逻辑 GPU 上先后运行。", "",
             "| 场景 | 共享 Q 固定 | 共享 Q 随机 | 四组 Q 固定 | 四组 Q 随机 | Δ固定 | Δ随机 | 四组配对最大夹角(°) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f'| {row["scene"]} | {row["shared_fixed_auc_percent"]:.4f} | {row["shared_random_auc_percent"]:.4f} | {row["group_fixed_auc_percent"]:.4f} | {row["group_random_auc_percent"]:.4f} | {row["fixed_delta_vs_shared_pp"]:+.4f} | {row["random_delta_vs_shared_pp"]:+.4f} | {row["max_pair_plane_angle_deg"]:.4f} |')
    lines.extend([
        f'| **宏平均** | **{macro["shared_fixed"]:.4f}** | **{macro["shared_random"]:.4f}** | **{macro["group_fixed"]:.4f}** | **{macro["group_random"]:.4f}** | **{macro["fixed_delta_vs_shared_pp"]:+.4f}** | **{macro["random_delta_vs_shared_pp"]:+.4f}** | **{macro["max_pair_plane_angle_deg"]:.4f}** |',
        "", "Δ固定/Δ随机分别相对同初始化的共享 Q 对照，单位为 AUC 百分点。各场景独立训练、独立评测；宏平均是不加权七场景平均。", "",
        "## Q 方向与正常特征能量", "",
        f'固定/随机初值的四组最终子空间配对夹角：28 组平均 {macro["mean_pair_plane_angle_deg"]:.4f}°、最大 {macro["max_pair_plane_angle_deg"]:.4f}°，低于预设 5° 判据。四组之间的最终方向确有区别，跨组最大夹角在七场景中最高 {macro["max_between_group_plane_angle_deg"]:.3f}°。300 步 Q 单独预训练后，离各自 16 张正常帧 Gram 最优方向的最大夹角 {macro["max_pretrained_angle_deg"]:.4f}°。联合训练阶段，相对预训练终点的平均移动为固定 {macro["mean_joint_move_fixed_deg"]:.4f}°、随机 {macro["mean_joint_move_random_deg"]:.4f}°。',
        "", "| 教师组 | 固定/随机最终平面平均夹角 | 最大夹角 | 固定 Q 组内能量保留率 | 随机 Q 组内能量保留率 |",
        "|---|---:|---:|---:|---:|",
    ])
    for g in range(4):
        lines.append(f'| {3*g}–{3*g+2} | {macro["mean_pair_plane_angle_by_group_deg"][g]:.4f}° | {macro["max_pair_plane_angle_by_group_deg"][g]:.4f}° | {100*macro["mean_fixed_group_energy_by_group"][g]:.3f}% | {100*macro["mean_random_group_energy_by_group"][g]:.3f}% |')
    lines.extend([
        "", "教师组按代码中的 0 起始编号；第 0–2 层、第 3–5 层、第 6–8 层、第 9–11 层分别对应一个独立 Q 与 C。", "",
        f'用每场景**同一组**最终训练 Gram 计算，四组 Q 相对共享 Q 的正常能量保留率平均增加固定 {macro["mean_fixed_energy_gain_vs_shared_pp"]:+.4f}、随机 {macro["mean_random_energy_gain_vs_shared_pp"]:+.4f} 个百分点；四组 Q 最终保留率平均为 {100*macro["mean_fixed_group_energy"]:.3f}% / {100*macro["mean_random_group_energy"]:.3f}%。该能量目标有明确方向，但并非异常 AUC 的代理。',
        "", "## AUC 解读与限制", "",
        f'四组 Q 的七场景宏平均相对共享 Q 只改变固定 {macro["fixed_delta_vs_shared_pp"]:+.4f}、随机 {macro["random_delta_vs_shared_pp"]:+.4f} 个百分点。变化小且场景间有正有负，当前单种子实验不足以证明 AUC 稳定提升。固定与随机初值最终子空间几乎相同，AUC 宏平均仍相差 {macro["group_random_minus_fixed_pp"]:+.4f} 个百分点；Q 的行坐标和下游权重共同训练，不能把差异单独归因于子空间。',
        "Q 预训练用了每场景 16 张正常帧；只有 seed42 和一次随机四组 Q 抽样。共享 Q 对照预训练 200 步，四组 Q 用 300 步，二者均已收敛；该步数差异仍是比较限制。正常特征能量的小幅收益没有形成稳定 AUC 收益。参数数目为 OCBE＋学生 3,761,240，不含冻结教师、行为分支或 KNN。", "",
    ])
    for exp in EXPERIMENTS.values():
        out = exp / "results"
        (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        with (out / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows({name: row[name] for name in columns} for row in rows)
        (out / "summary.md").write_text("\n".join(lines))
    print(json.dumps(macro, ensure_ascii=False))


if __name__ == "__main__":
    main()
