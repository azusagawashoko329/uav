#!/usr/bin/env python3
"""Audit seven-scene OCBE runs against paired frame-level baselines.

This shared, post-run tool is separate from the pinned training source. It checks
the scheduler, epoch logs, split fingerprints, frame keys/labels, and stored AUCs
before writing a Chinese result report.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from sklearn.metrics import roc_auc_score


SCENES = ("highway", "farmland", "bike", "solar", "vehicle", "crossroads", "railway")


def read_scores(path: Path):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or set(rows[0]) != {"video", "frame", "label", "anomaly_score"}:
        raise ValueError(f"invalid score CSV: {path}")
    keys = [(row["video"], row["frame"], row["label"]) for row in rows]
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["anomaly_score"]) for row in rows]
    if not all(math.isfinite(value) for value in scores):
        raise ValueError(f"non-finite anomaly score: {path}")
    if set(labels) != {0, 1}:
        raise ValueError(f"AUC requires both classes: {path}")
    return keys, labels, scores


def read_scheduler(path: Path):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if not events or events[-1]["kind"] != "all_finished":
        raise ValueError("scheduler lacks all_finished event")
    launches = [event for event in events if event["kind"] == "launch"]
    finishes = [event for event in events if event["kind"] == "finish"]
    if len(launches) != 7 or len(finishes) != 7:
        raise ValueError("scheduler did not launch and finish exactly seven scenes")
    if {event["scene"] for event in launches} != set(SCENES):
        raise ValueError("scheduler scene set differs from expected seven scenes")
    if {event["scene"] for event in finishes} != set(SCENES):
        raise ValueError("scheduler finish set differs from expected seven scenes")
    if any(event["returncode"] != 0 for event in finishes) or events[-1]["failures"]:
        raise ValueError("scheduler reports failed scenes")
    started = {event["scene"]: event for event in launches}
    ended = {event["scene"]: event for event in finishes}
    for scene in SCENES:
        if (started[scene]["gpu"], started[scene]["pid"]) != (ended[scene]["gpu"], ended[scene]["pid"]):
            raise ValueError(f"scheduler launch/finish mismatch: {scene}")
    active = {}
    launched_count = 0
    reassignments = 0
    for index, event in enumerate(events):
        if event["kind"] == "launch":
            if event["gpu"] in active:
                raise ValueError(f"GPU {event['gpu']} was assigned overlapping scenes")
            active[event["gpu"]] = event["scene"]
            launched_count += 1
        elif event["kind"] == "finish":
            if active.pop(event["gpu"], None) != event["scene"]:
                raise ValueError(f"GPU {event['gpu']} finished an unassigned scene")
            if launched_count < len(SCENES):
                if index + 1 >= len(events) or events[index + 1]["kind"] != "launch" or events[index + 1]["gpu"] != event["gpu"]:
                    raise ValueError(f"GPU {event['gpu']} was not refilled from the pending queue")
                reassignments += 1
    if active or reassignments != len(SCENES) - len({event["gpu"] for event in launches}):
        raise ValueError("scheduler did not dynamically reuse available GPU workers")
    return started, ended, events


def parse_comparison(raw: str):
    if "=" not in raw:
        raise ValueError("comparison must be NAME=EXPERIMENT_PATH")
    name, path = raw.split("=", 1)
    if not name or not name.replace("_", "").isalnum():
        raise ValueError(f"invalid comparison name: {name}")
    experiment = Path(path).resolve()
    if not experiment.is_dir():
        raise FileNotFoundError(experiment)
    return name, experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--comparison", required=True, action="append",
                        help="NAME=EXPERIMENT_PATH; repeat for multiple baselines")
    parser.add_argument("--label", required=True, help="Chinese label for the current OCBE variant")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    comparisons = dict(parse_comparison(raw) for raw in args.comparison)
    if len(comparisons) != len(args.comparison):
        raise ValueError("duplicate comparison names")
    record = json.loads((experiment / "record.json").read_text())
    runs = [run for run in record["runs"] if run.get("scheduler_log")]
    if len(runs) != 1:
        raise ValueError("expected exactly one scheduler run in this experiment")
    run = runs[0]
    launches, finishes, events = read_scheduler(experiment / run["scheduler_log"])
    config = json.loads((experiment / "config.json").read_text())
    expected_params = config["predicted_trainable_parameters"]["total"]
    details = []
    for scene in SCENES:
        metric = json.loads((experiment / f"results/{scene}.json").read_text())
        keys, labels, scores = read_scores(experiment / f"results/{scene}_scores.csv")
        if len(labels) != metric["frames"] or sum(labels) != metric["positive_frames"]:
            raise ValueError(f"frame and positive counts differ: {scene}")
        auc = roc_auc_score(labels, scores) * 100
        if abs(auc - metric["auc_percent"]) > 1e-8:
            raise ValueError(f"stored AUC differs from recomputation: {scene}")
        if metric["trainable_parameters"] != expected_params:
            raise ValueError(f"parameter count differs from config: {scene}")
        log = Path(launches[scene]["log"]).read_text().splitlines()
        if len([line for line in log if line.startswith("epoch=")]) != args.epochs:
            raise ValueError(f"unexpected completed epoch count: {scene}")
        if abs(json.loads(log[-1])["auc_percent"] - auc) > 1e-8:
            raise ValueError(f"scene log final metric differs: {scene}")
        normal = [score for label, score in zip(labels, scores) if label == 0]
        abnormal = [score for label, score in zip(labels, scores) if label == 1]
        row = {
            "scene": scene,
            "auc_percent": auc,
            "frames": len(labels),
            "positive_frames": sum(labels),
            "normal_mean_score": mean(normal),
            "abnormal_mean_score": mean(abnormal),
            "score_gap": mean(abnormal) - mean(normal),
            "trainable_parameters": metric["trainable_parameters"],
            "split_fingerprint": metric["split_fingerprint"],
            "gpu": launches[scene]["gpu"],
            "launch_time_utc": launches[scene]["time"],
            "finish_time_utc": finishes[scene]["time"],
        }
        for name, baseline in comparisons.items():
            old_metric = json.loads((baseline / f"results/{scene}.json").read_text())
            old_keys, old_labels, old_scores = read_scores(baseline / f"results/{scene}_scores.csv")
            if keys != old_keys or labels != old_labels:
                raise ValueError(f"paired frame keys or labels differ from {name}: {scene}")
            if metric["split_fingerprint"] != old_metric["split_fingerprint"]:
                raise ValueError(f"split fingerprint differs from {name}: {scene}")
            old_auc = roc_auc_score(old_labels, old_scores) * 100
            if abs(old_auc - old_metric["auc_percent"]) > 1e-8:
                raise ValueError(f"stored {name} AUC differs from recomputation: {scene}")
            row[f"{name}_auc_percent"] = old_auc
            row[f"delta_vs_{name}_pp"] = auc - old_auc
            row[f"{name}_trainable_parameters"] = old_metric["trainable_parameters"]
        details.append(row)
    macro = {
        "auc_percent": mean(row["auc_percent"] for row in details),
        "trainable_parameters": expected_params,
    }
    for name in comparisons:
        macro[f"{name}_auc_percent"] = mean(row[f"{name}_auc_percent"] for row in details)
        macro[f"delta_vs_{name}_pp"] = mean(row[f"delta_vs_{name}_pp"] for row in details)
        baseline_params = details[0][f"{name}_trainable_parameters"]
        if any(row[f"{name}_trainable_parameters"] != baseline_params for row in details):
            raise ValueError(f"{name} parameter count differs across scenes")
        macro[f"{name}_trainable_parameters"] = baseline_params
        macro[f"parameter_reduction_vs_{name}_percent"] = 100 * (1 - expected_params / baseline_params)
    output = experiment / "results"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(details[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(details)
    (output / "summary.json").write_text(json.dumps({
        "run_id": run["run_id"],
        "method": "recomputed ROC AUC from paired frame-score CSVs; verified scheduler, epochs, labels and split fingerprints",
        "scenes": details,
        "macro": macro,
        "scheduler": {"events": len(events), "launches": 7, "zero_return_finishes": 7,
                      "gpu_assignments": {scene: launches[scene]["gpu"] for scene in SCENES}},
    }, ensure_ascii=False, indent=2) + "\n")
    names = list(comparisons)
    header = "| 场景 | 本实验 AUC | " + " | ".join(f"{name} AUC | Δ本实验−{name}" for name in names) + " |"
    rule = "|---|---:|" + "---:|---:|" * len(names)
    lines = [f"# {args.label}：七场景复核", "", "逐帧分数重新计算 AUC；七场景逐一核对帧键、标签、数据划分指纹、15 个 epoch 和调度器零返回状态。", "", header, rule]
    for row in details:
        cells = [row["scene"], f'{row["auc_percent"]:.4f}']
        for name in names:
            cells.extend([f'{row[f"{name}_auc_percent"]:.4f}', f'{row[f"delta_vs_{name}_pp"]:+.4f}'])
        lines.append("| " + " | ".join(cells) + " |")
    cells = ["**七场景宏平均**", f'**{macro["auc_percent"]:.4f}**']
    for name in names:
        cells.extend([f'**{macro[f"{name}_auc_percent"]:.4f}**',
                      f'**{macro[f"delta_vs_{name}_pp"]:+.4f}**'])
    lines.extend(["| " + " | ".join(cells) + " |", "", "## 参数量与观察", "",
                  f'本实验 OCBE＋学生可训练参数为 **{expected_params:,}**。'])
    for name in names:
        lines.append(f'相对 {name} 的 {macro[f"{name}_trainable_parameters"]:,} 个参数，减少 **{macro[f"parameter_reduction_vs_{name}_percent"]:.2f}%**，宏平均 AUC 变化 **{macro[f"delta_vs_{name}_pp"]:+.4f}** 个百分点。')
        worst = min(details, key=lambda row: row[f"delta_vs_{name}_pp"])
        lines.append(f'相对 {name} 降幅最大的场景是 {worst["scene"]}（{worst[f"delta_vs_{name}_pp"]:+.4f} 个百分点）。')
    lines.extend(["", "## 限制", "",
                  "仅 seed=42，不能据此推断跨种子稳定性；正常特征能量和训练 loss 均不能代替异常 AUC。参数量不包含冻结教师、行为分支或 KNN 数据库；无人机端推理延迟尚未测量。", ""])
    (output / "summary.md").write_text("\n".join(lines))
    print(json.dumps(macro, ensure_ascii=False))


if __name__ == "__main__":
    main()
