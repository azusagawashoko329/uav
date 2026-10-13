#!/usr/bin/env python3
"""Verify the 28 seed-43 scene runs and compare them with paired seed-42 runs."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "da" / "appearance"
DATA = Path("/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly")
RUN_ID = "20261013T100419Z_seed43_four_variant_matrix"
SCENES = {
    "bike": "Bike Roundabout", "crossroads": "Crossroads",
    "farmland": "Farmland Inspection", "highway": "Highway",
    "railway": "Railway Inspection", "solar": "Solar Panel Inspection",
    "vehicle": "Vehicle Roundabout",
}
VARIANTS = (
    ("768+原版OCBE", "20261013_seed43_glll768_ocbe512_s43",
     "20261012_structural_mixer_glll_s42"),
    ("192+原版OCBE", "20261013_seed43_glll192_ocbe512_s43",
     "20261012_student_width192_glll_s42"),
    ("192+dense128", "20261013_seed43_glll192_ocbe128_s43",
     "20261013_ocbe_hidden128_dense_s42"),
    ("192+dense128+rank2", "20261013_seed43_glll192_ocbe128_rank2_s43",
     "20261013_ocbe_depthrank2_hidden128_s42"),
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def frame_key(path):
    return int(Path(path).stem)


def manifest(scene_dir):
    train = sorted(glob.glob(str(scene_dir / "sequence*" / "train" / "*" / "*.jpg")))
    videos = sorted(Path(p) for p in glob.glob(str(scene_dir / "sequence*" / "test" / "*"))
                    if Path(p).is_dir())
    require(train and videos, f"missing data: {scene_dir}")
    lines = [f"{Path(p).relative_to(scene_dir)}:{Path(p).stat().st_size}" for p in train]
    for video in videos:
        frames = sorted(video.glob("*.jpg"), key=frame_key)
        labels = Path(str(video) + ".npy")
        require(frames and labels.is_file(), f"missing frames/labels: {video}")
        lines.extend(f"{p.relative_to(scene_dir)}:{p.stat().st_size}" for p in frames)
        lines.append(f"{labels.relative_to(scene_dir)}:{hashlib.sha256(labels.read_bytes()).hexdigest()}")
    return len(train), len(videos), hashlib.sha256("\n".join(lines).encode()).hexdigest()


def score_rows(path, scene_dir):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == ["video", "frame", "label", "anomaly_score"],
                f"unexpected score CSV header: {path}")
        rows = list(reader)
    require(rows, f"empty score CSV: {path}")
    keys = [(r["video"], r["frame"], int(r["label"])) for r in rows]
    require(len(keys) == len(set(keys)), f"duplicate frame key: {path}")
    scores = np.asarray([float(r["anomaly_score"]) for r in rows])
    require(np.isfinite(scores).all(), f"non-finite score: {path}")
    grouped = defaultdict(list)
    for video, frame, label in keys:
        require(label in (0, 1), f"non-binary label: {path}")
        grouped[video].append((frame, label))
    actual_videos = sorted(Path(p) for p in glob.glob(str(scene_dir / "sequence*" / "test" / "*"))
                           if Path(p).is_dir())
    require(set(grouped) == {str(v.relative_to(scene_dir)) for v in actual_videos},
            f"test-video mismatch: {path}")
    for video in actual_videos:
        expected_frames = [p.name for p in sorted(video.glob("*.jpg"), key=frame_key)]
        labels = np.asarray(np.load(str(video) + ".npy", allow_pickle=True)).astype(int).ravel()
        actual = grouped[str(video.relative_to(scene_dir))]
        require(actual == list(zip(expected_frames, labels.tolist())),
                f"raw frame/label mismatch: {path}, {video}")
    y = np.asarray([k[2] for k in keys])
    auc = float(roc_auc_score(y, scores) * 100)
    return keys, auc, int(y.sum()), float(scores[y == 0].mean()), float(scores[y == 1].mean())


def scheduler_events():
    path = BASE / VARIANTS[0][1] / "logs" / "matrix_stdout.log"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    require(events and all(e["run_id"] == RUN_ID for e in events), "scheduler run_id mismatch")
    kinds = Counter(e["kind"] for e in events)
    require(kinds == {"gpu_probe": 1, "launch": 28, "finish": 28, "all_finished": 1},
            f"scheduler event counts: {kinds}")
    require(events[-1]["kind"] == "all_finished" and not events[-1]["failures"],
            "scheduler did not finish cleanly")
    launches = {(e["experiment"], e["scene"]): e for e in events if e["kind"] == "launch"}
    finishes = {(e["experiment"], e["scene"]): e for e in events if e["kind"] == "finish"}
    expected = {(name, scene) for _, name, _ in VARIANTS for scene in SCENES}
    require(set(launches) == expected and set(finishes) == expected,
            "scheduler scene matrix incomplete or duplicated")
    require(all(e["returncode"] == 0 for e in finishes.values()), "nonzero scene exit")
    for _, name, _ in VARIANTS:
        path = BASE / name / "logs" / f"scheduler_{RUN_ID}.jsonl"
        local = [json.loads(line) for line in path.read_text().splitlines()]
        require(sum(e["kind"] == "launch" for e in local) == 7 and
                sum(e["kind"] == "finish" for e in local) == 7,
                f"local scheduler log incomplete: {name}")
    return launches, finishes, events[-1]["time"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write validated summaries")
    args = ap.parse_args()
    launches, finishes, ended_utc = scheduler_events()
    manifests = {scene: manifest(DATA / folder) for scene, folder in SCENES.items()}
    paired_keys = {}
    results = {}
    for label, name, old_name in VARIANTS:
        directory = BASE / name
        cfg = load_json(directory / "config.json")
        rec = load_json(directory / "record.json")
        old = load_json(BASE / old_name / "results" / "summary.json")
        old_scenes = {row["scene"]: row for row in old["scenes"]}
        require(rec["source_commit"] == "07af639ce4514b238ab207afe04b604ae7cde477",
                f"source commit mismatch: {name}")
        require(cfg["seed"] == 43 and cfg["student"]["stages"] == 4,
                f"config mismatch: {name}")
        scenes = []
        for scene in SCENES:
            scene_dir = DATA / SCENES[scene]
            metric = load_json(directory / "results" / f"{scene}.json")
            train_n, videos_n, fingerprint = manifests[scene]
            launch, finish = launches[(name, scene)], finishes[(name, scene)]
            log = (directory / launch["log"]).read_text().splitlines()
            require(len(log) >= 17, f"short training log: {name}/{scene}")
            header, last = json.loads(log[0]), json.loads(log[-1])
            epochs = [int(re.match(r"epoch=(\d+)\b", line).group(1)) for line in log[1:-1]
                      if re.match(r"epoch=(\d+)\b", line)]
            require(epochs == list(range(1, 16)), f"epoch count/order: {name}/{scene}")
            require(header["seed"] == metric["seed"] == last["seed"] == 43,
                    f"seed mismatch: {name}/{scene}")
            require(header["architecture"] == metric["architecture"] == last["architecture"]
                    == cfg["variant"], f"architecture mismatch: {name}/{scene}")
            require(header["train_frames"] == train_n and header["test_videos"] == videos_n,
                    f"split count mismatch: {name}/{scene}")
            require(header["split_fingerprint"] == metric["split_fingerprint"] == fingerprint,
                    f"split fingerprint mismatch: {name}/{scene}")
            require(header["trainable_parameters"] == metric["trainable_parameters"]
                    == cfg["predicted_trainable_parameters"]["total"],
                    f"parameter count mismatch: {name}/{scene}")
            require(last == metric, f"metric/log mismatch: {name}/{scene}")
            keys, auc, positives, normal_mean, abnormal_mean = score_rows(
                directory / "results" / f"{scene}_scores.csv", scene_dir)
            require(len(keys) == metric["frames"] and positives == metric["positive_frames"],
                    f"score count mismatch: {name}/{scene}")
            require(math.isclose(auc, metric["auc_percent"], abs_tol=1e-10),
                    f"AUC recomputation mismatch: {name}/{scene}")
            baseline_keys, baseline_auc, _, _, _ = score_rows(
                BASE / old_name / "results" / f"{scene}_scores.csv", scene_dir)
            require(keys == baseline_keys, f"seed42 frame/label mismatch: {name}/{scene}")
            require(math.isclose(baseline_auc, old_scenes[scene]["auc_percent"], abs_tol=1e-10),
                    f"seed42 AUC mismatch: {name}/{scene}")
            if scene in paired_keys:
                require(keys == paired_keys[scene], f"cross-variant frame/label mismatch: {scene}")
            else:
                paired_keys[scene] = keys
            scenes.append({"scene": scene, "auc_percent": auc,
                           "seed42_auc_percent": baseline_auc,
                           "delta_seed43_minus_seed42_pp": auc - baseline_auc,
                           "frames": len(keys), "positive_frames": positives,
                           "normal_mean_score": normal_mean,
                           "abnormal_mean_score": abnormal_mean,
                           "trainable_parameters": metric["trainable_parameters"],
                           "split_fingerprint": fingerprint,
                           "gpu_logical": launch["gpu"], "cuda_device_name": header["cuda"],
                           "launch_time_utc": launch["time"],
                           "finish_time_utc": finish["time"]})
        macro43 = mean(row["auc_percent"] for row in scenes)
        macro42 = mean(row["seed42_auc_percent"] for row in scenes)
        results[name] = {"label": label, "run_id": RUN_ID, "source_commit": rec["source_commit"],
                         "seed43_experiment": name, "seed42_experiment": old_name,
                         "scenes": scenes,
                         "macro": {"seed43_auc_percent": macro43,
                                   "seed42_auc_percent": macro42,
                                   "delta_seed43_minus_seed42_pp": macro43 - macro42,
                                   "two_seed_mean_auc_percent": (macro43 + macro42) / 2,
                                   "trainable_parameters": scenes[0]["trainable_parameters"]},
                         "validation": {"completed_scenes": 7, "epochs_per_scene": 15,
                                        "auc_recomputed": True,
                                        "raw_labels_and_frame_keys_verified": True,
                                        "seed42_frame_keys_matched": True,
                                        "split_fingerprints_matched": True,
                                        "scheduler_finish_utc": ended_utc}}
    for name, report in results.items():
        m = report["macro"]
        print(f"{name}: seed43={m['seed43_auc_percent']:.6f}, "
              f"seed42={m['seed42_auc_percent']:.6f}, "
              f"delta={m['delta_seed43_minus_seed42_pp']:+.6f}, "
              f"params={m['trainable_parameters']:,}")
    if args.write:
        write_reports(results)


def write_reports(results):
    for name, report in results.items():
        directory = BASE / name / "results"
        write_json(directory / "summary.json", report)
        with (directory / "summary.csv").open("w", newline="") as f:
            fields = list(report["scenes"][0])
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(report["scenes"])
        m = report["macro"]
        lines = [f"# Seed43 七场景复核：{report['label']}", "",
                 f"训练源码 `{report['source_commit']}`；运行 `{RUN_ID}`。28 项调度均零返回。逐帧分数重新计算 AUC，并与原始 `.npy` 标签、帧键、数据指纹及对应 seed42 实验配对核验。",
                 "", "| 场景 | seed42 AUC | seed43 AUC | Δ43−42 (pp) | 帧数 | GPU（PyTorch） |",
                 "|---|---:|---:|---:|---:|---|"]
        for row in report["scenes"]:
            lines.append(f"| {row['scene']} | {row['seed42_auc_percent']:.4f} | "
                         f"{row['auc_percent']:.4f} | "
                         f"{row['delta_seed43_minus_seed42_pp']:+.4f} | "
                         f"{row['frames']} | {row['cuda_device_name']} |")
        lines += [f"| **七场景宏平均** | **{m['seed42_auc_percent']:.4f}** | "
                  f"**{m['seed43_auc_percent']:.4f}** | "
                  f"**{m['delta_seed43_minus_seed42_pp']:+.4f}** | | |",
                  "", f"可训练 OCBE＋学生参数：**{m['trainable_parameters']:,}**。",
                  "", "观察：两颗种子各自七场景宏平均的均值为 "
                  f"**{m['two_seed_mean_auc_percent']:.4f}**。场景差异和与其他结构的配对比较见 "
                  "[四结构总表](../../20261013_seed43_glll192_ocbe512_s43/results/four_variant_review.md)。",
                  "", "限制：只有 seed42/43 两颗种子；宏平均差异不能直接解释为稳定性能提升。"
                  "不同场景独立训练和测试，不能在测试集上事后选择最优结构作为无偏结果。"
                  "此参数量不含冻结教师、YOLO 行为分支或 KNN 数据库；无人机端时延未测。", ""]
        (directory / "summary.md").write_text("\n".join(lines))
    write_matrix_report(results)


def write_matrix_report(results):
    reports = [results[name] for _, name, _ in VARIANTS]
    by_label = {report["label"]: report for report in reports}
    baseline = reports[0]["macro"]["seed43_auc_percent"]
    rank = reports[-1]["macro"]
    dense = reports[-2]["macro"]
    original = reports[1]["macro"]
    lines = ["# Seed42/43：四种四 block Global-Local＋OCBE 配置复核", "",
             "四组仅按学生内部宽度与 OCBE 融合结构区分；每组七场景分别训练、分别评测。"
             "本轮 seed43 的 28 项任务全部完成且零返回；所有逐帧 AUC、原始标签、帧键及数据指纹已复核。",
             "", "| 配置 | 可训练参数 | seed42 宏AUC | seed43 宏AUC | 两种子均值 | seed43 相对 768 (pp) |",
             "|---|---:|---:|---:|---:|---:|"]
    for report in reports:
        m = report["macro"]
        lines.append(f"| [{report['label']}](../../{report['seed43_experiment']}/results/summary.md) | "
                     f"{m['trainable_parameters']:,} | {m['seed42_auc_percent']:.4f} | "
                     f"{m['seed43_auc_percent']:.4f} | "
                     f"{m['two_seed_mean_auc_percent']:.4f} | "
                     f"{m['seed43_auc_percent'] - baseline:+.4f} |")
    lines += ["", "| 场景（seed43） | 768+原版 | 192+原版 | 192+dense128 | 192+dense128+rank2 | rank2−dense128 |",
              "|---|---:|---:|---:|---:|---:|"]
    scene_maps = [{r["scene"]: r["auc_percent"] for r in report["scenes"]} for report in reports]
    for scene in SCENES:
        vals = [d[scene] for d in scene_maps]
        lines.append(f"| {scene} | " + " | ".join(f"{v:.4f}" for v in vals)
                     + f" | {vals[3]-vals[2]:+.4f} |")
    rank_dense = {scene: scene_maps[3][scene] - scene_maps[2][scene] for scene in SCENES}
    rank_wide = {scene: scene_maps[3][scene] - scene_maps[0][scene] for scene in SCENES}
    lines += ["", "## 观察与判读", "",
              f"- rank2 相对 dense128：seed42 为 "
              f"{rank['seed42_auc_percent']-dense['seed42_auc_percent']:+.4f} pp，"
              f"seed43 为 {rank['seed43_auc_percent']-dense['seed43_auc_percent']:+.4f} pp；"
              f"两种子宏平均差 {rank['two_seed_mean_auc_percent']-dense['two_seed_mean_auc_percent']:+.4f} pp，"
              f"参数少 {dense['trainable_parameters']-rank['trainable_parameters']:,}（"
              f"{100*(dense['trainable_parameters']-rank['trainable_parameters'])/dense['trainable_parameters']:.2f}%）。",
              f"- rank2 相对 192+原版：seed43 宏平均差 "
              f"{rank['seed43_auc_percent']-original['seed43_auc_percent']:+.4f} pp；"
              f"相对 768+原版差 {rank['seed43_auc_percent']-baseline:+.4f} pp，"
              f"参数少 {100*(reports[0]['macro']['trainable_parameters']-rank['trainable_parameters'])/reports[0]['macro']['trainable_parameters']:.2f}%。",
              f"- seed43 的 rank2−dense128 场景差异集中在 Vehicle "
              f"{rank_dense['vehicle']:+.4f} pp 和 Crossroads {rank_dense['crossroads']:+.4f} pp；"
              f"Solar {rank_dense['solar']:+.4f} pp、Railway {rank_dense['railway']:+.4f} pp。"
              "seed42 中 Railway 为正、Crossroads 为负，说明逐场景优势没有稳定复现。",
              f"- 与 768 维原版相比，seed43 的 rank2 在 Solar {rank_wide['solar']:+.4f} pp、"
              f"Railway {rank_wide['railway']:+.4f} pp；减参仍伴随这两个场景的明显 AUC 代价。",
              "- 七场景变化并不一致；逐场景 AUC 应与宏平均同时报告。两颗种子能看到方向是否重复，"
              "但不足以估计稳定方差或显著性，尤其不能把 0.1–0.3 pp 的差异写成已证实提升。",
              "", "## 口径与限制", "",
              "AUC 为每场景全部测试视频的逐帧 ROC AUC×100；七场景算术平均，seed42/43 再算算术平均。"
              "分数越大越异常。训练只用正常帧；12 个教师 tap、四个学生 block、三对一反向对齐、"
              "group-balanced top-5 loss 和训练超参保持一致。所有参数量只含可训练 OCBE＋学生，"
              "不含冻结教师、YOLO 行为分支或 KNN 库。未测部署端时延。"
              "本报告只比较预先确定的四个结构，不以测试集上的场景最优组合推断无偏总体性能。", ""]
    target = BASE / "20261013_seed43_glll192_ocbe512_s43" / "results" / "four_variant_review.md"
    target.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
