"""Verify and summarize the five completed UIT runs without rerunning inference."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.uit_appearance_runner import build_variant, frame_key

HOST = Path(__file__).resolve().parents[1]
BASE = HOST.parent
VARIANTS = [
    ("4block + Dense512", "20261030_uit_4block_plain_retry2_s42"),
    ("4block + GL768 + Dense512", "20261030_uit_4block_gl768_retry2_s42"),
    ("4block + GL192 + Dense512", "20261030_uit_4block_gl192_retry2_s42"),
    ("4block + GL192 + Dense128", "20261030_uit_gl192_dense128_retry2_s42"),
    ("4block + GL192 + Dense128 + Rank2", HOST.name),
]
SCHEDULER_ID = "20261030T070733Z_uit_five_variant_matrix"
SCHEDULER_LOG = f"logs/scheduler_{SCHEDULER_ID}.jsonl"
TRAIN_SOURCE = "ff5660e4077484f895755cce95a9a3917582edf6"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def timestamp(value):
    return dt.datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


def verify(name, ident, expected_rows):
    directory = BASE / ident
    config = read_json(directory / "config.json")
    metric = read_json(directory / "results/uit.json")
    events = [json.loads(line) for line in (directory / SCHEDULER_LOG).read_text().splitlines()]
    assert events[-1]["kind"] == "all_finished" and events[-1]["failures"] == []
    launch = next(e for e in events if e["kind"] == "launch" and e["variant"] == ident)
    finish = next(e for e in events if e["kind"] == "finish" and e["variant"] == ident)
    assert finish["returncode"] == 0
    log = directory / "logs" / Path(launch["log"]).name
    lines = log.read_text().splitlines()
    startup = json.loads(lines[0])
    pattern = re.compile(r"^epoch=(\d+) loss=([\d.e+-]+) elapsed_s=([\d.]+)$")
    epochs = []
    last_time = 0.0
    for line in lines:
        match = pattern.fullmatch(line)
        if match:
            epoch, loss, elapsed = int(match[1]), float(match[2]), float(match[3])
            epochs.append({"epoch": epoch, "loss": loss, "elapsed_s": elapsed,
                           "epoch_s": elapsed - last_time})
            last_time = elapsed
    assert [e["epoch"] for e in epochs] == list(range(1, 31))
    assert metric["max_steps"] == 0 and metric["seed"] == config["seed"] == 42
    assert metric["train_frames"] == startup["train_frames"] == 59186
    with (directory / "results/uit_scores.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    keys_labels = [(r["video"], r["frame"], int(r["label"])) for r in rows]
    assert keys_labels == expected_rows, f"frame/label mismatch: {ident}"
    assert len(set((v, f) for v, f, _ in keys_labels)) == len(rows)
    labels = np.array([r[2] for r in keys_labels])
    scores = np.array([float(r["anomaly_score"]) for r in rows])
    assert np.isfinite(scores).all() and set(labels) == {0, 1}
    assert len(rows) == metric["frames"] == 147005
    assert int(labels.sum()) == metric["positive_frames"] == 63485
    assert len(set(r[0] for r in keys_labels)) == metric["test_videos"] == 51
    auc = float(roc_auc_score(labels, scores) * 100)
    fpr, tpr, _ = roc_curve(labels, scores)
    eer = float(fpr[np.argmin(np.abs(1 - tpr - fpr))])
    assert abs(auc - metric["auc_percent"]) < 1e-10
    assert abs(eer - metric["eer"]) < 1e-12
    assert json.loads(lines[-1]) == metric
    checkpoint_path = directory / metric["checkpoint"]
    assert sha256(checkpoint_path) == metric["checkpoint_sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["variant"] == metric["variant"] == config["variant"]
    assert checkpoint["seed"] == 42
    assert checkpoint["split_fingerprint"] == metric["split_fingerprint"] == startup["split_fingerprint"]
    ocbe, student = build_variant(config, 42)
    ocbe.load_state_dict(checkpoint["ocbe_state_dict"], strict=True)
    student.load_state_dict(checkpoint["student_state_dict"], strict=True)
    ocbe_parameters = sum(p.numel() for p in ocbe.parameters())
    student_parameters = sum(p.numel() for p in student.parameters())
    parameters = ocbe_parameters + student_parameters
    assert parameters == metric["trainable_parameters"] == config["resources"]["expected_trainable_parameters"]
    verification = {
        "epoch_sequence": "1..30; no missing epochs", "max_steps": 0,
        "scheduler_returncode": 0, "frames": len(rows), "positive_frames": int(labels.sum()),
        "labels_and_frame_order": "exact match against dataset and all five variants",
        "finite_scores": True, "recomputed_auc_percent": auc, "recomputed_eer": eer,
        "metrics_match_logged_result": True, "checkpoint_sha256_matches": True,
        "checkpoint_strict_load": True, "checkpoint_finite": all(
            torch.isfinite(t).all().item() for state in
            (checkpoint["ocbe_state_dict"], checkpoint["student_state_dict"]) for t in state.values()),
        "ocbe_parameters": ocbe_parameters, "student_parameters": student_parameters,
        "source_commit": TRAIN_SOURCE,
        "scope": "post-hoc score/label/checkpoint checks; inference was not rerun",
    }
    assert verification["checkpoint_finite"]
    if hasattr(ocbe, "effective_depth_basis"):
        with torch.no_grad():
            q = ocbe.effective_depth_basis()
            verification["rank2"] = {
                "initialization": "random_orthonormal, seed42", "shared_groups": 4,
                "constraint": "differentiable Gram-Schmidt (no extra orthogonality loss)",
                "raw_q": ocbe.raw_depth_basis.tolist(), "effective_q": q.tolist(),
                "orthogonality_error_fro": float(torch.linalg.matrix_norm(q @ q.T - torch.eye(2))),
            }
    timing = {
        "method": name, "experiment_id": ident, "gpu_index_at_launch": launch["gpu"],
        "gpu_name_from_runtime": startup["cuda"], "epochs": 30,
        "train_s": epochs[-1]["elapsed_s"], "first_epoch_s": epochs[0]["epoch_s"],
        "median_epoch_2_30_s": statistics.median(e["epoch_s"] for e in epochs[1:]),
        "evaluation_and_save_s": metric["elapsed_s"] - epochs[-1]["elapsed_s"],
        "total_logged_s": metric["elapsed_s"],
        "process_wall_s": (timestamp(finish["time"]) - timestamp(launch["time"])).total_seconds(),
    }
    row = {"method": name, "experiment_id": ident, "auc_percent": auc, "eer_percent": eer * 100,
           "trainable_parameters": parameters, "parameters_m": parameters / 1e6,
           "ocbe_parameters": ocbe_parameters, "student_parameters": student_parameters,
           "train_h": timing["train_s"] / 3600,
           "checkpoint_mib": checkpoint_path.stat().st_size / 2**20}
    return directory, row, timing, epochs, verification, launch, finish


def main():
    started = dt.datetime.now(dt.timezone.utc)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = read_json(HOST / "config.json")
    data_root = Path(config["training"]["data_root"])
    expected_rows = []
    for video in sorted((data_root / "test/frames").iterdir()):
        if not video.is_dir():
            continue
        paths = sorted(video.glob("*.jpg"), key=frame_key)
        labels = np.load(data_root / "test/test_frame_mask" / f"{video.name}.npy").astype(int).ravel()
        assert len(paths) == len(labels)
        expected_rows.extend((video.name, p.name, int(y)) for p, y in zip(paths, labels))
    verified = [verify(name, ident, expected_rows) for name, ident in VARIANTS]
    rows = [v[1] for v in verified]
    timings = [v[2] for v in verified]
    for i, row in enumerate(rows):
        row["auc_delta_vs_previous_pp"] = 0.0 if not i else row["auc_percent"] - rows[i-1]["auc_percent"]
        row["auc_delta_vs_plain_pp"] = row["auc_percent"] - rows[0]["auc_percent"]
        row["parameter_reduction_vs_plain_percent"] = 100 * (1 - row["trainable_parameters"] / rows[0]["trainable_parameters"])
    write_csv(HOST / "results/five_variant_comparison.csv", rows)
    write_csv(HOST / "results/five_variant_training_timing.csv", timings)
    lines = [
        "# UIT-ADrone 关键轻量化版本对比：seed42 / 30 epoch", "",
        "五个版本均完整训练 30 epoch，调度器返回码为 0。指标由保存的逐帧分数独立复算；最终权重 SHA-256、严格加载和参数量已核验。", "",
        "## 统一口径", "",
        "仅外观分支。41 个 train 视频 / 59,186 帧；51 个 test 视频 / 147,005 帧，其中异常帧 63,485、正常帧 83,520。",
        "教师为冻结 DINOv3 ViT-B/16，输入 256×256，12 层 patch taps；相邻三层对应一个学生 stage，反向对齐不变。",
        "AdamW、OneCycleLR、batch64、top-5 group-balanced cosine loss（alpha=0.5）。使用完整 epoch：drop_last 后每轮 924 个 batch / 59,136 帧，30 轮共 27,720 个更新。",
        "异常分数越高越异常：各教师层取 patch 余弦距离最大的 5 个取均值，再对 12 层取均值。整个 test 直接拼接计算一个全局 AUC/EER，不按视频归一化、不平滑、不选取最好 epoch。EER 采用离散 ROC 上 |FNR-FPR| 最小点的 FPR。", "",
        "保留历史 UIT 官方 train/test 使用方式，不对 test 中可能与 train 重叠的帧去重。本批次没有重新审计这种重叠。Params 是 OCBE + Student 的可训练参数，不含冻结教师。", "",
        "## 结果", "",
        "| 方法 | AUC↑ (%) | EER↓ (%) | Params (M) | 相对前一版本 AUC (百分点) | 相对 4block 减参 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f'| {row["method"]} | {row["auc_percent"]:.4f} | {row["eer_percent"]:.4f} | {row["parameters_m"]:.6f} | {row["auc_delta_vs_previous_pp"]:+.4f} | {row["parameter_reduction_vs_plain_percent"]:.2f}% |')
    lines += ["", "## 观察与解释", "",
        f'1. GL768 相对普通 4block 的 AUC 下降 {-rows[1]["auc_delta_vs_previous_pp"]:.4f} 个百分点，参数减少 {rows[1]["parameter_reduction_vs_plain_percent"]:.2f}%。',
        f'2. GL768 → GL192 的 AUC 下降 {-rows[2]["auc_delta_vs_previous_pp"]:.4f} 个百分点，是这条迭代链最大的掉点来源。相同 30 epoch 下，这说明学生宽度压缩在本次 UIT 设置中的代价明显；不能把 Drone-Anomaly 上的表现直接外推到 UIT。',
        f'3. Dense512 → Dense128 再下降 {-rows[3]["auc_delta_vs_previous_pp"]:.4f} 个百分点，参数从 {rows[2]["parameters_m"]:.3f}M 降至 {rows[3]["parameters_m"]:.3f}M。',
        f'4. Dense128 → Rank2 的 AUC 仅变化 {rows[4]["auc_delta_vs_previous_pp"]:+.4f} 个百分点，同时继续减少 {rows[3]["trainable_parameters"]-rows[4]["trainable_parameters"]:,} 个参数。本次 Rank2 为共享、随机正交初始化、可学习的 2×3 Q，不是固定 Mean+Trend，也不是 Group-Block LR。',
        f'5. 最轻版本相对普通 4block 共减少 {rows[4]["parameter_reduction_vs_plain_percent"]:.2f}% 参数，但 AUC 下降 {-rows[4]["auc_delta_vs_plain_pp"]:.4f} 个百分点。因此本批次支持 Rank2 的额外压缩代价很小，不支持整条轻量化链在 UIT 上几乎无损的说法。',
        "这些是单个 seed42 的观察值。尚未估计多种子方差，不能把 0.023 个百分点的差距解释为稳定优劣。最终蒸馏 loss 也不等价于 AUC。", "",
        "## 训练耗时（仅日志观察，非速度基准）", "",
        "| 方法 | GPU（运行时名称） | 30 轮训练 (h) | 第 1 轮 (s) | 第 2–30 轮中位数 (s) | 评测与保存 (h) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for t in timings:
        lines.append(f'| {t["method"]} | {t["gpu_name_from_runtime"]} | {t["train_s"]/3600:.3f} | {t["first_epoch_s"]:.1f} | {t["median_epoch_2_30_s"]:.1f} | {t["evaluation_and_save_s"]/3600:.3f} |')
    lines += ["", "GPU 型号不同，双任务共享存储与 CPU，轮间耗时波动明显；上述时间包含训练读取和冻结教师计算，不能据此比较推理 FPS/Latency。评测还调用了全部 anomaly_readouts 后选择 topk，没有使用 Q 推理缓存或融合内核；耗时不代表优化后的部署速度。", "",
        "## 权重与追溯", "",
        f'训练源码快照：`{TRAIN_SOURCE}`；评审脚本源码：`{source}`。调度器 run_id：`{SCHEDULER_ID}`。',
        "旧 wrapper 失败和 retry1 中断均保留；本报告只使用 retry2。未新增训练，也未用已有 ckpt 再微调。", "",
        "| 实验目录 | 权重（本地，Git 忽略） | SHA-256 |", "|---|---|---|",
    ]
    for directory, row, *_ in verified:
        metric = read_json(directory / "results/uit.json")
        relative = "../../" + directory.name
        lines.append(f'| [{directory.name}]({relative}/README.md) | `{directory.relative_to(ROOT)}/ckpt/uit_final.pth` | `{metric["checkpoint_sha256"]}` |')
    (HOST / "results/five_variant_summary.md").write_text("\n".join(lines) + "\n")
    ended = dt.datetime.now(dt.timezone.utc)
    report_run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_uit_matrix_review"
    for directory, row, timing, epochs, verification, launch, finish in verified:
        verification["review_source_commit"] = source
        write_json(directory / "results/verification.json", verification)
        write_csv(directory / "results/training_timing.csv", epochs)
        if directory == HOST:
            summary = "[五版本完整对比](five_variant_summary.md)"
        else:
            summary = f"[五版本完整对比](../../{HOST.name}/results/five_variant_summary.md)"
        (directory / "results/summary.md").write_text(
            f'# {row["method"]}：UIT-ADrone，seed42，30 epoch\n\n'
            f'AUC **{row["auc_percent"]:.4f}%**，EER **{row["eer_percent"]:.4f}%**，OCBE + Student 参数 **{row["parameters_m"]:.6f}M**。\n\n'
            f'完整 30 轮，逐帧指标复算、标签对齐、权重哈希、严格加载均通过。训练日志耗时 {timing["train_s"]/3600:.3f} 小时，不能作为推理速度对比。\n\n'
            f'{summary}；[核验记录](verification.json)；[逐轮耗时](training_timing.csv)。\n\n'
            '最终权重：`ckpt/uit_final.pth`（本地保存，不入 Git）。\n')
        record = read_json(directory / "record.json")
        record["status"] = "completed"
        record["weight_files_local"] = 1
        record["result_summary"] = "results/summary.md"
        record["review"] = row
        if directory == HOST and "scripts/review_matrix.py" not in record["scripts"]:
            record["scripts"].append("scripts/review_matrix.py")
        run = record["runs"][0]
        run.update(status="completed", started_at=timestamp(launch["time"]).isoformat(),
                   ended_at=timestamp(finish["time"]).isoformat(), command=launch["command"],
                   log="logs/" + Path(launch["log"]).name, scheduler_log=SCHEDULER_LOG,
                   outputs=["results/uit.json", "results/uit_scores.csv", "ckpt/uit_final.pth"],
                   gpu_index_at_launch=launch["gpu"], gpu_name=timing["gpu_name_from_runtime"],
                   data_root=str(data_root), split_fingerprint=read_json(directory / "results/uit.json")["split_fingerprint"],
                   notes="Detached retry2 completed 30 full epochs; returncode=0. No checkpoint warm start.")
        record["runs"] = [run, {"run_id": report_run_id + "_" + directory.name,
            "status": "completed", "source_commit": source,
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "seed": 42, "started_at": started.isoformat(), "ended_at": ended.isoformat(),
            "command": [str(Path(__file__).resolve().relative_to(ROOT))],
            "outputs": ["results/summary.md", "results/verification.json", "results/training_timing.csv"],
            "notes": "CPU post-hoc review, no training or inference rerun."}]
        if directory == HOST:
            record["runs"][-1]["outputs"] += ["results/five_variant_summary.md", "results/five_variant_comparison.csv", "results/five_variant_training_timing.csv"]
        write_json(directory / "record.json", record)
        artifacts = []
        for folder in ("logs", "results", "ckpt"):
            for path in sorted((directory / folder).glob("*")):
                if not path.is_file():
                    continue
                rel = str(path.relative_to(directory))
                derived = folder == "results" and path.name not in {"uit.json", "uit_scores.csv"}
                kind = "checkpoint" if folder == "ckpt" else "log" if folder == "logs" else "result"
                artifacts.append({"path": rel, "kind": kind,
                    "producer_script": str(Path(__file__).resolve().relative_to(ROOT)) if derived else "scripts/uit_variant_scheduler.py" if folder == "logs" else "scripts/uit_appearance_runner.py",
                    "run_id": report_run_id + "_" + directory.name if derived else run["run_id"],
                    "inputs": "config.json; logs; results/uit_scores.csv; ckpt/uit_final.pth" if derived else "config.json; UIT-ADrone official split",
                    "status": "local_only" if folder == "ckpt" else "completed",
                    "size_bytes": path.stat().st_size, "sha256": sha256(path),
                    "storage": "ignored_local" if folder == "ckpt" else "git"})
        write_json(directory / "artifacts.json", {"schema_version": 2, "artifacts": artifacts})
        print(f'{row["method"]}: AUC={row["auc_percent"]:.4f}, EER={row["eer_percent"]:.4f}, Params={row["parameters_m"]:.6f}M', flush=True)


if __name__ == "__main__":
    main()
