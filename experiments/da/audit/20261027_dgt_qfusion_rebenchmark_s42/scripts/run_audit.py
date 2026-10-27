"""Dynamic two-GPU scheduler and report builder for the DGT Q remeasurement."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
EXP = Path(__file__).resolve().parents[1]
PYTHON = "/home/lyx/miniconda3/envs/DMAD/bin/python"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_record(record):
    dump(EXP / "record.json", record)


def launch(variant, gpu, config):
    log = EXP / "logs" / f"{variant}_gpu{gpu}.log"
    affinity = "20-23" if gpu == 0 else "0-3"
    command = ["taskset", "-c", affinity, PYTHON, "-u",
               str((EXP / "scripts" / "benchmark.py").relative_to(ROOT)),
               "--gpu", str(gpu), "--variant", variant]
    fh = log.open("w")
    proc = subprocess.Popen(command, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    return proc, fh, dict(variant=variant, gpu=gpu, affinity=affinity,
                          command=command, log=str(log.relative_to(EXP)),
                          started_at=stamp(), status="running", process=proc)


def flatten(config, results):
    rows = []
    for variant, result in results.items():
        for scene in result["scenes"]:
            for batch, scopes in scene["timings"].items():
                for scope, modes in scopes.items():
                    for condition, metric in modes.items():
                        rows.append(dict(variant=variant, label=result["label"],
                                         gpu=result["environment"]["gpu"], scene=scene["scene"],
                                         batch=int(batch), scope=scope, condition=condition,
                                         **metric))
    return rows


def report(config, results):
    rows = flatten(config, results)
    with (EXP / "results" / "measurements.csv").open("w", newline="") as f:
        fields = list(rows[0])
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary = {"configuration": config, "variants": results,
               "report_generated_at": stamp()}
    dump(EXP / "results" / "measurements.json", summary)

    lines = ["# DGT及后续融合版本：Rank2 Q执行优化后的统一测速", "",
             "主表统计七场景、batch=1、OCBE＋GL192学生前向；不含教师、预处理、H2D、评分。每个条件在同一模型、同一场景、同一输入下配对测量。", "",
             "## batch=1：七场景p50均值", "",
             "| 版本 | GPU | 原循环 ms/FPS | 缓存Q原循环 ms/FPS | 融合Q ms/FPS | 融合相对原循环 |", "|---|---|---:|---:|---:|---:|"]
    for spec in config["variants"]:
        result = results[spec["id"]]
        vals = {k: [] for k in ("original", "cached_loop", "fused")}
        for scene in result["scenes"]:
            for condition in vals:
                vals[condition].append(scene["timings"]["1"]["subnet"][condition])
        mean_ms = {k: sum(x["latency_p50_ms"] for x in v) / len(v) for k, v in vals.items()}
        fps = {k: 1000 / v for k, v in mean_ms.items()}
        gain = (fps["fused"] / fps["original"] - 1) * 100
        lines.append(f"| {spec['label']} (`{spec['id']}`) | GPU{result['environment']['gpu']} | "
                     f"{mean_ms['original']:.4f} / {fps['original']:.1f} | "
                     f"{mean_ms['cached_loop']:.4f} / {fps['cached_loop']:.1f} | "
                     f"{mean_ms['fused']:.4f} / {fps['fused']:.1f} | {gain:+.2f}% |")
    lines += ["", "## 逐场景融合结果", "", "| 版本 | 场景 | GPU | 原循环 FPS | 缓存循环 FPS | 融合 FPS | 融合相对原循环 |\n|---|---|---|---:|---:|---:|---:|"]
    for spec in config["variants"]:
        result = results[spec["id"]]
        for scene in result["scenes"]:
            t = scene["timings"]["1"]["subnet"]
            a, b, c = (t[k]["serial_fps"] for k in ("original", "cached_loop", "fused"))
            lines.append(f"| {spec['id']} | {scene['scene']} | GPU{result['environment']['gpu']} | {a:.1f} | {b:.1f} | {c:.1f} | {(c/a-1)*100:+.2f}% |")
    lines += ["", "## 解释与边界", "",
              "原实现的每组三层Q投影被展开为逐元素小算子；缓存Q只消除了每帧Gram–Schmidt，仍保留乘法、加法、select和临时张量。融合路径将四组投影合并为单个CUDA kernel；DGT变体按[group,batch,patch,2×768]输出，保持原有每组Linear的输入形状。Q系数、有效方向、模型权重、loss和学生结构没有改变。", "",
              "所有场景均核对原循环、缓存循环和融合路径的code、12个student重建和异常分数；最大绝对误差写入各GPU JSON。训练诊断在部署测速中显式关闭，旧版本和新版本口径一致；关闭诊断本身不作为Q融合收益。", "",
              "参数、主要GFLOPs、Model Size和历史AUC沿用旧审计并在config中重新核对；本审计只重新测Latency/FPS。旧完整checkpoint缺失的DGT主路径与DenseRank2记录不补造AUC。不同GPU的绝对值不混合比较，正式主表保留GPU列。", "",
              "启用：`set_diagnostics_enabled(False)`（DGT变体）→ `eval()` → `cache_depth_basis_for_inference(fuse_projection=True)` → `torch.inference_mode()`。训练及梯度开启的eval仍走原实现。"]
    (EXP / "results" / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    config = json.loads((EXP / "config.json").read_text())
    record = json.loads((EXP / "record.json").read_text())
    record.update(status="running", source_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    record["runs"] = []
    write_record(record)
    queue = [x["id"] for x in config["variants"]]
    running = {}
    results = {}
    while queue or running:
        for gpu in (0, 1):
            if gpu in running or not queue:
                continue
            variant = queue.pop(0)
            proc, fh, meta = launch(variant, gpu, config)
            meta["process"] = proc
            running[gpu] = (variant, proc, fh, meta)
            record["runs"].append({k: v for k, v in meta.items() if k != "process"})
            write_record(record)
        time.sleep(2)
        for gpu, state in list(running.items()):
            variant, proc, fh, meta = state
            if proc.poll() is None:
                continue
            fh.close()
            result_file = EXP / "results" / f"{variant}_gpu{gpu}.json"
            run = next(x for x in record["runs"] if x["variant"] == variant and x["gpu"] == gpu)
            run.update(ended_at=stamp(), exit_code=proc.returncode,
                       status="completed" if proc.returncode == 0 else "failed",
                       outputs=[str(result_file.relative_to(EXP))] if proc.returncode == 0 else [])
            if proc.returncode != 0:
                record.update(status="failed")
                run["reason"] = f"worker exit {proc.returncode}"
                write_record(record)
                raise RuntimeError(f"worker failed: {variant} GPU{gpu}; see {meta['log']}")
            results[variant] = json.loads(result_file.read_text())
            del running[gpu]
            write_record(record)
    report(config, results)
    record.update(status="completed", result_summary="results/summary.md")
    for run in record["runs"]:
        if run["status"] == "completed":
            run.setdefault("outputs", []).append("results/measurements.json")
    write_record(record)
    print(json.dumps({"event": "completed", "variants": len(results)}), flush=True)


if __name__ == "__main__":
    main()
