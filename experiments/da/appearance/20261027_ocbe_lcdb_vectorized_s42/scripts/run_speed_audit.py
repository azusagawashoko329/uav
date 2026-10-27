"""Dynamic two-GPU scheduler and report for the pre-training structure audit."""
from __future__ import annotations

import csv
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
VARIANTS = ["dense128", "rank2", "dgt", "vectorized_dgt", "lcdb"]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def launch(variant, gpu):
    affinity = "20-23" if gpu == 0 else "0-3"
    log = EXP / "logs" / f"speed_{variant}_gpu{gpu}.log"
    cmd = ["taskset", "-c", affinity, PYTHON, "-u",
           str((EXP / "scripts" / "benchmark.py").relative_to(ROOT)),
           "--gpu", str(gpu), "--variant", variant]
    fh = log.open("w")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    return proc, fh, dict(variant=variant, gpu=gpu, affinity=affinity,
                          command=cmd, log=str(log.relative_to(EXP)),
                          started_at=stamp(), status="running", process=proc)


def report(config, results):
    rows = []
    for variant, result in results.items():
        for scene in result["scenes"]:
            for batch, scopes in scene["timings"].items():
                for scope, conditions in scopes.items():
                    for condition, metric in conditions.items():
                        rows.append(dict(variant=variant, gpu=result["environment"]["gpu"],
                                         scene=scene["scene"], batch=int(batch), scope=scope,
                                         condition=condition,
                                         **{k: v for k, v in metric.items() if k != "batch"}))
    with (EXP / "results" / "speed_measurements.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    dump(EXP / "results" / "speed_measurements.json", {"config": config, "results": results})
    lines = ["# Vectorized-DGT与LCDB速度审计", "",
             "batch=1，统计OCBE＋GL192 student；不包含teacher、预处理、H2D和异常评分。", "",
             "| 版本 | GPU | 原始Q ms/FPS | 缓存Q ms/FPS | 融合Q ms/FPS | 融合相对原始 |", "|---|---|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        result = results[variant]
        vals = {k: [s["timings"]["1"]["subnet"][k]["latency_p50_ms"] for s in result["scenes"]]
                for k in ("original", "cached_loop", "fused")}
        means = {k: sum(v) / len(v) for k, v in vals.items()}
        fps = {k: 1000 / v for k, v in means.items()}
        lines.append(f"| {variant} | GPU{result['environment']['gpu']} | "
                     f"{means['original']:.4f}/{fps['original']:.1f} | "
                     f"{means['cached_loop']:.4f}/{fps['cached_loop']:.1f} | "
                     f"{means['fused']:.4f}/{fps['fused']:.1f} | "
                     f"{(fps['fused']/fps['original']-1)*100:+.2f}% |")
    lines += ["", "## 解释", "", "Vectorized-DGT只改变组内Linear的批量执行，不改变参数和图传播；LCDB在四组1536维特征上先做固定Mean＋Trend，再做一次统一MLP。所有条件的code、student重建和score误差写在各variant JSON中。速度审计不产生AUC结论；只有通过结构审计的LCDB才进入训练。", ""]
    (EXP / "results" / "speed_summary.md").write_text("\n".join(lines))


def main():
    config = json.loads((EXP / "config.json").read_text())
    record = json.loads((EXP / "record.json").read_text())
    record.update(status="running", source_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    record["runs"] = []
    dump(EXP / "record.json", record)
    queue = list(VARIANTS); running = {}; results = {}
    while queue or running:
        for gpu in (0, 1):
            if gpu not in running and queue:
                variant = queue.pop(0); proc, fh, meta = launch(variant, gpu)
                running[gpu] = (variant, proc, fh, meta)
                record["runs"].append({k: v for k, v in meta.items() if k != "process"})
                dump(EXP / "record.json", record)
        time.sleep(2)
        for gpu, (variant, proc, fh, meta) in list(running.items()):
            if proc.poll() is None: continue
            fh.close(); run = next(x for x in record["runs"] if x["variant"] == variant and x["gpu"] == gpu)
            run.update(ended_at=stamp(), exit_code=proc.returncode,
                       status="completed" if proc.returncode == 0 else "failed",
                       outputs=[f"results/{variant}_gpu{gpu}.json"] if proc.returncode == 0 else [])
            if proc.returncode != 0:
                record["status"] = "failed"; run["reason"] = f"worker exit {proc.returncode}"
                dump(EXP / "record.json", record)
                raise RuntimeError(f"speed worker failed: {variant} GPU{gpu}")
            results[variant] = json.loads((EXP / "results" / f"{variant}_gpu{gpu}.json").read_text())
            del running[gpu]; dump(EXP / "record.json", record)
    report(config, results)
    record.update(status="completed", result_summary="results/speed_summary.md")
    dump(EXP / "record.json", record)
    print(json.dumps({"event": "speed_audit_completed", "variants": len(results)}), flush=True)


if __name__ == "__main__": main()
