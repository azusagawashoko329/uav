#!/usr/bin/env python3
"""Dynamic two-GPU scene scheduler.

Each worker owns one PyTorch CUDA device. When a scene process exits, the next queued scene is
started immediately on that same worker. The scheduler does not use fixed-duration turns.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCENES = ["bike", "crossroads", "farmland", "highway", "railway", "solar", "vehicle"]


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def gpu_probe(gpus):
    cmd = ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
           "--format=csv,noheader,nounits"]
    raw = subprocess.check_output(cmd, text=True)
    result = {}
    for line in raw.strip().splitlines():
        idx, used, total = [int(x.strip()) for x in line.split(",")]
        if idx in gpus:
            result[str(idx)] = {"used_mb": used, "total_mb": total}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--python", default="/home/lyx/miniconda3/envs/DMAD/bin/python")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--poll-seconds", type=float, default=2.0)
    ap.add_argument("--max-memory-mb", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenes", nargs="+", choices=SCENES, default=SCENES)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.experiment = args.experiment.resolve()
    args.experiment.joinpath("logs").mkdir(parents=True, exist_ok=True)
    scheduler_log = args.experiment / "logs" / f"scheduler_{now()}.jsonl"
    events = []

    def emit(kind, **payload):
        event = {"time": now(), "kind": kind, **payload}
        events.append(event)
        with scheduler_log.open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(event, ensure_ascii=False), flush=True)

    probe = gpu_probe(args.gpus)
    emit("gpu_probe", gpus=args.gpus, status=probe,
         max_memory_mb=args.max_memory_mb, poll_seconds=args.poll_seconds)
    missing = [str(g) for g in args.gpus if str(g) not in probe]
    if missing:
        raise RuntimeError(f"requested GPUs missing from nvidia-smi: {missing}")
    busy = {g: probe[str(g)]["used_mb"] for g in args.gpus
            if probe[str(g)]["used_mb"] > args.max_memory_mb}
    if busy:
        raise RuntimeError(f"requested GPUs are busy above threshold: {busy}")
    if args.dry_run:
        assignments = {scene: args.gpus[i % len(args.gpus)] for i, scene in enumerate(args.scenes)}
        emit("dry_run_ok", scenes=args.scenes, initial_assignments=assignments)
        return 0

    queue = list(args.scenes)
    running = {}
    failures = []

    def launch(gpu, scene):
        stamp = now()
        log_path = args.experiment / "logs" / f"{stamp}_{scene}_gpu{gpu}.log"
        output = args.experiment
        cmd = [args.python, str(args.experiment / "scripts" / "run_baseline30.py"),
               "--scene", scene, "--gpu", str(gpu), "--output", str(output),
               "--seed", str(args.seed), "--epochs", str(args.epochs),
               "--batch", str(args.batch), "--workers", str(args.workers), "--variant", "baseline_rank2"]
        log = log_path.open("w")
        proc = subprocess.Popen(cmd, cwd=args.repo_root, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "UAV_OCBE_GPU": str(gpu)})
        running[gpu] = {"scene": scene, "proc": proc, "log": log, "log_path": str(log_path),
                        "started_at": stamp, "command": cmd}
        emit("launch", gpu=gpu, scene=scene, pid=proc.pid, log=str(log_path), command=cmd)

    # Initial workers are assigned only after the probe. Thereafter a completed worker is
    # reused immediately, which keeps both cards occupied while the queue is non-empty.
    for gpu in args.gpus:
        if queue:
            launch(gpu, queue.pop(0))
    while running:
        for gpu, state in list(running.items()):
            rc = state["proc"].poll()
            if rc is None:
                continue
            state["log"].close()
            emit("finish", gpu=gpu, scene=state["scene"], pid=state["proc"].pid,
                 returncode=rc, log=state["log_path"])
            if rc != 0:
                failures.append({"scene": state["scene"], "gpu": gpu, "returncode": rc})
            del running[gpu]
            if queue:
                launch(gpu, queue.pop(0))
        if running:
            time.sleep(args.poll_seconds)
    emit("all_finished", scenes=args.scenes, failures=failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
