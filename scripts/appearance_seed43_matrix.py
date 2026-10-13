#!/usr/bin/env python3
"""Run four seed-43 appearance variants across seven scenes on two dynamic GPU workers."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCENES = ("bike", "crossroads", "farmland", "highway", "railway", "solar", "vehicle")
EXPERIMENTS = (
    "20261013_seed43_glll768_ocbe512_s43",
    "20261013_seed43_glll192_ocbe512_s43",
    "20261013_seed43_glll192_ocbe128_s43",
    "20261013_seed43_glll192_ocbe128_rank2_s43",
)


def stamp():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def gpu_memory(gpus):
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
         "--format=csv,noheader,nounits"], text=True)
    found = {}
    for line in raw.strip().splitlines():
        index, used, total = (int(x.strip()) for x in line.split(","))
        if index in gpus:
            found[index] = {"used_mb": used, "total_mb": total}
    if set(found) != set(gpus):
        raise RuntimeError(f"missing GPU: requested={gpus}, found={sorted(found)}")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--python", default="/home/lyx/miniconda3/envs/DMAD/bin/python")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--max-memory-mb", type=int, default=2000)
    ap.add_argument("--poll-seconds", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = args.repo_root.resolve()
    base = root / "experiments" / "da" / "appearance"
    experiments = {name: base / name for name in EXPERIMENTS}
    for name, directory in experiments.items():
        cfg = json.loads((directory / "config.json").read_text())
        if cfg["id"] != name or cfg["seed"] != 43 or cfg["student"]["stages"] != 4:
            raise ValueError(f"unexpected experiment config: {name}")
        (directory / "logs").mkdir(parents=True, exist_ok=True)
        (directory / "results").mkdir(parents=True, exist_ok=True)
    memory = gpu_memory(args.gpus)
    busy = {gpu: stats for gpu, stats in memory.items()
            if stats["used_mb"] > args.max_memory_mb}
    if busy:
        raise RuntimeError(f"GPU memory above start threshold: {busy}")
    queue = [(name, scene) for name in EXPERIMENTS for scene in SCENES]
    if args.dry_run:
        print(json.dumps({"jobs": len(queue), "gpus": memory,
                          "initial_jobs": queue[:len(args.gpus)]}, ensure_ascii=False))
        return 0

    run_id = stamp() + "_seed43_four_variant_matrix"
    scheduler_logs = {name: directory / "logs" / f"scheduler_{run_id}.jsonl"
                      for name, directory in experiments.items()}

    def emit(kind, targets=None, **payload):
        event = {"time": stamp(), "run_id": run_id, "kind": kind, **payload}
        line = json.dumps(event, ensure_ascii=False) + "\n"
        for name in (targets or EXPERIMENTS):
            with scheduler_logs[name].open("a") as f:
                f.write(line)
        print(line, end="", flush=True)

    emit("gpu_probe", gpus=args.gpus, status=memory, queue_size=len(queue),
         max_memory_mb=args.max_memory_mb, poll_seconds=args.poll_seconds)
    running = {}
    failures = []

    def launch(gpu, name, scene):
        directory = experiments[name]
        start = stamp()
        log_path = directory / "logs" / f"{start}_{scene}_gpu{gpu}.log"
        command = [args.python, str(directory / "scripts" / "run_scene.py"),
                   "--scene", scene, "--gpu", str(gpu), "--output", str(directory),
                   "--seed", "43", "--epochs", "15", "--batch", "64", "--workers", "8"]
        log = log_path.open("w")
        try:
            process = subprocess.Popen(command, cwd=root, stdout=log,
                                       stderr=subprocess.STDOUT, env=os.environ.copy())
        except Exception:
            log.close()
            raise
        running[gpu] = (name, scene, process, log, log_path)
        emit("launch", targets=(name,), experiment=name, scene=scene, gpu=gpu,
             pid=process.pid, log=str(log_path.relative_to(directory)), command=command)

    for gpu in args.gpus:
        if queue:
            launch(gpu, *queue.pop(0))
    while running:
        for gpu, state in list(running.items()):
            name, scene, process, log, log_path = state
            returncode = process.poll()
            if returncode is None:
                continue
            log.close()
            emit("finish", targets=(name,), experiment=name, scene=scene, gpu=gpu,
                 pid=process.pid, returncode=returncode,
                 log=str(log_path.relative_to(experiments[name])))
            if returncode:
                failures.append({"experiment": name, "scene": scene,
                                 "gpu": gpu, "returncode": returncode})
            del running[gpu]
            if queue:
                launch(gpu, *queue.pop(0))
        if running:
            time.sleep(args.poll_seconds)
    emit("all_finished", total_jobs=len(EXPERIMENTS) * len(SCENES), failures=failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
