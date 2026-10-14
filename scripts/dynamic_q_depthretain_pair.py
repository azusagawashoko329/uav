#!/usr/bin/env python3
"""Run paired Q-initialization scene jobs on the same GPU, refilling idle workers."""
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


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def gpu_probe(gpus: list[int]) -> dict[str, dict[str, int]]:
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                                   "--format=csv,noheader,nounits"], text=True)
    result = {}
    for line in raw.strip().splitlines():
        idx, used, total = [int(item.strip()) for item in line.split(",")]
        if idx in gpus:
            result[str(idx)] = {"used_mb": used, "total_mb": total}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--fixed-experiment", type=Path, required=True)
    parser.add_argument("--random-experiment", type=Path, required=True)
    parser.add_argument("--python", default="/home/lyx/miniconda3/envs/DMAD/bin/python")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--max-memory-mb", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    experiments = {"fixed_init": args.fixed_experiment.resolve(),
                   "random_init": args.random_experiment.resolve()}
    for exp in experiments.values():
        if not (exp / "scripts/run_ocbe.py").is_file():
            raise FileNotFoundError(exp / "scripts/run_ocbe.py")
        (exp / "logs").mkdir(parents=True, exist_ok=True)
    stamp = now()
    logs = [exp / "logs" / f"scheduler_{stamp}.jsonl" for exp in experiments.values()]

    def emit(kind: str, **payload) -> None:
        event = {"time": now(), "kind": kind, **payload}
        line = json.dumps(event, ensure_ascii=False) + "\n"
        for path in logs:
            with path.open("a") as stream:
                stream.write(line)
        print(line, end="", flush=True)

    probe = gpu_probe(args.gpus)
    emit("gpu_probe", gpus=args.gpus, status=probe, max_memory_mb=args.max_memory_mb,
         poll_seconds=args.poll_seconds, scheduler_stamp=stamp)
    if any(str(gpu) not in probe for gpu in args.gpus):
        raise RuntimeError("requested GPU missing from nvidia-smi")
    busy = {gpu: probe[str(gpu)]["used_mb"] for gpu in args.gpus
            if probe[str(gpu)]["used_mb"] > args.max_memory_mb}
    if busy:
        raise RuntimeError(f"requested GPUs busy: {busy}")
    if args.dry_run:
        emit("dry_run_ok", scenes=SCENES,
             initial_assignments={scene: args.gpus[i % len(args.gpus)] for i, scene in enumerate(SCENES)},
             pairs_per_scene=["fixed_init", "random_init"])
        return 0

    pending = list(SCENES)
    active = {}
    failures = []

    def launch(gpu: int, scene: str, variant: str) -> None:
        exp = experiments[variant]
        path = exp / "logs" / f"{now()}_{scene}_gpu{gpu}.log"
        command = [args.python, str(exp / "scripts/run_ocbe.py"),
                   "--scene", scene, "--gpu", str(gpu), "--output", str(exp),
                   "--seed", str(args.seed), "--epochs", str(args.epochs),
                   "--batch", str(args.batch), "--workers", str(args.workers)]
        stream = path.open("w")
        proc = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, "UAV_OCBE_GPU": str(gpu)})
        active[gpu] = {"scene": scene, "variant": variant, "proc": proc,
                       "stream": stream, "log": path}
        emit("launch", gpu=gpu, scene=scene, variant=variant, pid=proc.pid,
             log=str(path), command=command)

    for gpu in args.gpus:
        if pending:
            launch(gpu, pending.pop(0), "fixed_init")
    while active:
        for gpu, state in list(active.items()):
            code = state["proc"].poll()
            if code is None:
                continue
            state["stream"].close()
            emit("finish", gpu=gpu, scene=state["scene"], variant=state["variant"],
                 pid=state["proc"].pid, returncode=code, log=str(state["log"]))
            if code:
                failures.append({"scene": state["scene"], "variant": state["variant"],
                                 "gpu": gpu, "returncode": code})
            del active[gpu]
            if state["variant"] == "fixed_init":
                launch(gpu, state["scene"], "random_init")
            elif pending:
                launch(gpu, pending.pop(0), "fixed_init")
        if active:
            time.sleep(args.poll_seconds)
    emit("all_finished", scenes=SCENES, failures=failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
