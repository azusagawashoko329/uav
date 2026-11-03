import json, os, subprocess, sys, time
from pathlib import Path
root = Path(__file__).resolve().parents[1]
scenes = ["bike", "crossroads", "farmland"]
python = sys.executable
for scene in scenes:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log = root / "logs" / f"{stamp}_{scene}_gpu0.log"
    cmd = [python, str(root / "scripts" / "run_scene.py"), "--scene", scene, "--gpu", "0", "--output", str(root), "--seed", "42", "--epochs", "30", "--batch", "64", "--workers", "8", "--variant", "plain12"]
    with log.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(Path(__file__).resolve().parents[5]))
    summary = {"time": stamp, "scene": scene, "returncode": proc.returncode, "log": str(log)}
    with (root / "logs" / "scheduler.jsonl").open("a") as fh:
        fh.write(json.dumps(summary) + "\n")
    if proc.returncode:
        raise SystemExit(proc.returncode)
