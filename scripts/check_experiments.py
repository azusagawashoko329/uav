#!/usr/bin/env python3
"""Check project experiment bookkeeping without importing model dependencies."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
ID = re.compile(r"^\d{8}_(?:da|uit|da_uit)_[a-z0-9]+(?:_[a-z0-9]+)*_s\d+$")
STATES = {"planned", "running", "completed", "failed", "cancelled", "unknown"}
DATASETS = {"da", "uit", "da_uit"}
REQUIRED = {"schema_version", "id", "dataset", "question", "variant", "status", "source_commit", "scripts", "runs", "result_summary", "limitations"}
errors = []

for exp in sorted(p for p in EXPERIMENTS.iterdir() if p.is_dir()):
    if ID.fullmatch(exp.name):
        rec = exp / "record.json"
        for name in ("README.md", "record.json", "config.json", "artifacts.json"):
            if not (exp / name).is_file():
                errors.append(f"{exp.name}: missing {name}")
        if not (exp / "scripts").is_dir() or not any((exp / "scripts").glob("*.*")):
            errors.append(f"{exp.name}: missing scripts/")
    else:
        rec = exp / "record.json"
        if exp.name != "fusion" and not (exp / "RECORD.md").is_file() and not rec.is_file():
            errors.append(f"{exp.name}: legacy directory needs RECORD.md")
    if not rec.is_file():
        continue
    try:
        data = json.loads(rec.read_text())
    except (ValueError, OSError) as exc:
        errors.append(f"{rec.relative_to(ROOT)}: invalid JSON: {exc}")
        continue
    missing = REQUIRED - data.keys()
    if missing:
        errors.append(f"{exp.name}: missing fields {sorted(missing)}")
    if data.get("id") != exp.name:
        errors.append(f"{exp.name}: record id mismatch")
    if data.get("dataset") not in DATASETS:
        errors.append(f"{exp.name}: invalid dataset")
    if data.get("status") not in STATES:
        errors.append(f"{exp.name}: invalid status")
    commit = data.get("source_commit")
    if commit is not None and not re.fullmatch(r"[0-9a-f]{40}", str(commit)):
        errors.append(f"{exp.name}: source_commit must be full SHA or null")
    for script in data.get("scripts", []):
        if not isinstance(script, str) or not (exp / script).is_file():
            errors.append(f"{exp.name}: missing script {script}")
    for run in data.get("runs", []):
        if run.get("status") not in STATES:
            errors.append(f"{exp.name}: invalid run status")

tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
for name in filter(None, tracked):
    path = Path(name)
    if path.suffix.lower() in {".pth", ".pt", ".ckpt", ".safetensors", ".onnx", ".mp4", ".avi", ".mov"}:
        errors.append(f"tracked large artifact: {name}")

if errors:
    print("Experiment record check failed:", file=sys.stderr)
    for error in errors:
        print(" - " + error, file=sys.stderr)
    sys.exit(1)
print("Experiment records and tracked-artifact rules: OK")
