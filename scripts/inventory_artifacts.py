#!/usr/bin/env python3
"""Snapshot legacy binary artifact locations; never reads model contents."""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'LOCAL_ARTIFACTS.csv'
BINARY = {'.pth', '.pt', '.ckpt', '.safetensors', '.onnx', '.mp4', '.avi', '.mov', '.npz', '.npy', '.pkl', '.png', '.jpg', '.jpeg'}
SKIP = {'.git', '__pycache__', '.vscode', '.claude'}
TRACKED = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))


def producer(path: str) -> str:
    if path.startswith('Visualization/output/'):
        return 'Visualization/visualize_appearance.py (path documented; run commit unknown)'
    if path.startswith('detect_preview/'):
        return 'behavior/detect_vis.py (path documented; run commit unknown)'
    if path.startswith('audit/figs/'):
        return 'audit/cloud_dist.py (path documented; run commit unknown)'
    if path.startswith('experiments/fusion/scores/app_'):
        return 'fusion/dump_appearance_scores.py (path documented; run commit unknown)'
    if path.startswith('experiments/fusion/scores/beh_'):
        return 'fusion/dump_behavior_scores.py (path documented; run commit unknown)'
    return 'unknown'


rows = []
for path in ROOT.rglob('*'):
    if any(part in SKIP for part in path.relative_to(ROOT).parts):
        continue
    if not path.is_file() or path.suffix.lower() not in BINARY:
        continue
    relative = path.relative_to(ROOT).as_posix()
    rows.append({
        'path': relative,
        'size_bytes': path.stat().st_size,
        'git_storage': 'tracked' if relative in TRACKED else 'local_only',
        'producer': producer(relative),
        'run_commit': 'unknown',
        'sha256': '',
    })
rows.sort(key=lambda r: r['path'])
with OUT.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['path', 'size_bytes', 'git_storage', 'producer', 'run_commit', 'sha256'], lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
print(f'{len(rows)} binary artifacts indexed in {OUT.relative_to(ROOT)}')
