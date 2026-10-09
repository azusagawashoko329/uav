#!/usr/bin/env python3
"""Validate the nested experiment registry without importing model code."""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'
STATES = {'planned','running','completed','failed','cancelled','unknown'}
DATASETS = {'da','uit','da_uit'}
FAMILIES = {'appearance','behavior','fusion','aggregation','boundary','gating','audit'}
NEW_ID = re.compile(r'^\d{8}_[a-z0-9]+(?:_[a-z0-9]+)*_s\d+$')
REQUIRED = {'schema_version','id','path','dataset','family','status','source_commit','scripts','runs','result_summary','limitations'}
errors=[]
records=[]
for path in sorted(EXP.glob('*/**/record.json')):
    if 'source_snapshot_' in str(path):continue
    d=path.parent;rel=d.relative_to(ROOT).as_posix()
    try:data=json.loads(path.read_text())
    except (OSError,ValueError) as exc:
        errors.append(f'{rel}: bad JSON: {exc}');continue
    records.append(rel)
    missing=REQUIRED-data.keys()
    if missing:errors.append(f'{rel}: missing {sorted(missing)}')
    if data.get('id')!=d.name or data.get('path')!=rel:errors.append(f'{rel}: id/path mismatch')
    if data.get('dataset') not in DATASETS or data.get('family') not in FAMILIES:errors.append(f'{rel}: dataset/family invalid')
    if data.get('status') not in STATES:errors.append(f'{rel}: status invalid')
    if not data.get('legacy') and not NEW_ID.fullmatch(d.name):errors.append(f'{rel}: new ID does not follow naming rule')
    commit=data.get('source_commit')
    if commit is not None and not re.fullmatch(r'[0-9a-f]{40}',str(commit)):errors.append(f'{rel}: source_commit must be SHA-40 or null')
    for script in data.get('scripts',[]):
        if not (d/script).is_file():errors.append(f'{rel}: missing script {script}')
    if not data.get('legacy'):
        for name in ('README.md','config.json','artifacts.json'):
            if not (d/name).is_file():errors.append(f'{rel}: missing {name}')
        if not (d/'scripts').is_dir():errors.append(f'{rel}: missing scripts/')
    for run in data.get('runs',[]):
        if run.get('status') not in STATES:errors.append(f'{rel}: run status invalid')

catalog=EXP/'CATALOG.csv'
if not catalog.exists():errors.append('experiments/CATALOG.csv missing')
else:
    with catalog.open() as f:catalog_paths={r['path'] for r in csv.DictReader(f)}
    expected={x.removeprefix('experiments/') for x in records}
    if catalog_paths!=expected:errors.append(f'catalog mismatch: missing={sorted(expected-catalog_paths)}, stale={sorted(catalog_paths-expected)}')
tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
for name in filter(None,tracked):
    if Path(name).suffix.lower() in {'.pth','.pt','.ckpt','.safetensors','.onnx','.mp4','.avi','.mov'}:
        errors.append(f'tracked large artifact: {name}')
if errors:
    print('Experiment registry check failed:',file=sys.stderr)
    for item in errors:print(' - '+item,file=sys.stderr)
    sys.exit(1)
print(f'Experiment registry OK: {len(records)} records')
