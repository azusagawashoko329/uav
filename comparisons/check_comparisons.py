#!/usr/bin/env python3
"""Validate comparisons/<dataset>/<dataset>-<method>-<variant>/record.json.

No third-party dependency. Checks the v1 schema, internal source links, result
summaries, and CATALOG.csv consistency (run build_catalog.py first).
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / 'comparisons'
DATASETS = {'da', 'uit'}
STATUSES = {'planned', 'running', 'completed', 'failed', 'cancelled', 'blocked', 'unknown'}
ID_RE = re.compile(r'^(da|uit)-[a-z0-9]+(?:[-_][a-z0-9]+)*$')
METHOD_RE = re.compile(r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
VARIANT_RE = re.compile(r'^[a-z0-9]+(?:[-_][a-z0-9]+)*$')
SOURCE_KEYS = {'type', 'url', 'commit', 'local_path', 'experiment', 'run_id', 'license', 'notes'}
PROTOCOL_KEYS = {'one_class', 'train', 'test', 'metrics', 'seed', 'epochs', 'input_size', 'notes'}
RESULT_KEYS = {'macro_auc', 'macro_eer', 'per_scene', 'summary', 'source_results'}

errors = []
records = {}


def fail(path, msg):
    errors.append(f'{path}: {msg}')


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def close(a, b, tol=1e-6):
    return is_num(a) and is_num(b) and abs(a - b) <= tol


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        fail(path, f'bad JSON: {exc}')
        return None


def check_record(path: Path):
    rel = path.relative_to(ROOT).as_posix()
    data = load(path)
    if data is None:
        return None
    required = {'schema_version', 'id', 'dataset', 'method', 'method_display', 'year',
                'variant', 'status', 'source', 'protocol', 'results', 'notes'}
    missing = required - data.keys()
    if missing:
        fail(rel, f'missing keys {sorted(missing)}')
        return data
    if data.get('schema_version') != 1:
        fail(rel, 'schema_version must be 1')
    dataset = data.get('dataset')
    method = data.get('method')
    variant = data.get('variant')
    rid = data.get('id')
    if dataset not in DATASETS:
        fail(rel, f'invalid dataset {dataset!r}')
    if not isinstance(method, str) or not METHOD_RE.fullmatch(method):
        fail(rel, f'invalid method {method!r}')
    if not isinstance(variant, str) or not VARIANT_RE.fullmatch(variant):
        fail(rel, f'invalid variant {variant!r}')
    if not isinstance(rid, str) or not ID_RE.fullmatch(rid):
        fail(rel, f'invalid id {rid!r}')
    if rid != f'{dataset}-{method}-{variant}':
        fail(rel, f'id must equal {dataset}-{method}-{variant}')
    if path.parent.name != rid:
        fail(rel, f'directory name must equal id {rid!r}')
    if path.parent.parent.name != dataset:
        fail(rel, f'dataset directory mismatch: {path.parent.parent.name!r}')
    if data.get('status') not in STATUSES:
        fail(rel, f'invalid status {data.get("status")!r}')
    if not isinstance(data.get('method_display'), str) or not data['method_display'].strip():
        fail(rel, 'method_display must be non-empty string')
    year = data.get('year')
    if year is not None and (not isinstance(year, int) or isinstance(year, bool) or not 1990 <= year <= 2100):
        fail(rel, f'invalid year {year!r}')
    if not isinstance(data.get('notes'), str):
        fail(rel, 'notes must be a string')

    source = data.get('source')
    if not isinstance(source, dict):
        fail(rel, 'source must be an object')
        return data
    if set(source) != SOURCE_KEYS:
        fail(rel, f'source keys must be {sorted(SOURCE_KEYS)}')
    stype = source.get('type')
    if stype not in {'internal', 'external'}:
        fail(rel, f'invalid source.type {stype!r}')
    status = data.get('status')

    if stype == 'external' and status in {'running', 'completed'}:
        if not source.get('url'):
            fail(rel, 'external running/completed source.url is required')
        if not source.get('commit'):
            fail(rel, 'external running/completed source.commit is required')

    if stype == 'internal':
        exp = source.get('experiment')
        run_id = source.get('run_id')
        commit = source.get('commit')
        if not exp or not isinstance(exp, str):
            fail(rel, 'internal source.experiment is required')
        else:
            exp_dir = ROOT / exp
            rec_path = exp_dir / 'record.json'
            if not rec_path.is_file():
                fail(rel, f'internal source experiment missing: {exp}')
            else:
                src = load(rec_path)
                if src is not None:
                    run_ids = {r.get('run_id') for r in src.get('runs', [])}
                    if run_id not in run_ids:
                        fail(rel, f'run_id {run_id!r} not in source experiment runs')
                    src_commits = {src.get('source_commit')} | {
                        r.get('source_commit') for r in src.get('runs', [])}
                    if commit not in src_commits:
                        fail(rel, f'source.commit {commit!r} not in source experiment commits')
                    summary = exp_dir / 'results' / 'summary.json'
                    if status == 'completed' and not summary.is_file():
                        fail(rel, f'internal source summary missing: {summary}')
                    elif status == 'completed':
                        result = data.get('results')
                        if isinstance(result, dict):
                            src_sum = load(summary)
                            if src_sum:
                                mm = src_sum.get('macro_mean', {})
                                if not close(result.get('macro_auc'), mm.get('auc_percent')):
                                    fail(rel, f'macro_auc != source auc_percent ({result.get("macro_auc")} vs {mm.get("auc_percent")})')
                                if not close(result.get('macro_eer'), mm.get('eer')):
                                    fail(rel, f'macro_eer != source eer ({result.get("macro_eer")} vs {mm.get("eer")})')
                                src_scenes = src_sum.get('scenes', {})
                                for scene, vals in result.get('per_scene', {}).items():
                                    if scene not in src_scenes:
                                        fail(rel, f'per_scene {scene!r} not in source summary')
                                    else:
                                        if not close(vals.get('auc'), src_scenes[scene].get('auc_percent')):
                                            fail(rel, f'per_scene[{scene}].auc != source')
                                        if not close(vals.get('eer'), src_scenes[scene].get('eer')):
                                            fail(rel, f'per_scene[{scene}].eer != source')

    protocol = data.get('protocol')
    if status == 'completed' and not isinstance(protocol, dict):
        fail(rel, 'completed entry requires protocol object')
    if isinstance(protocol, dict):
        if set(protocol) != PROTOCOL_KEYS:
            fail(rel, f'protocol keys must be {sorted(PROTOCOL_KEYS)}')
        if protocol.get('one_class') is not True:
            fail(rel, 'protocol.one_class must be true for the main comparison')
        if not isinstance(protocol.get('train'), str) or not protocol['train']:
            fail(rel, 'protocol.train must be non-empty')
        if not isinstance(protocol.get('test'), str) or not protocol['test']:
            fail(rel, 'protocol.test must be non-empty')
        metrics = protocol.get('metrics')
        if not isinstance(metrics, list) or 'AUC' not in metrics:
            fail(rel, 'protocol.metrics must include AUC')
        for key in ('seed', 'epochs', 'input_size'):
            val = protocol.get(key)
            if val is not None and (not isinstance(val, int) or isinstance(val, bool)):
                fail(rel, f'protocol.{key} must be int or null')

    result = data.get('results')
    if status == 'completed' and not isinstance(result, dict):
        fail(rel, 'completed entry requires results object')
    if isinstance(result, dict):
        if set(result) != RESULT_KEYS:
            fail(rel, f'results keys must be {sorted(RESULT_KEYS)}')
        if not is_num(result.get('macro_auc')) or not is_num(result.get('macro_eer')):
            fail(rel, 'results.macro_auc/macro_eer must be finite numbers')
        per_scene = result.get('per_scene')
        if not isinstance(per_scene, dict) or not per_scene:
            fail(rel, 'results.per_scene must be a non-empty object')
        else:
            for scene, vals in per_scene.items():
                if not isinstance(vals, dict) or set(vals) != {'auc', 'eer'}:
                    fail(rel, f'per_scene[{scene}] must contain exactly auc/eer')
                elif not is_num(vals.get('auc')) or not is_num(vals.get('eer')):
                    fail(rel, f'per_scene[{scene}] auc/eer must be finite numbers')
        summary = result.get('summary')
        if status == 'completed':
            if not isinstance(summary, str):
                fail(rel, 'results.summary must be a path string')
            elif not (path.parent / summary).is_file():
                fail(rel, f'results.summary missing: {summary}')
    return data


def main():
    rec_paths = sorted(COMP.glob('*/*/record.json'))
    for path in rec_paths:
        data = check_record(path)
        if data is not None:
            records[path.parent.relative_to(COMP).as_posix()] = data

    catalog = COMP / 'CATALOG.csv'
    if not catalog.is_file():
        fail(catalog.relative_to(ROOT), 'missing; run comparisons/build_catalog.py')
    else:
        with catalog.open() as f:
            rows = list(csv.DictReader(f))
        cat_paths = {r.get('path') for r in rows}
        exp_paths = set(records)
        if cat_paths != exp_paths:
            fail(catalog.relative_to(ROOT),
                 f'catalog mismatch: missing={sorted(exp_paths - cat_paths)}, stale={sorted(cat_paths - exp_paths)}')

    if errors:
        print('Comparison registry check failed:', file=sys.stderr)
        for item in errors:
            print(' - ' + item, file=sys.stderr)
        return 1
    counts = {}
    for data in records.values():
        counts[data['status']] = counts.get(data['status'], 0) + 1
    summary = ', '.join(f'{k}={v}' for k, v in sorted(counts.items()))
    print(f'Comparison registry OK: {len(records)} entries ({summary})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
