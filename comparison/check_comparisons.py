#!/usr/bin/env python3
"""Validate comparison/<dataset>/<dataset>-<method>-<variant>/record.json.

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
COMP = ROOT / 'comparison'
DATASETS = {'da', 'uit'}
STATUSES = {'planned', 'running', 'completed', 'failed', 'cancelled', 'blocked', 'unknown'}
SOURCE_KEYS = {'type', 'url', 'commit', 'local_path', 'experiment', 'run_id', 'license', 'notes'}
PROTOCOL_KEYS = {'one_class', 'train', 'test', 'metrics', 'seed', 'epochs', 'input_size', 'notes'}
RESULT_KEYS = {'macro_auc', 'macro_eer', 'per_scene', 'summary', 'source_results'}
EFFICIENCY_STATUSES = {'measured', 'partial/unverified', 'paper-only', 'blocked'}
EFFICIENCY_KEYS = {
    'status', 'input_hwt', 'code_commit', 'weights_state', 'da_adapted',
    'auc_source', 'profiling_config', 'inference_params_m',
    'training_trainable_params_m', 'training_only_params_m',
    'frozen_inference_params_m', 'flops_g_per_output_frame', 'flops_window_g',
    'flops_verified', 'flops_counter', 'flops_ops_covered', 'flops_ops_missing',
    'latency_bs1_p50_ms_per_output_frame', 'latency_bs1_mean_ms_per_output_frame',
    'latency_bs1_p95_ms_per_output_frame', 'fps_bs1', 'fps_bs16',
    'input_frames_t', 'effective_output_scores_k', 'window_stride',
    'window_latency_ms', 'fps_independent_window', 'fps_stream',
    'score_coverage', 'benchmark_source', 'notes',
}
EFFICIENCY_NUM_KEYS = {
    'inference_params_m', 'training_trainable_params_m', 'training_only_params_m',
    'frozen_inference_params_m', 'flops_g_per_output_frame', 'flops_window_g',
    'latency_bs1_p50_ms_per_output_frame', 'latency_bs1_mean_ms_per_output_frame',
    'latency_bs1_p95_ms_per_output_frame', 'fps_bs1', 'fps_bs16',
    'window_latency_ms', 'fps_independent_window', 'fps_stream', 'score_coverage',
}
EFFICIENCY_INT_KEYS = {'input_frames_t', 'effective_output_scores_k', 'window_stride'}

errors = []
records = {}


def fail(path, msg):
    errors.append(f'{path}: {msg}')


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def close(a, b, tol=1e-6):
    return is_num(a) and is_num(b) and abs(a - b) <= tol


def check_efficiency(path: Path, rel: str, data: dict):
    """Validate the optional v2 efficiency block.

    Unknown values must be null (or []), not omitted, so a half-filled block
    cannot be mistaken for a measured result.
    """
    eff = data.get('efficiency')
    if eff is None:
        return
    if not isinstance(eff, dict):
        fail(rel, 'efficiency must be an object or null')
        return
    missing = EFFICIENCY_KEYS - set(eff)
    extra = set(eff) - EFFICIENCY_KEYS
    if missing:
        fail(rel, f'efficiency missing keys {sorted(missing)}')
    if extra:
        fail(rel, f'efficiency has unknown keys {sorted(extra)}')
    if missing or extra:
        return

    status = eff.get('status')
    if status not in EFFICIENCY_STATUSES:
        fail(rel, f'invalid efficiency.status {status!r}')

    input_hwt = eff.get('input_hwt')
    if not isinstance(input_hwt, str) or not re.fullmatch(r'[0-9]+x[0-9]+x[0-9]+', input_hwt):
        fail(rel, 'efficiency.input_hwt must match HxWxT, e.g. 256x256x1')

    for key in ('weights_state', 'auc_source', 'profiling_config'):
        val = eff.get(key)
        if not isinstance(val, str) or not val.strip():
            fail(rel, f'efficiency.{key} must be a non-empty string')

    code_commit = eff.get('code_commit')
    if code_commit is not None and not isinstance(code_commit, str):
        fail(rel, 'efficiency.code_commit must be string or null')
    if status == 'measured':
        if not isinstance(code_commit, str) or not re.fullmatch(r'[0-9a-f]{40}', code_commit):
            fail(rel, 'efficiency.code_commit must be a full 40-char SHA when measured')

    for key in ('da_adapted', 'flops_verified'):
        if not isinstance(eff.get(key), bool):
            fail(rel, f'efficiency.{key} must be boolean')

    for key in EFFICIENCY_NUM_KEYS:
        val = eff.get(key)
        if val is not None and (not is_num(val) or val < 0):
            fail(rel, f'efficiency.{key} must be a non-negative finite number or null')

    for key in EFFICIENCY_INT_KEYS:
        val = eff.get(key)
        if val is not None and (not isinstance(val, int) or isinstance(val, bool) or val < 1):
            fail(rel, f'efficiency.{key} must be a positive integer or null')

    for key in ('flops_ops_covered', 'flops_ops_missing'):
        val = eff.get(key)
        if not isinstance(val, list) or any(not isinstance(x, str) or not x for x in val):
            fail(rel, f'efficiency.{key} must be a list of non-empty strings')

    if not isinstance(eff.get('notes'), str):
        fail(rel, 'efficiency.notes must be a string')

    inference = eff.get('inference_params_m')
    frozen = eff.get('frozen_inference_params_m')
    trainable = eff.get('training_trainable_params_m')
    training_only = eff.get('training_only_params_m')
    if inference is not None and frozen is not None and frozen > inference + 1e-9:
        fail(rel, 'efficiency.frozen_inference_params_m exceeds inference_params_m')
    if trainable is not None and training_only is not None and training_only > trainable + 1e-9:
        fail(rel, 'efficiency.training_only_params_m exceeds training_trainable_params_m')

    benchmark_source = eff.get('benchmark_source')
    if benchmark_source is not None and not isinstance(benchmark_source, str):
        fail(rel, 'efficiency.benchmark_source must be string or null')
    elif benchmark_source and not (ROOT / benchmark_source).is_file():
        fail(rel, f'efficiency.benchmark_source missing: {benchmark_source}')

    k = eff.get('effective_output_scores_k')
    stride = eff.get('window_stride')
    coverage = eff.get('score_coverage')
    if k is not None and stride is not None and coverage is not None:
        if not close(coverage, k / stride, tol=1e-6):
            fail(rel, f'efficiency.score_coverage must equal K/stride ({k}/{stride})')

    if status == 'measured':
        if eff.get('flops_verified') is not True:
            fail(rel, 'measured efficiency requires flops_verified=true')
        if eff.get('flops_ops_missing'):
            fail(rel, 'measured efficiency requires flops_ops_missing=[]')
        for key in ('inference_params_m', 'training_trainable_params_m',
                    'training_only_params_m', 'flops_g_per_output_frame',
                    'latency_bs1_p50_ms_per_output_frame',
                    'latency_bs1_mean_ms_per_output_frame', 'fps_bs1', 'fps_bs16',
                    'input_frames_t', 'effective_output_scores_k', 'window_stride'):
            if eff.get(key) is None:
                fail(rel, f'measured efficiency requires {key}')
        if not isinstance(eff.get('flops_counter'), str) or not eff['flops_counter'].strip():
            fail(rel, 'measured efficiency requires flops_counter')
        if not eff.get('flops_ops_covered'):
            fail(rel, 'measured efficiency requires non-empty flops_ops_covered')
        if not benchmark_source:
            fail(rel, 'measured efficiency requires benchmark_source')
        t = eff.get('input_frames_t')
        if t is not None and t > 1:
            for key in ('window_latency_ms', 'flops_window_g', 'fps_independent_window',
                        'fps_stream', 'score_coverage'):
                if eff.get(key) is None:
                    fail(rel, f'temporal measured efficiency requires {key}')
    elif status in {'partial/unverified', 'blocked', 'paper-only'}:
        if not eff.get('notes', '').strip():
            fail(rel, f'efficiency.status={status} requires explanatory notes')


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
    if not isinstance(method, str) or not method.strip():
        fail(rel, f'invalid method {method!r}')
    if not isinstance(variant, str) or not variant.strip():
        fail(rel, f'invalid variant {variant!r}')
    if not isinstance(rid, str) or not rid.strip():
        fail(rel, f'invalid id {rid!r}')
    if rid != f'{dataset}-{method}-{variant}':
        fail(rel, f'id must equal {dataset}-{method}-{variant}')
    if path.parent.name != variant:
        fail(rel, f'directory name must equal variant {variant!r}')
    if path.parent.parent.name != method:
        fail(rel, f'method directory mismatch: {path.parent.parent.name!r}')
    if path.parent.parent.parent.name != dataset:
        fail(rel, f'dataset directory mismatch: {path.parent.parent.parent.name!r}')
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
    check_efficiency(path, rel, data)
    return data


def main():
    rec_paths = sorted(COMP.glob('*/*/*/record.json'))
    for path in rec_paths:
        data = check_record(path)
        if data is not None:
            records[path.parent.relative_to(COMP).as_posix()] = data

    catalog = COMP / 'CATALOG.csv'
    if not catalog.is_file():
        fail(catalog.relative_to(ROOT), 'missing; run comparison/build_catalog.py')
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
