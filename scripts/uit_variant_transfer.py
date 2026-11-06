#!/usr/bin/env python3
"""Archive completed UIT six-variant experiments into comparison/uit/ours/.

This is a post-run step: it does not train or evaluate. It:
  * validates results/uit.json against the experiment config and canonical variant;
  * writes experiment results/summary.json + results/summary.md (if absent);
  * creates comparison/uit/ours/<variant>/record.json plus result summaries;
  * copies the architecture-only efficiency benchmark from comparison/da/ours/<variant>
    when present, with provenance updated;
  * refreshes comparison/CATALOG.csv and INDEX.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.uit_appearance_runner import VARIANT_CANONICAL  # noqa: E402

BUILDER_TO_VID = {builder: vid for vid, (_, builder, _) in VARIANT_CANONICAL.items()}
DEFAULT_RUN_ID = '20261104T153558Z_uit_variant_matrix_s42'


def now_utc():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def build_summary(cfg: dict, metric: dict, vid: str, label: str) -> dict:
    return {
        'dataset': 'UIT-ADrone',
        'variant_id': vid,
        'variant_label': label,
        'runner_variant': cfg['variant'],
        'seed': int(cfg['seed']),
        'epochs': int(cfg['training']['epochs']),
        'status': 'completed',
        'completed_scenes': ['uit'],
        'missing_scenes': [],
        'failures': [],
        'macro_mean': {
            'auc_percent': float(metric['auc_percent']),
            'eer': float(metric['eer']),
        },
        'scenes': {
            'uit': {
                'auc_percent': float(metric['auc_percent']),
                'eer': float(metric['eer']),
                'frames': int(metric['frames']),
                'positive_frames': int(metric['positive_frames']),
                'trainable_parameters': int(metric['trainable_parameters']),
                'checkpoint': metric.get('checkpoint'),
                'checkpoint_sha256': metric.get('checkpoint_sha256'),
            }
        },
    }


def summary_md(cfg: dict, metric: dict, vid: str, label: str) -> str:
    return (
        f"# {label} / `{vid}`：UIT-ADrone "
        f"{cfg['training']['epochs']} epoch\n\n"
        f"AUC **{metric['auc_percent']:.4f}%**，EER **{metric['eer'] * 100:.4f}%**，"
        f"参数 **{metric['trainable_parameters'] / 1e6:.6f}M**。\n\n"
        f"变体 `{cfg['variant']}`，seed{cfg['seed']}，45 epoch 完整运行；"
        "checkpoint 10/20/30/40/45 与 final 保存在实验目录，权重不入 Git。\n"
    )


def update_experiment(exp: Path, run_id: str, ended_at: str) -> None:
    p = exp / 'record.json'
    rec = json.loads(p.read_text())
    rec['status'] = 'completed'
    rec['result_summary'] = 'results/summary.md'
    rec['weight_files_local'] = len(list((exp / 'ckpt').glob('*.pth')))
    for run in rec.get('runs', []):
        if run.get('run_id', '').startswith(run_id):
            run['status'] = 'completed'
            run['ended_at'] = ended_at
            run['outputs'] = [
                'results/uit.json', 'results/uit_matched.json', 'results/uit_scores.csv',
                'results/uit_matched_scores.csv', 'results/summary.json', 'results/summary.md',
                'ckpt/uit_final.pth',
                'ckpt/uit_epoch10.pth', 'ckpt/uit_epoch20.pth',
                'ckpt/uit_epoch30.pth', 'ckpt/uit_epoch40.pth',
                'ckpt/uit_epoch45.pth',
            ]
    write_json(exp / 'record.json', rec)


def update_artifacts(exp: Path, run_id: str) -> None:
    items = []
    for rel in [
        'results/uit.json', 'results/uit_matched.json', 'results/summary.json', 'results/summary.md',
        'results/uit_scores.csv', 'results/uit_matched_scores.csv',
        'ckpt/uit_final.pth', 'ckpt/uit_epoch10.pth', 'ckpt/uit_epoch20.pth',
        'ckpt/uit_epoch30.pth', 'ckpt/uit_epoch40.pth', 'ckpt/uit_epoch45.pth',
    ]:
        p = exp / rel
        if not p.is_file():
            continue
        items.append({
            'path': rel,
            'producer_script': 'scripts/uit_appearance_runner.py',
            'run_id': run_id,
            'status': 'present',
            'size_bytes': p.stat().st_size,
            'sha256': sha256(p),
            'storage': 'git' if rel.startswith('results/') and not rel.endswith('_scores.csv') else 'ignored',
        })
    for p in sorted((exp / 'logs').glob('*')):
        if p.is_file() and p.suffix in {'.log', '.jsonl'}:
            items.append({
                'path': str(p.relative_to(exp)),
                'producer_script': 'scripts/uit_resource_scheduler.py',
                'run_id': run_id,
                'status': 'present',
                'size_bytes': p.stat().st_size,
                'sha256': sha256(p),
                'storage': 'git',
            })
    write_json(exp / 'artifacts.json', {'schema_version': 2, 'experiment': exp.name, 'artifacts': items})


def copy_efficiency(vid: str, out: Path, exp: Path) -> dict | None:
    da_record = ROOT / 'comparison/da/ours' / vid / 'record.json'
    da_bench = ROOT / 'comparison/da/ours' / vid / 'benchmark_v2.json'
    if not da_record.is_file() or not da_bench.is_file():
        return None
    src = json.loads(da_record.read_text()).get('efficiency')
    if not src:
        return None
    bench_out = out / 'benchmark_v2.json'
    shutil.copyfile(da_bench, bench_out)
    bench = json.loads(bench_out.read_text())
    bench['benchmark_source'] = str(bench_out.relative_to(ROOT))
    bench['da_adapted'] = False
    bench['auc_source'] = str(exp.relative_to(ROOT)) + ' (seed42, 45ep UIT)'
    write_json(bench_out, bench)
    src = dict(src)
    src['auc_source'] = str(exp.relative_to(ROOT)) + ' (seed42, 45ep UIT)'
    src['da_adapted'] = False
    src['benchmark_source'] = str(bench_out.relative_to(ROOT))
    src['notes'] = (src.get('notes', '').rstrip() +
                    ' Efficiency benchmark is architecture-only and dataset-independent; '
                    'copied from the DA comparison entry with UIT provenance.')
    return src


def archive_one(exp: Path, run_id: str, dry_run: bool) -> Path:
    cfg = json.loads((exp / 'config.json').read_text())
    vid = cfg.get('variant_id') or BUILDER_TO_VID.get(cfg['variant'])
    if vid is None:
        raise RuntimeError(f'{exp}: unknown builder {cfg["variant"]}')
    matched = exp / 'results/uit_matched.json'
    if not matched.is_file():
        if vid == 'rd_s12_t4_h512' and (exp / 'results/uit.json').is_file():
            metric = json.loads((exp / 'results/uit.json').read_text())
            metric['scoring'] = 'matched_group_balanced_top5'
            metric['scoring_note'] = ('V0 4-tap case: old topk readout is exactly the matched '
                                      'mean-over-taps score; no re-evaluation needed.')
            write_json(matched, metric)
        else:
            raise FileNotFoundError(f'{exp}: matched result missing; run uit_matched_reeval.py')
    metric = json.loads(matched.read_text())
    if metric.get('scoring') != 'matched_group_balanced_top5':
        raise RuntimeError(f'{exp}: result is not matched-scored: {metric.get("scoring")}')
    label, builder, taps = VARIANT_CANONICAL[vid]
    if cfg['variant'] != builder:
        raise RuntimeError(f'{exp}: builder mismatch {cfg["variant"]} != {builder}')
    if metric.get('variant') != builder:
        raise RuntimeError(f'{exp}: result variant {metric.get("variant")} != {builder}')
    expected = cfg['resources']['expected_trainable_parameters']
    if int(metric['trainable_parameters']) != int(expected):
        raise RuntimeError(f'{exp}: parameter mismatch {metric["trainable_parameters"]} != {expected}')
    required_ckpts = ['ckpt/uit_final.pth'] + [f'ckpt/uit_epoch{ep}.pth'
                                                for ep in cfg['training']['checkpoint_epochs']]
    missing = [rel for rel in required_ckpts if not (exp / rel).is_file()]
    if missing:
        raise RuntimeError(f'{exp}: missing checkpoints: {missing}')
    summary = build_summary(cfg, metric, vid, label)
    out = ROOT / 'comparison/uit/ours' / vid
    print(f'[{label} {vid}] {exp.name} -> {out.relative_to(ROOT)}  '
          f'AUC={metric["auc_percent"]:.4f} EER={metric["eer"]:.4f} params={expected}')
    if dry_run:
        return out

    write_json(exp / 'results/summary.json', summary)
    (exp / 'results/summary.md').write_text(summary_md(cfg, metric, vid, label))
    update_experiment(exp, run_id, now_utc())
    update_artifacts(exp, run_id)

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'results/summary.json', summary)
    shutil.copyfile(exp / 'results/summary.md', out / 'results/summary.md')
    src_metric = exp / 'results/uit.json'
    shutil.copyfile(src_metric if src_metric.is_file() else matched, out / 'results/uit.json')

    source = {
        'type': 'internal', 'url': None,
        'commit': json.loads((exp / 'record.json').read_text())['source_commit'],
        'local_path': None,
        'experiment': str(exp.relative_to(ROOT)),
        'run_id': f'{run_id}_{exp.name}',
        'license': 'internal',
        'notes': 'Raw experiment remains authoritative; checkpoints are local and registered in the source artifacts.json.',
    }
    rec = {
        'schema_version': 1,
        'id': f'uit-ours-{vid}',
        'dataset': 'uit',
        'method': 'ours',
        'method_display': 'ours',
        'year': 2026,
        'variant': vid,
        'status': 'completed',
        'source': source,
        'protocol': {
            'one_class': True,
            'train': 'official train normal frames only: train/frames/<video>/*.jpg, 41 videos',
            'test': 'concat all 51 test videos and 147005 frames; one global AUC/EER',
            'metrics': ['AUC', 'EER'],
            'seed': 42,
            'epochs': 45,
            'input_size': 256,
            'notes': ('UIT-ADrone official train/test layout; train/test overlap retained; '
                      f'variant {vid} = {builder}; teacher taps {taps}.'),
        },
        'results': {
            'macro_auc': summary['macro_mean']['auc_percent'],
            'macro_eer': summary['macro_mean']['eer'],
            'per_scene': {'uit': {'auc': summary['scenes']['uit']['auc_percent'],
                                  'eer': summary['scenes']['uit']['eer']}},
            'summary': 'results/summary.json',
            'source_results': str(exp.relative_to(ROOT) / 'results') + '/',
        },
        'notes': f'Comparison view of {label}; raw experiment: {exp.relative_to(ROOT)}.',
    }
    eff = copy_efficiency(vid, out, exp)
    if eff:
        rec['efficiency'] = eff
    write_json(out / 'record.json', rec)
    (out / 'README.md').write_text(
        f'# {label} / `{vid}`：UIT-ADrone 45 epoch\n\n'
        f'原始实验：[{exp.name}](../../../../experiments/uit/appearance/{exp.name}/README.md)。\n\n'
        f'AUC **{metric["auc_percent"]:.4f}%**，EER **{metric["eer"] * 100:.4f}%**，'
        f'参数 **{metric["trainable_parameters"] / 1e6:.6f}M**；seed42，45 epoch。\n\n'
        '该条目只归档结果；权重、逐帧分数和大日志留在原始实验目录。\n')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', default=DEFAULT_RUN_ID)
    p.add_argument('--experiments', nargs='*', default=None)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.experiments:
        exps = [Path(x) if Path(x).is_absolute() else ROOT / x for x in args.experiments]
    else:
        exps = sorted((ROOT / 'experiments/uit/appearance').glob('20261104_variant_matrix_*'))
    for exp in exps:
        if not (exp / 'results/uit_matched.json').is_file():
            print(f'SKIP {exp.name}: results/uit_matched.json missing')
            continue
        archive_one(exp, args.run_id, args.dry_run)
    if not args.dry_run:
        subprocess.run([sys.executable, str(ROOT / 'comparison/build_catalog.py')], check=True)
        subprocess.run([sys.executable, str(ROOT / 'comparison/check_comparisons.py')], check=True)


if __name__ == '__main__':
    main()
