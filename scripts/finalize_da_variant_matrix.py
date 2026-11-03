#!/usr/bin/env python3
"""Finalize the DA six-variant matrix: summaries, record runs, artifacts.

Run this after scripts/da_variant_scheduler.py exits (or after a partial run that the
operator wants to archive). It is idempotent for a given --run-id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCENES = ['bike', 'crossroads', 'farmland', 'highway', 'railway', 'solar', 'vehicle']


def rel_sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_events(exp: Path, run_id: str):
    log = exp / 'logs' / f'scheduler_{run_id}.jsonl'
    if not log.is_file():
        return log, []
    events = []
    for line in log.read_text().splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return log, events


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--experiments', nargs='+', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--matrix-summary', type=Path, default=None)
    args = p.parse_args()
    root = args.repo_root.resolve()
    experiments = [root / e if not Path(e).is_absolute() else Path(e) for e in args.experiments]
    matrix_rows = []

    for exp in experiments:
        exp = exp.resolve()
        cfg = json.loads((exp / 'config.json').read_text())
        rec = json.loads((exp / 'record.json').read_text())
        variant_id = cfg['variant_id']
        label = cfg['variant_label']
        log, events = load_events(exp, args.run_id)
        start = next((e for e in events if e.get('kind') == 'start'), {})
        end = next((e for e in events if e.get('kind') == 'all_finished'), {})
        failures = [e for e in events if e.get('kind') == 'finish'
                    and e.get('variant_id') == variant_id and e.get('returncode') != 0]

        scene_results = {}
        missing = []
        for scene in SCENES:
            path = exp / 'results' / f'{scene}.json'
            if path.is_file():
                scene_results[scene] = json.loads(path.read_text())
            else:
                missing.append(scene)
        aucs = [scene_results[s]['auc_percent'] for s in scene_results]
        eers = [scene_results[s]['eer'] for s in scene_results]
        macro_auc = sum(aucs) / len(aucs) if aucs else None
        macro_eer = sum(eers) / len(eers) if eers else None
        complete = len(scene_results) == len(SCENES) and not failures
        status = 'completed' if complete else 'failed'

        summary = {
            'dataset': 'Drone-Anomaly',
            'variant_id': variant_id,
            'variant_label': label,
            'runner_variant': cfg.get('runner_variant'),
            'seed': cfg['seed'],
            'epochs': cfg['training']['epochs'],
            'status': status,
            'completed_scenes': sorted(scene_results),
            'missing_scenes': missing,
            'failures': failures,
            'macro_mean': {'auc_percent': macro_auc, 'eer': macro_eer},
            'scenes': {s: {
                'auc_percent': scene_results[s]['auc_percent'],
                'eer': scene_results[s]['eer'],
                'frames': scene_results[s]['frames'],
                'positive_frames': scene_results[s]['positive_frames'],
                'trainable_parameters': scene_results[s]['trainable_parameters'],
                'checkpoint': scene_results[s]['checkpoint'],
                'checkpoint_sha256': scene_results[s]['checkpoint_sha256'],
            } for s in sorted(scene_results)},
        }
        (exp / 'results').mkdir(exist_ok=True)
        (exp / 'results' / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

        md = [f'# {variant_id} ({label}) — DA 30ep seed42', '',
              f'- status: `{status}`',
              f'- scenes: {len(scene_results)}/{len(SCENES)}',
              f'- macro AUC: {macro_auc:.4f}' if macro_auc is not None else '- macro AUC: n/a',
              f'- macro EER: {macro_eer:.4f}' if macro_eer is not None else '- macro EER: n/a',
              '', '| scene | AUC | EER | frames | positives |', '|---|---:|---:|---:|---:|']
        for s in SCENES:
            if s in scene_results:
                m = scene_results[s]
                md.append(f"| {s} | {m['auc_percent']:.4f} | {m['eer']:.4f} | {m['frames']} | {m['positive_frames']} |")
            else:
                md.append(f'| {s} | — | — | — | — |')
        (exp / 'results' / 'summary.md').write_text('\n'.join(md) + '\n')

        # Record: replace this run entry, set status and outputs.
        outputs = [f'results/summary.json', f'results/summary.md', f'logs/scheduler_{args.run_id}.jsonl']
        for s in sorted(scene_results):
            outputs += [f'results/{s}.json', f'results/{s}_scores.csv', f'ckpt/{s}_final.pth']
        run_id_full = f'{args.run_id}_{rec["id"]}'
        rec['runs'] = [r for r in rec.get('runs', []) if r.get('run_id') != run_id_full]
        rec['runs'].append({
            'run_id': run_id_full,
            'status': status,
            'source_commit': rec.get('source_commit'),
            'script': 'scripts/da_variant_scheduler.py',
            'seed': cfg['seed'],
            'started_at': start.get('time'),
            'ended_at': end.get('time'),
            'command': ['scripts/da_variant_scheduler.py', '--run-id', args.run_id,
                        '--experiments', *[str(e.relative_to(root)) for e in experiments],
                        '--gpus', '0', '1', '--epochs', str(cfg['training']['epochs']),
                        '--batch', str(cfg['training']['batch']),
                        '--workers', str(cfg['training']['workers']),
                        '--seed', str(cfg['seed'])],
            'log': f'logs/scheduler_{args.run_id}.jsonl',
            'outputs': outputs,
            'notes': f'completed_scenes={len(scene_results)}/{len(SCENES)}; failures={len(failures)}',
        })
        rec['status'] = status
        rec['result_summary'] = 'results/summary.md'
        (exp / 'record.json').write_text(json.dumps(rec, indent=2, ensure_ascii=False) + '\n')

        artifacts = []
        for s in sorted(scene_results):
            m = scene_results[s]
            for rel, producer, git in [
                (f'results/{s}.json', 'scripts/da_variant_runner.py', 'tracked'),
                (f'results/{s}_scores.csv', 'scripts/da_variant_runner.py', 'tracked'),
                (f'ckpt/{s}_final.pth', 'scripts/da_variant_runner.py', 'ignored'),
            ]:
                path = exp / rel
                if not path.is_file():
                    continue
                item = {'path': rel, 'producer_script': producer, 'run_id': run_id_full,
                        'status': 'present', 'size': path.stat().st_size, 'git': git}
                if rel.endswith('.pth'):
                    item['sha256'] = m.get('checkpoint_sha256')
                else:
                    item['sha256'] = rel_sha(path)
                artifacts.append(item)
        for rel, producer in [
            ('results/summary.json', 'scripts/finalize_da_variant_matrix.py'),
            ('results/summary.md', 'scripts/finalize_da_variant_matrix.py'),
            (f'logs/scheduler_{args.run_id}.jsonl', 'scripts/da_variant_scheduler.py'),
        ]:
            path = exp / rel
            if path.is_file():
                artifacts.append({'path': rel, 'producer_script': producer, 'run_id': run_id_full,
                                  'status': 'present', 'size': path.stat().st_size,
                                  'sha256': rel_sha(path), 'git': 'tracked'})
        (exp / 'artifacts.json').write_text(json.dumps(
            {'schema_version': 2, 'artifacts': artifacts}, indent=2, ensure_ascii=False) + '\n')

        matrix_rows.append({
            'variant_id': variant_id, 'variant_label': label, 'status': status,
            'macro_auc': macro_auc, 'macro_eer': macro_eer,
            'params': cfg['resources']['expected_trainable_parameters'],
            'completed_scenes': len(scene_results), 'missing': missing,
            'scenes': {s: scene_results[s]['auc_percent'] for s in sorted(scene_results)},
        })
        print(f'{label} {variant_id}: {status} scenes={len(scene_results)}/{len(SCENES)} '
              f'macro AUC={macro_auc if macro_auc is None else round(macro_auc, 4)}')

    matrix_path = args.matrix_summary or (root / 'experiments/da/appearance/20261103_variant_matrix_summary.md')
    lines = ['# DA 六变体矩阵（seed42, 30 epoch）', '',
             '同一 `scripts/da_variant_scheduler.py` 双卡动态队列；每个场景独立模型，七场景 macro 平均。',
             '', '| ID | 编号 | status | macro AUC | macro EER | params | scenes |',
             '|---|---|---|---:|---:|---:|---:|']
    for r in matrix_rows:
        auc = '—' if r['macro_auc'] is None else f"{r['macro_auc']:.4f}"
        eer = '—' if r['macro_eer'] is None else f"{r['macro_eer']:.4f}"
        lines.append(f"| `{r['variant_id']}` | {r['variant_label']} | `{r['status']}` | {auc} | {eer} | "
                     f"{r['params']:,} | {r['completed_scenes']}/7 |")
    lines += ['', '## 分场景 AUC', '', '| ID | ' + ' | '.join(SCENES) + ' |',
              '|---|' + '|'.join(['---:'] * len(SCENES)) + '|']
    for r in matrix_rows:
        vals = [f"{r['scenes'][s]:.2f}" if s in r['scenes'] else '—' for s in SCENES]
        lines.append(f"| {r['variant_label']} | " + ' | '.join(vals) + ' |')
    lines += ['', '## 比较注意', '',
              '- 矩阵内部同 runner、seed、30 epoch、数据划分；V0 与 V1–V5 的 taps/loss 不同，V0→V1 不是严格单因素对照。',
              '- 单 seed42；分场景波动可大于 macro 差异，不能据此声称显著性。',
              '- 与旧 DA 同名运行的绝对数值差异可能来自 cuDNN deterministic、动态双卡分配和运行方差；旧运行模型代码与当前一致（git diff 05443fa..HEAD 对 model/ 为空），不是加载错模型。',
              '- 新旧 runner 的 `split_fingerprint` 配方不同：旧 DA runner 用绝对路径拼 manifest，新 runner 用场景相对路径；数据文件相同但哈希不同（Crossroads 旧 `8ded80ee…`，新 `22f9aa85…`），不要直接用 fingerprint 跨代判断数据划分。新矩阵六变体之间 fingerprint 一致。']
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text('\n'.join(lines) + '\n')
    print('matrix summary:', matrix_path)


if __name__ == '__main__':
    main()
