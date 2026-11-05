#!/usr/bin/env python3
"""Apply matched-score re-evaluation results to DA experiments and comparison.

Original results are preserved under results_legacy_topk/.  The comparison/da
entries are replaced in place (no legacy copy there), as requested.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.da_variant_runner import SCENES  # noqa: E402

ORDER = ['bike', 'crossroads', 'farmland', 'highway', 'railway', 'solar', 'vehicle']
DISPLAY = {'bike': 'Bike', 'crossroads': 'Crossroads', 'farmland': 'Farmland',
           'highway': 'Highway', 'railway': 'Railway', 'solar': 'Solar', 'vehicle': 'Vehicle'}


def fmt(x, n=4):
    return f'{x:.{n}f}'


def macro(scenes):
    return (statistics.mean(v['auc'] for v in scenes.values()),
            statistics.mean(v['eer'] for v in scenes.values()))


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def new_scenes(exp):
    out = {}
    for scene in ORDER:
        p = exp / 'results_matched' / f'{scene}.json'
        if not p.is_file():
            raise FileNotFoundError(f'missing matched result: {p}')
        m = load_json(p)
        if m.get('scoring') != 'matched_group_balanced_top5':
            raise ValueError(f'wrong scoring marker: {p}')
        out[scene] = {'auc': float(m['auc_percent']), 'eer': float(m['eer']),
                      'frames': int(m['frames']), 'positive_frames': int(m['positive_frames'])}
    return out


def write_experiment_summary(exp, cfg, scenes, old):
    macro_auc, macro_eer = macro(scenes)
    summary = {
        'dataset': 'Drone-Anomaly',
        'variant_id': cfg['variant_id'], 'variant_label': cfg['variant_label'],
        'runner_variant': cfg.get('runner_variant', cfg.get('variant')),
        'seed': int(cfg['seed']), 'epochs': int(cfg['training']['epochs']),
        'status': 'completed', 'completed_scenes': list(ORDER),
        'missing_scenes': [], 'failures': [],
        'macro_mean': {'auc_percent': macro_auc, 'eer': macro_eer},
        'scenes': {s: {'auc_percent': v['auc'], 'eer': v['eer'], 'frames': v['frames'],
                       'positive_frames': v['positive_frames'],
                       'trainable_parameters': cfg['resources']['expected_trainable_parameters'],
                       'checkpoint': f'ckpt/{s}_final.pth'}
                   for s, v in scenes.items()},
        'scoring': 'matched_group_balanced_top5',
        'scoring_note': ('V0: mean over 4 taps of top5 mean; V1-V5: mean over 4 groups of '
                         '0.5*mean_tap(top5 mean)+0.5*max_tap(top5 mean).'),
        'previous_topk_summary': 'results_legacy_topk/summary.json',
    }
    write_json(exp / 'results' / 'summary.json', summary)
    lines = [f"# {cfg['variant_label']} `{cfg['variant_id']}`：DA 30 epoch（matched score）", '',
             f"- macro AUC: {fmt(macro_auc)}", f"- macro EER: {fmt(macro_eer)}",
             f"- scoring: {summary['scoring_note']}",
             "- original top-k results: results_legacy_topk/", '',
             '| scene | AUC | EER |', '|---|---:|---:|']
    for s in ORDER:
        lines.append(f"| {s} | {fmt(scenes[s]['auc'])} | {fmt(scenes[s]['eer'])} |")
    (exp / 'results' / 'summary.md').write_text('\n'.join(lines) + '\n')
    return summary


def update_experiment_record(exp, summary, old_summary):
    p = exp / 'record.json'; rec = load_json(p)
    rec['result_summary'] = 'results/summary.md'
    rec['scoring_correction'] = {
        'reason': 'training used group-balanced top-k loss; old evaluation used mean-over-taps top-k readout',
        'old_scoring': 'legacy_mean_over_taps_topk',
        'new_scoring': 'matched_group_balanced_top5',
        'old_summary': 'results_legacy_topk/summary.json',
        'old_macro_auc': old_summary['macro_mean']['auc_percent'],
        'old_macro_eer': old_summary['macro_mean']['eer'],
        'new_macro_auc': summary['macro_mean']['auc_percent'],
        'new_macro_eer': summary['macro_mean']['eer'],
    }
    outs = rec['runs'][0].get('outputs', []) if rec.get('runs') else []
    for rel in ['results/summary.json', 'results/summary.md', 'results_legacy_topk/summary.json']:
        if rel not in outs:
            outs.append(rel)
    if rec.get('runs'):
        rec['runs'][0]['outputs'] = outs
    lim = rec.setdefault('limitations', [])
    note = 'AUC/EER corrected to training-matched score; original top-k results retained under results_legacy_topk/'
    if note not in lim:
        lim.append(note)
    write_json(p, rec)


def update_comparison(vid, label, exp, summary, old_summary):
    out = ROOT / 'comparison/da/ours' / vid
    p = out / 'record.json'; rec = load_json(p)
    rec['results'] = {
        'macro_auc': summary['macro_mean']['auc_percent'],
        'macro_eer': summary['macro_mean']['eer'],
        'per_scene': {s: {'auc': v['auc_percent'], 'eer': v['eer']} for s, v in summary['scenes'].items()},
        'summary': 'results/summary.json',
        'source_results': str(exp.relative_to(ROOT) / 'results') + '/',
    }
    rec['notes'] = (rec.get('notes', '').rstrip() +
                    ' Scoring corrected to the training-matched group-balanced top-k objective; '
                    'original top-k results remain in the raw experiment legacy directory.')
    write_json(p, rec)
    write_json(out / 'results/summary.json', summary)
    (out / 'results/summary.md').write_text((exp / 'results/summary.md').read_text())
    readme = out / 'README.md'
    old_text = readme.read_text() if readme.is_file() else f'# da-ours-{vid}\n\n'
    prefix = old_text.split('## 结果')[0].rstrip() + '\n\n'
    lines = ['## 结果', '',
             f"macro AUC：**{fmt(summary['macro_mean']['auc_percent'])}**　"
             f"macro EER：**{fmt(summary['macro_mean']['eer'])}**", '',
             '| scene | AUC | EER |', '|---|---:|---:|']
    for s in ORDER:
        lines.append(f"| {s} | {fmt(summary['scenes'][s]['auc_percent'])} | {fmt(summary['scenes'][s]['eer'])} |")
    lines += ['', '完整逐帧分数与 checkpoint 以原始实验目录为准；本条目只复制 `results/summary.{json,md}`。']
    readme.write_text(prefix + '\n'.join(lines) + '\n')
    return rec


def update_docs(results):
    # results: list of (vid,label,summary,old_summary)
    tbl = ROOT / 'comparison/da/comparison_summary.md'
    text = tbl.read_text()
    for vid, label, sm, old in results:
        old_s = fmt(old['macro_mean']['auc_percent'])
        new_s = fmt(sm['macro_mean']['auc_percent'])
        # Main efficiency table row
        text = re.sub(rf'(\| Ours {label} `{re.escape(vid)}` \|.*\| ){re.escape(old_s)} (\| measured)',
                      rf'\g<1>{new_s} \2', text)
        # Per-scene matrix table row
        row = (f"| {vid} | {label} | " +
               ' | '.join(fmt(sm['scenes'][s]['auc_percent']) for s in ['highway', 'farmland', 'bike', 'solar', 'vehicle', 'crossroads', 'railway']) +
               f" | {new_s} |")
        text = re.sub(rf'^\| {re.escape(vid)} \| {label} \|.*$', row, text, flags=re.M)
    tbl.write_text(text)

    # Experiment matrix summary
    p = ROOT / 'experiments/da/appearance/20261103_variant_matrix_summary.md'
    if p.is_file():
        t = p.read_text()
        for vid, label, sm, old in results:
            row = f"| `{vid}` | {label} | `completed` | {fmt(sm['macro_mean']['auc_percent'])} | {fmt(sm['macro_mean']['eer'])} |"
            t = re.sub(rf'^\| `{re.escape(vid)}` \| {label} \| `completed` \|.*$', row, t, flags=re.M)
            vals = ' | '.join(fmt(sm['scenes'][s]['auc_percent'], 2) for s in ORDER)
            t = re.sub(rf'^\| {label} \| .*$', f'| {label} | {vals} |', t, flags=re.M)
        p.write_text(t)

    # Data audit leave-bike sensitivity
    p = ROOT / 'experiments/da/appearance/20261103_variant_matrix_data_audit.md'
    if p.is_file():
        t = p.read_text()
        order = sorted(results, key=lambda r: -r[2]['macro_mean']['auc_percent'])
        for vid, label, sm, old in results:
            no_bike = [sm['scenes'][s]['auc_percent'] for s in ORDER if s != 'bike']
            row = f"| {label} | {fmt(sm['macro_mean']['auc_percent'])} | {fmt(statistics.mean(no_bike))} |"
            t = re.sub(rf'^\| {label} \| .*$', row, t, flags=re.M)
        # Update ranking sentence if present
        ranking = ' > '.join(r[1] for r in order)
        t = re.sub(r'V2 > V1 > V0 > V3 > V5 > V4', ranking, t)
        p.write_text(t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiments', nargs='+', required=True)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    exps = [(ROOT / e).resolve() if not Path(e).is_absolute() else Path(e).resolve()
            for e in args.experiments]
    pending = []
    for exp in exps:
        cfg = load_json(exp / 'config.json')
        vid = cfg['variant_id']; label = cfg['variant_label']
        scenes = new_scenes(exp)
        old_path = exp / 'results_legacy_topk' / 'summary.json'
        if old_path.is_file():
            old = load_json(old_path)
        else:
            old = load_json(exp / 'results' / 'summary.json')
        old_macro = old['macro_mean']
        new_macro = {'auc_percent': macro(scenes)[0], 'eer': macro(scenes)[1]}
        print(f'{label} {vid}: AUC {fmt(old_macro["auc_percent"])} -> {fmt(new_macro["auc_percent"])}; '
              f'EER {fmt(old_macro["eer"])} -> {fmt(new_macro["eer"])}')
        pending.append((exp, cfg, scenes, old))
    if not args.apply:
        print('dry-run only; pass --apply to write')
        return
    results = []
    for exp, cfg, scenes, old in pending:
        # Preserve the original results once, before overwriting.
        legacy = exp / 'results_legacy_topk'
        if not legacy.is_dir():
            shutil.copytree(exp / 'results', legacy)
        for scene in ORDER:
            shutil.copyfile(exp / 'results_matched' / f'{scene}.json',
                            exp / 'results' / f'{scene}.json')
            src_csv = exp / 'results_matched' / f'{scene}_scores.csv'
            if src_csv.is_file():
                shutil.copyfile(src_csv, exp / 'results' / f'{scene}_scores.csv')
        summary = write_experiment_summary(exp, cfg, scenes, old)
        update_experiment_record(exp, summary, old)
        update_comparison(cfg['variant_id'], cfg['variant_label'], exp, summary, old)
        results.append((cfg['variant_id'], cfg['variant_label'], summary, old))
    update_docs(results)
    subprocess.run([sys.executable, str(ROOT / 'comparison/build_catalog.py')], check=True)
    subprocess.run([sys.executable, str(ROOT / 'comparison/check_comparisons.py')], check=True)


if __name__ == '__main__':
    import subprocess
    main()
