#!/usr/bin/env python3
"""Generate comparisons/CATALOG.csv and comparisons/INDEX.md from record.json files."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / 'comparisons'
COLUMNS = [
    'path', 'id', 'dataset', 'method', 'method_display', 'year', 'variant', 'status',
    'source_type', 'source_url', 'source_commit', 'source_experiment',
    'protocol', 'metrics', 'macro_auc', 'macro_eer', 'result_summary', 'notes',
]


def compact_protocol(p):
    if not isinstance(p, dict):
        return ''
    metrics = '+'.join(p.get('metrics') or [])
    return (f"one_class={p.get('one_class')};metrics={metrics};"
            f"seed={p.get('seed')};epochs={p.get('epochs')};input={p.get('input_size')}")


def main():
    records = []
    for path in sorted(COMP.glob('*/*/record.json')):
        data = json.loads(path.read_text())
        rel = path.parent.relative_to(COMP).as_posix()
        source = data.get('source') or {}
        results = data.get('results') or {}
        records.append({
            'path': rel,
            'id': data.get('id', ''),
            'dataset': data.get('dataset', ''),
            'method': data.get('method', ''),
            'method_display': data.get('method_display', ''),
            'year': data.get('year', ''),
            'variant': data.get('variant', ''),
            'status': data.get('status', ''),
            'source_type': source.get('type', ''),
            'source_url': source.get('url') or '',
            'source_commit': source.get('commit') or '',
            'source_experiment': source.get('experiment') or '',
            'protocol': compact_protocol(data.get('protocol')),
            'metrics': '+'.join((data.get('protocol') or {}).get('metrics') or []),
            'macro_auc': '' if not results else results.get('macro_auc', ''),
            'macro_eer': '' if not results else results.get('macro_eer', ''),
            'result_summary': (results.get('summary') or '') if isinstance(results, dict) else '',
            'notes': data.get('notes', ''),
        })
    method_order = ['ours', 'advae', 'ganomaly', 'memae', 'mkd', 'andt', 'future-frame',
                    'st-dissoc', 'mlep', 'astt', 'dad-fsm', 'hstforu', 'madvit', 'lightvit-ad']
    variant_order = {
        'rd_s12_t4_h512': 0, 'rd_s4_t12_h512': 1, 'rd_s4_t12_gl768_h512': 2,
        'rd_s4_t12_gl192_h512': 3, 'rd_s4_t12_gl192_h128': 4,
        'rd_s4_t12_gl192_h128_qr2': 5, 'full': 0, 'multi-layer': 1, 'last-layer': 2,
    }
    def sort_key(r):
        mi = method_order.index(r['method']) if r['method'] in method_order else 99
        return (r['dataset'], mi, r['method'], variant_order.get(r['variant'], 0), r['variant'])
    records.sort(key=sort_key)

    with (COMP / 'CATALOG.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)

    counts = {}
    for r in records:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    lines = [
        '# 对比实验导航', '',
        f'共 {len(records)} 个条目：' + '，'.join(f'{k}={v}' for k, v in sorted(counts.items())) + '。',
        '', '字段规范见 [SCHEMA.md](SCHEMA.md)；统一协议与添加外部方法见 [README.md](README.md)。',
        '', '> 原始实验仍以 `experiments/` 为准；本目录只做对比编排，不移动/删除原始实验。',
    ]
    for dataset in ('da', 'uit'):
        subset = [r for r in records if r['dataset'] == dataset]
        lines += ['', f'## {dataset}', '']
        if not subset:
            lines += ['（暂无条目）']
            continue
        lines += ['| ID | 方法 | 变体 | 状态 | macro AUC | macro EER | 来源 |',
                  '|---|---|---|---|---:|---:|---|']
        for r in subset:
            target = COMP / r['path'] / 'README.md'
            link = f"{r['path']}/README.md" if target.is_file() else f"{r['path']}/record.json"
            if r['source_type'] == 'internal' and r['source_experiment']:
                source = f"[experiment](../{r['source_experiment']})"
            elif r['source_url']:
                source = f"[upstream]({r['source_url']})"
            else:
                source = 'awaiting URL'
            auc = '—' if r['macro_auc'] == '' else f"{float(r['macro_auc']):.4f}"
            eer = '—' if r['macro_eer'] == '' else f"{float(r['macro_eer']):.4f}"
            lines.append(f"| [{r['id']}]({link}) | {r['method_display']} | `{r['variant']}` | "
                         f"`{r['status']}` | {auc} | {eer} | {source} |")
    lines += ['', '重新生成：`python comparisons/build_catalog.py`；校验：`python comparisons/check_comparisons.py`。', '']
    (COMP / 'INDEX.md').write_text('\n'.join(lines))
    print(f'{len(records)} comparison entries indexed')


if __name__ == '__main__':
    main()
