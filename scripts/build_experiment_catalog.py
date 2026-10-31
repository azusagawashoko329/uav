#!/usr/bin/env python3
"""Build the navigable experiment catalog from per-experiment records."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'
records=[]
RESULT_NAMES=('summary.json','summary.csv','summary.md','results_mean.csv','results.csv','results_fusion.csv','loss_compare.csv','report.md','REVIEW.md','uit.json','five_variant_summary.md','five_variant_comparison.csv')
for path in sorted(EXP.glob('*/**/record.json')):
    if 'source_snapshot_' in str(path):
        continue
    d=json.loads(path.read_text())
    relative=path.parent.relative_to(EXP).as_posix()
    parts=relative.split('/')
    if len(parts)<3:continue
    evidence=d.get('evidence',[])
    result_files=[name for name in RESULT_NAMES if (path.parent/name).is_file()]
    records.append({
        'id':d.get('id',path.parent.name),'path':relative,
        'dataset':d.get('dataset',parts[0]),'family':d.get('family',parts[1]),
        'status':d.get('status','unknown'),'source_ref':d.get('source_ref') or '',
        'evidence_count':len(evidence),'weight_files_local':d.get('weight_files_local',''),
        'scripts_count':len(d.get('scripts',[])),
        'result_evidence':';'.join(result_files)
    })
records.sort(key=lambda r:(r['dataset'],r['family'],r['path']))
with (EXP/'CATALOG.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
groups=defaultdict(list)
for row in records:groups[(row['dataset'],row['family'])].append(row)
lines=['# 实验导航','',f'共 {len(records)} 个已登记实验或历史实验组。状态来自各目录的 `record.json`；`unknown` 表示缺证据，不表示失败。', '', '| 数据集 | 方向 | 数量 | 入口 |', '|---|---|---:|---|']
for (dataset,family),items in sorted(groups.items()):
    lines.append(f'| `{dataset}` | `{family}` | {len(items)} | [查看](#{dataset}-{family}) |')
for (dataset,family),items in sorted(groups.items()):
    lines += ['',f'## {dataset} / {family}','','| 实验或实验组 | 状态 | 结果证据 | 脚本数 | 本地权重数 |','|---|---|---|---:|---:|']
    for r in items:
        d=EXP/r['path'];target=d/'RECORD.md'
        if not target.exists():target=d/'README.md'
        link=target.relative_to(EXP).as_posix() if target.exists() else r['path']
        result_link='、'.join(f"[{name}]({r['path']}/{name})" for name in r['result_evidence'].split(';') if name) or '未找到'
        lines.append(f"| [{r['id']}]({link}) | `{r['status']}` | {result_link} | {r['scripts_count']} | {r['weight_files_local']} |")
lines += ['', '旧路径请查 [迁移表](../docs/MIGRATION_MAP.csv)；跨分支历史见 [历史索引](../docs/LEGACY_EXPERIMENTS.md)。', '']
(EXP/'INDEX.md').write_text('\n'.join(lines))
print(f'{len(records)} experiment records indexed')
