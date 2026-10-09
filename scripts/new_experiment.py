#!/usr/bin/env python3
"""Create one structured experiment scaffold; no model code is run."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from experiment_paths import ROOT, VALID_DATASETS, VALID_FAMILIES

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=sorted(VALID_DATASETS))
parser.add_argument('family', choices=sorted(VALID_FAMILIES))
parser.add_argument('question', help='short lowercase ASCII token')
parser.add_argument('variant', help='one changed factor, lowercase ASCII token')
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--date', help='YYYYMMDD; default Asia/Shanghai today')
args = parser.parse_args()
for key in ('question','variant'):
    value=getattr(args,key)
    if not value.isascii() or not value.replace('_','').isalnum() or value.lower()!=value:
        parser.error(f'{key} must be lowercase ASCII letters/digits/underscores')
if args.seed < 0:parser.error('seed must be nonnegative')
date=args.date or datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
if len(date)!=8 or not date.isdigit():parser.error('date must be YYYYMMDD')
identifier=f'{date}_{args.question}_{args.variant}_s{args.seed}'
relative=Path('experiments')/args.dataset/args.family/identifier
target=ROOT/relative
if target.exists():parser.error(f'experiment already exists: {relative}')
target.mkdir(parents=True)
for name in ('scripts','logs','results','ckpt','visualizations'):(target/name).mkdir()
record={'schema_version':2,'id':identifier,'path':relative.as_posix(),'dataset':args.dataset,'family':args.family,'legacy':False,'status':'planned','source_commit':None,'question':args.question,'variant':args.variant,'scripts':[],'runs':[],'result_summary':None,'limitations':[]}
(target/'record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
(target/'config.json').write_text(json.dumps({'dataset':args.dataset,'seed':args.seed,'data_root':None,'split_fingerprint':None,'parameters':{}},ensure_ascii=False,indent=2)+'\n')
(target/'artifacts.json').write_text(json.dumps({'schema_version':2,'artifacts':[]},indent=2)+'\n')
(target/'README.md').write_text(f'# {identifier}\n\n## 问题与基线\n\n待填写。\n\n## 数据、配置和评测口径\n\n待填写；参数见 `config.json`。\n\n## 运行与结果\n\n状态：planned；尚未运行。先提交源码和脚本，填入运行时 `source_commit`。\n\n## 限制\n\n待填写。\n')
(target/'scripts'/'README.md').write_text('此目录只存该实验专用的 Python 和 Shell 脚本。运行前提交脚本并在 record.json 中登记。\n')
print(relative)
