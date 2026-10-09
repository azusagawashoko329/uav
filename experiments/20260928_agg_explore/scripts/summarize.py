"""把 experiments/20260928_agg_explore/raw_*.txt 汇总成 results.csv 和 markdown 表。

用法:  python experiments/20260928_agg_explore/scripts/summarize.py
"""
import glob
import os
import re

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(ROOT, 'experiments', '20260928_agg_explore')
SCENES = ['highway', 'farmland', 'bike', 'solar', 'vehicle', 'crossroads', 'railway']

VARIANT_DOC = {
    'base12':       '基线原样复刻: 12 层学生, taps [5,7,9,11], OCBE 同源, 无反相对齐改动',
    'rep4_ocbe4':   '4 个单层 [2,5,8,11], OCBE 同源, 学生 4 层 (三三分组每组取代表层)',
    'mean4_ocbe4':  '4 个组平均当目标, OCBE 同源, 学生 4 层 (组会方案: 相邻三层取平均)',
    'mean4':        '4 个组平均当目标, OCBE 吃全 12 层, 学生 4 层',
    'rep4':         '4 个单层 [2,5,8,11], OCBE 吃全 12 层, 学生 4 层',
    'old4tap':      '老配置: 4 个单层 [5,7,9,11], OCBE 同源, 学生 4 层',
    'raw12':        '12 个单层全当目标(一 block 挂 3 头), OCBE 吃 12 层, 学生 4 层',
    'rawmean4':     '4 个组平均(不逐层归一化), OCBE 吃 12 层, 学生 4 层',
}


def load():
    rows = {}
    for f in sorted(glob.glob(os.path.join(EXP, 'raw_*.txt'))):
        variant = os.path.basename(f)[len('raw_'):-len('.txt')]
        for ln in open(f):
            m = re.search(r'>>> \[(\w+)\] (\w+): AUC=([\d.]+)', ln)
            if m:
                rows.setdefault(m.group(1), {})[m.group(2)] = float(m.group(3))
    return rows


def main():
    rows = load()
    if not rows:
        print('没有找到任何 raw_*.txt 结果')
        return
    known = [s for s in SCENES if any(s in v for v in rows.values())]
    order = sorted(rows, key=lambda v: -sum(rows[v].get(s, 0) for s in known) / max(
        sum(1 for s in known if s in rows[v]), 1))

    with open(os.path.join(EXP, 'results.csv'), 'w') as f:
        f.write('variant,' + ','.join(known) + ',mean\n')
        for v in order:
            vals = [rows[v].get(s) for s in known]
            mean = sum(x for x in vals if x is not None) / max(sum(x is not None for x in vals), 1)
            f.write(v + ',' + ','.join('' if x is None else f'{x:.2f}' for x in vals)
                    + f',{mean:.2f}\n')

    print(f'{"variant":<14}' + ''.join(s[:9].rjust(11) for s in known) + 'mean'.rjust(9))
    for v in order:
        vals = [rows[v].get(s) for s in known]
        mean = sum(x for x in vals if x is not None) / max(sum(x is not None for x in vals), 1)
        print(f'{v:<14}' + ''.join('—'.rjust(11) if x is None else f'{x:.2f}'.rjust(11)
                                   for x in vals) + f'{mean:.2f}'.rjust(9))
    print(f'\n写了 {os.path.join(EXP, "results.csv")}')


if __name__ == '__main__':
    main()
