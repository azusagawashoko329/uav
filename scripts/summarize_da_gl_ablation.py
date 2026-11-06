#!/usr/bin/env python3
"""Summarize the completed DA GL ablation matrix using matched-score outputs."""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
AB = ROOT / 'ablation' / 'da' / 'GL'
SCENES = ['bike', 'crossroads', 'farmland', 'highway', 'railway', 'solar', 'vehicle']
ORDERS = ['GGGG', 'LLLL', 'GLLL', 'LLLG']
FFNS = ['gate', 'gelumlp']

def load():
    data = {}
    for d in sorted(AB.glob('20261106_*')):
        cfg = json.loads((d / 'config.json').read_text())
        ab = cfg['ablation']
        cells = {}
        for scene in SCENES:
            p = d / 'results' / f'{scene}.json'
            if not p.is_file():
                raise RuntimeError(f'missing {p}')
            cells[scene] = json.loads(p.read_text())
        key = (ab['block_order'], ab['ffn_type'])
        if key in data:
            raise RuntimeError(f'duplicate cell {key}')
        auc = mean(x['auc_percent'] for x in cells.values())
        eer = mean(x['eer'] for x in cells.values())
        params = {x['trainable_parameters'] for x in cells.values()}
        if len(params) != 1:
            raise RuntimeError(f'parameter mismatch in {d}: {params}')
        data[key] = {'dir': d.name, 'config': cfg, 'cells': cells,
                     'macro_auc': auc, 'macro_eer': eer, 'params': params.pop()}
    expected = {(o, f) for o in ORDERS for f in FFNS}
    if set(data) != expected:
        raise RuntimeError(f'matrix mismatch: missing={expected-set(data)} extra={set(data)-expected}')
    return data

def fmt(x, n=4): return f'{x:.{n}f}'
def cell_label(o, f): return f'{o}+{f}'

def main():
    data = load()
    rows = []
    for o in ORDERS:
        for f in FFNS:
            d = data[(o, f)]
            rows.append({'block_order': o, 'ffn_type': f, 'experiment': d['dir'],
                         'macro_auc_percent': d['macro_auc'], 'macro_eer': d['macro_eer'],
                         'trainable_parameters': d['params'], 'scenes': len(d['cells'])})
    def delta(a, b):
        return {'auc_delta': data[a]['macro_auc'] - data[b]['macro_auc'],
                'eer_delta': data[a]['macro_eer'] - data[b]['macro_eer']}
    comparisons = {
        'ffn_effect_by_order': {o: delta((o, 'gelumlp'), (o, 'gate')) for o in ORDERS},
        'order_effect_vs_glll_gate': {o: delta((o, 'gate'), ('GLLL', 'gate')) for o in ORDERS if o != 'GLLL'},
        'order_effect_vs_glll_gelumlp': {o: delta((o, 'gelumlp'), ('GLLL', 'gelumlp')) for o in ORDERS if o != 'GLLL'},
    }
    payload = {'schema_version': 1, 'dataset': 'da', 'family': 'GL', 'seed': 42,
               'score': 'matched_group_balanced_top5 (alpha=0.5)', 'aggregation': 'mean of seven scene AUC/EER',
               'mother_template': 'rd_s4_t12_gl768_h512', 'scenes': SCENES,
               'matrix': rows, 'comparisons': comparisons,
               'limitations': ['single seed42', 'seven independent scene models',
                              'macro mean gives each scene equal weight',
                              'differences are descriptive and have no multi-seed uncertainty estimate']}
    (AB / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    lines = [
        '# DA / GL 消融矩阵总结', '',
        '本矩阵以 V2 `rd_s4_t12_gl768_h512` 为母模板，固定 DINOv3 ViT-B/16、12 个 teacher taps、OCBE hidden 512、code_dim 64、student width 768、4 stages、12 个 reconstruction heads、seed42、30 epoch、batch64 和 group-balanced top-5 cosine loss（alpha=0.5）。只改变 student 的四块顺序和 FFN 类型。', '',
        '测试分数使用与训练一致的 `matched_score(k=5, alpha=0.5)`；表中 AUC/EER 是七个 DA 场景分别计算后再做等权宏平均。', '',
        '## 总体矩阵', '',
        '| block order | FFN | trainable params | macro AUC (%) | macro EER | 状态 |',
        '|---|---:|---:|---:|---:|---|',
    ]
    rank_auc = sorted(rows, key=lambda r: r['macro_auc_percent'], reverse=True)
    for r in sorted(rows, key=lambda r: (-r['macro_auc_percent'], r['block_order'], r['ffn_type'])):
        status = '迁移完成' if r['block_order'] == 'GLLL' and r['ffn_type'] == 'gate' or r['block_order'] == 'GGGG' and r['ffn_type'] == 'gelumlp' else '新跑完成'
        lines.append(f"| {r['block_order']} | {r['ffn_type']} | {r['trainable_parameters']:,} | {r['macro_auc_percent']:.4f} | {r['macro_eer']:.4f} | {status} |")
    lines += ['', '## 逐场景 AUC (%)', '', '| scene | GGGG gate | GGGG gelumlp | LLLL gate | LLLL gelumlp | GLLL gate | GLLL gelumlp | LLLG gate | LLLG gelumlp |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in SCENES:
        lines.append('| ' + s + ' | ' + ' | '.join(f"{data[(o,f)]['cells'][s]['auc_percent']:.2f}" for o in ['GGGG','LLLL','GLLL','LLLG'] for f in FFNS) + ' |')
    lines += ['', '## 逐场景 EER', '', '| scene | GGGG gate | GGGG gelumlp | LLLL gate | LLLL gelumlp | GLLL gate | GLLL gelumlp | LLLG gate | LLLG gelumlp |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in SCENES:
        lines.append('| ' + s + ' | ' + ' | '.join(f"{data[(o,f)]['cells'][s]['eer']:.4f}" for o in ['GGGG','LLLL','GLLL','LLLG'] for f in FFNS) + ' |')
    best_auc = rank_auc[0]; best_eer = min(rows, key=lambda r: r['macro_eer'])
    lines += ['', '## 矩阵分析', '',
        f"- **宏平均 AUC 最高**：`{best_auc['block_order']} + {best_auc['ffn_type']}`，{best_auc['macro_auc_percent']:.4f}%。**宏平均 EER 最低**：`{best_eer['block_order']} + {best_eer['ffn_type']}`，{best_eer['macro_eer']:.4f}。两个指标的最优组合不同。",
        f"- V2 母模板 `GLLL + gate` 的宏平均 AUC/EER 为 {data[('GLLL','gate')]['macro_auc']:.4f}% / {data[('GLLL','gate')]['macro_eer']:.4f}。在 gate 条件下，`LLLL` 相对 V2 的 AUC 变化为 {comparisons['order_effect_vs_glll_gate']['LLLL']['auc_delta']:+.4f} 个百分点，但 EER 变化为 {comparisons['order_effect_vs_glll_gate']['LLLL']['eer_delta']:+.4f}。",
        f"- FFN 效应依赖 block 顺序：GELU MLP 相对 gate 在 `GLLL` 上 AUC {comparisons['ffn_effect_by_order']['GLLL']['auc_delta']:+.4f} 个百分点，在 `LLLG` 上 {comparisons['ffn_effect_by_order']['LLLG']['auc_delta']:+.4f}；但在 `LLLL` 上为 {comparisons['ffn_effect_by_order']['LLLL']['auc_delta']:+.4f}，在 `GGGG` 上为 {comparisons['ffn_effect_by_order']['GGGG']['auc_delta']:+.4f}。因此不能把 FFN 类型的效果脱离 block 顺序解释。",
        '- 场景差异主要集中在 Crossroads 和 Vehicle；Bike、Farmland、Railway 等场景的 AUC 较高且不同配置间差距较小，宏平均排名会受到低分场景的相对变化影响。',
        '- 结果是单 seed42、每场景独立训练的描述性比较；没有多 seed 方差或显著性检验，不能据此宣称全局结构优劣。', '',
        '## 运行与限制', '',
        '- 六个新单元通过两张 GPU 动态队列完成；恢复运行跳过已有完整场景，未创建 retry 目录。',
        '- 两个迁移单元为已有 DA V2/V1 matched-score 结果：`GLLL + gate` 与 `GGGG + gelumlp`。',
        '- 所有结果均保留每场景 JSON、逐帧 CSV、checkpoint 哈希和 scheduler 日志；大权重留在被 `.gitignore` 忽略的 `ckpt/`。',
    ]
    (AB / 'summary.md').write_text('\n'.join(lines) + '\n')
    for r in rows:
        d=AB/r['experiment']; dres=d/'results'; dres.mkdir(exist_ok=True)
        local={'schema_version':1,'experiment':r['experiment'],'ablation':{'block_order':r['block_order'],'ffn_type':r['ffn_type']},'score':payload['score'],'aggregation':payload['aggregation'],'macro_auc_percent':r['macro_auc_percent'],'macro_eer':r['macro_eer'],'scenes':SCENES}
        (dres/'summary.json').write_text(json.dumps(local,ensure_ascii=False,indent=2)+'\n')
        (dres/'summary.md').write_text(f"# {r['experiment']}\n\n配置：`{r['block_order']} + {r['ffn_type']}`。\n\nmatched-score 宏平均 AUC：**{r['macro_auc_percent']:.4f}%**；宏平均 EER：**{r['macro_eer']:.4f}**。完整矩阵分析见 [../../summary.md](../../summary.md)。\n")
    print('wrote', AB/'summary.md')

if __name__ == '__main__': main()
