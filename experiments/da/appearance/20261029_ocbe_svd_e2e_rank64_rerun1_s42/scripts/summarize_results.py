"""Rebuild AUC, timing and speed reports from this experiment's JSON outputs."""
import csv, json
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]
RES = EXP / 'results'
SCENES = ['highway','farmland','bike','solar','vehicle','crossroads','railway']
REF = EXP.parent / '20261028_rank2_checkpoint_refresh_s42' / 'results'
PEER_IDS = {
 '20261029_rank2_baseline30_rerun1_s42': '20261029_ocbe_svd_e2e_rank64_rerun1_s42',
 '20261029_ocbe_svd_e2e_rank64_rerun1_s42': '20261029_rank2_baseline30_rerun1_s42',
}

def load_result(path, scene):
    return json.loads((path / f'{scene}.json').read_text())
def mean(xs): return sum(xs) / len(xs)

def main():
    cfg = json.loads((EXP/'config.json').read_text())
    current = {s: load_result(RES, s) for s in SCENES}
    reference = {s: load_result(REF, s) for s in SCENES}
    peer_dir = EXP.parent / PEER_IDS[EXP.name]
    peer = {s: load_result(peer_dir/'results', s) for s in SCENES} if (peer_dir/'results').is_dir() else None
    rows=[]
    for s in SCENES:
        rows.append({'scene':s,'baseline15_auc':reference[s]['auc_percent'],
                     'rank2_30_auc': current[s]['auc_percent'] if EXP.name.startswith('20261029_rank2') else (peer[s]['auc_percent'] if peer else ''),
                     'svd_e2e_30_auc': current[s]['auc_percent'] if EXP.name.startswith('20261029_ocbe_svd') else (peer[s]['auc_percent'] if peer else ''),
                     'delta_30_vs_baseline15': current[s]['auc_percent']-reference[s]['auc_percent'] if EXP.name.startswith('20261029_rank2') else ((current[s]['auc_percent']-reference[s]['auc_percent']) if EXP.name.startswith('20261029_ocbe_svd') else '')})
    # If the peer is not yet present, keep the current variant row usable.
    if EXP.name.startswith('20261029_rank2'):
        for row,s in zip(rows,SCENES): row['delta_svd30_vs_rank230'] = '' if peer is None else peer[s]['auc_percent']-current[s]['auc_percent']
    else:
        for row,s in zip(rows,SCENES): row['delta_svd30_vs_rank230'] = '' if peer is None else current[s]['auc_percent']-peer[s]['auc_percent']
    with (RES/'auc_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    timing=[]
    for s in SCENES:
        x=current[s]; timing.append({'scene':s,'auc_percent':x['auc_percent'],'trainable_parameters':x['trainable_parameters'],
          'elapsed_s':x['elapsed_s'],'elapsed_min':x['elapsed_s']/60,'checkpoint_sha256':x['checkpoint_sha256']})
    timing.append({'scene':'sum','auc_percent':mean([x['auc_percent'] for x in timing]),
      'trainable_parameters':timing[0]['trainable_parameters'],'elapsed_s':sum(x['elapsed_s'] for x in timing),
      'elapsed_min':sum(x['elapsed_s'] for x in timing)/60,'checkpoint_sha256':''})
    with (RES/'training_timing.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(timing[0])); w.writeheader(); w.writerows(timing)
    auc15=mean([reference[s]['auc_percent'] for s in SCENES])
    auc30=mean([current[s]['auc_percent'] for s in SCENES])
    if peer:
        if EXP.name.startswith('20261029_rank2'):
            auc_rank2=auc30; auc_svd=mean([peer[s]['auc_percent'] for s in SCENES])
        else:
            auc_svd=auc30; auc_rank2=mean([peer[s]['auc_percent'] for s in SCENES])
    else: auc_rank2=auc_svd=None
    params=current[SCENES[0]]['trainable_parameters']
    lines=[f"# {cfg['variant']} 结果复盘",'',
      '本实验与历史 Rank2 15 epoch 参考采用相同七场景、seed42 和评测口径。SVD-e2e 版本从随机低秩因子化头端到端训练，不加载 Rank2 checkpoint。','',
      '| 场景 | Rank2 15 epoch | Rank2 30 epoch | SVD-e2e 30 epoch | 当前 - Rank2 15 | SVD-e2e - Rank2 30 |','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        b=r['rank2_30_auc']; q=r['svd_e2e_30_auc'];
        lines.append(f"| {r['scene']} | {r['baseline15_auc']:.4f} | {b:.4f} | {q:.4f} | {(float(b)-r['baseline15_auc']):+.4f} | {(float(q)-float(b)):+.4f} |")
    lines += [f"| **mean(7)** | **{auc15:.4f}** | **{auc_rank2:.4f}** | **{auc_svd:.4f}** | **{auc30-auc15:+.4f}** | **{auc_svd-auc_rank2:+.4f}** |",'',
      f'- 当前变体可训练参数：**{params:,}**。',
      f'- 当前变体七场景训练耗时合计：**{sum(x["elapsed_s"] for x in timing[:-1])/60:.2f} 分钟**。',
      '- 训练预算：每场景 30 epoch；七场景分别独立训练，双卡动态调度。',
      '- 推理测速见 [`speed_summary.md`](speed_summary.md)。', '', '## 解释', '']
    if EXP.name.startswith('20261029_rank2'):
        lines += [f'- Rank2 从 15 到 30 epoch 的均值变化为 **{auc30-auc15:+.4f} pp**；这是训练预算效应。',
                  f'- 与同为 30 epoch 的 SVD-e2e 相比，Rank2 均值高 **{auc_rank2-auc_svd:+.4f} pp**。']
    else:
        lines += [f'- SVD-e2e 30 epoch 相对 Rank2 15 epoch 为 **{auc30-auc15:+.4f} pp**，不能单独解释为结构收益，因为两者预算不同。',
                  f'- 与同为 30 epoch 的 Rank2 相比，SVD-e2e 均值为 **{auc_svd-auc_rank2:+.4f} pp**。']
    lines += ['- 因此严格的结构比较应使用两组 30 epoch；历史“Rank2 checkpoint 经 SVD 压缩后再微调”属于压缩恢复实验，单独归档。',
              '- 单种子结果；AUC 没有直接的训练代理，30 epoch 的结论仍需多 seed 或验证集早停复核。']
    (RES/'summary.md').write_text('\n'.join(lines)+'\n')
    bench = list(RES.glob('*_trained_gpu0.json'))
    if bench:
        d=json.loads(bench[0].read_text()); out=['# 训练权重速度复核','', 'FP32、GPU 0、教师特征已在 GPU；不含教师和预处理。', '', '| batch | core ms | core FPS | score ms | score FPS |','|---:|---:|---:|---:|---:|']
        for b in ['1','16']:
            core=[x['timings'][b]['core'] for x in d['scenes']]; score=[x['timings'][b]['score'] for x in d['scenes']]
            out.append(f"| {b} | {mean([x['latency_p50_ms'] for x in core]):.4f} | {mean([x['serial_fps'] for x in core]):.1f} | {mean([x['latency_p50_ms'] for x in score]):.4f} | {mean([x['serial_fps'] for x in score]):.1f} |")
        out += ['', '测速前执行了 30 次 warmup，并对每个场景分别重复测量；输出同时保留在对应 JSON。']
        (RES/'speed_summary.md').write_text('\n'.join(out)+'\n')
    if EXP.name.startswith('20261029_rank2') and peer:
        (RES/'equal_budget_comparison.md').write_text((RES/'summary.md').read_text())
if __name__=='__main__': main()
