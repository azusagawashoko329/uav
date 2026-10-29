"""Build the Group-Context AUC and resource reports."""
import csv, json
from pathlib import Path
EXP=Path(__file__).resolve().parents[1]; RES=EXP/'results'; REF=EXP.parent/'20261029_rank2_baseline30_rerun1_s42/results'
SCENES=['highway','farmland','bike','solar','vehicle','crossroads','railway']
def load(d,s): return json.loads((Path(d)/f'{s}.json').read_text())
def mean(xs): return sum(xs)/len(xs)
def main():
 g={s:load(RES,s) for s in SCENES}; b={s:load(REF,s) for s in SCENES}
 rows=[]
 for s in SCENES:
  rows.append({'scene':s,'rank2_30_auc':b[s]['auc_percent'],'group_context_auc':g[s]['auc_percent'],'delta':g[s]['auc_percent']-b[s]['auc_percent']})
 with (RES/'auc_comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 timing=[]
 for s in SCENES:
  x=g[s];timing.append({'scene':s,'auc_percent':x['auc_percent'],'trainable_parameters':x['trainable_parameters'],'elapsed_s':x['elapsed_s'],'elapsed_min':x['elapsed_s']/60,'checkpoint_sha256':x['checkpoint_sha256']})
 timing.append({'scene':'sum','auc_percent':mean([x['auc_percent'] for x in timing]),'trainable_parameters':timing[0]['trainable_parameters'],'elapsed_s':sum(x['elapsed_s'] for x in timing),'elapsed_min':sum(x['elapsed_s'] for x in timing)/60,'checkpoint_sha256':''})
 with (RES/'training_timing.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(timing[0]));w.writeheader();w.writerows(timing)
 bm=next(RES.glob('*_trained_gpu0.json'),None)
 lines=['# Group-Context Fusion 结果复盘','', '| 场景 | Rank2 30 epoch | GCF 30 epoch | Δ GCF−Rank2 |','|---|---:|---:|---:|']
 for r in rows: lines.append(f"| {r['scene']} | {r['rank2_30_auc']:.4f} | {r['group_context_auc']:.4f} | {r['delta']:+.4f} |")
 mean_b=mean([r['rank2_30_auc'] for r in rows]);mean_g=mean([r['group_context_auc'] for r in rows])
 lines += [f'| **mean(7)** | **{mean_b:.4f}** | **{mean_g:.4f}** | **{mean_g-mean_b:+.4f}** |','',f'- 可训练参数：**{timing[0]["trainable_parameters"]:,}**；Rank2 30 epoch 为 3,761,222，减少约 **17.4%**。',f'- 七场景训练耗时合计：**{sum(x["elapsed_s"] for x in timing[:-1])/60:.2f} 分钟**。', '- GCF 通过共享组内投影、四组均值上下文和逐元素乘性残差完成交互，没有新增 loss。', '- 结果没有达到 Rank2 baseline：均值下降主要来自 Solar、Vehicle、Railway。', '- 推理测速见 [`speed_summary.md`](speed_summary.md)。', '', '## 判断', '', '该结构确实把四组关系显式写入模型，且训练速度较快；但当前乘性全局上下文与蒸馏目标不匹配，不能作为当前主线结构。单种子结果仍需多 seed 复核。']
 (RES/'summary.md').write_text('\n'.join(lines)+'\n')
 if bm:
  d=json.loads(bm.read_text()); out=['# 训练权重速度复核','','FP32、GPU 0、教师特征已在 GPU；不含教师和预处理。','','| batch | core ms | core FPS | score ms | score FPS |','|---:|---:|---:|---:|---:|']
  for q in ['1','16']:
   c=[x['timings'][q]['core'] for x in d['scenes']];s=[x['timings'][q]['score'] for x in d['scenes']]
   out.append(f"| {q} | {mean([x['latency_p50_ms'] for x in c]):.4f} | {mean([x['serial_fps'] for x in c]):.1f} | {mean([x['latency_p50_ms'] for x in s]):.4f} | {mean([x['serial_fps'] for x in s]):.1f} |")
  out += ['', '30 次 warmup；每个场景独立重复测量；详细值保存在 JSON。'];(RES/'speed_summary.md').write_text('\n'.join(out)+'\n')
if __name__=='__main__':main()
