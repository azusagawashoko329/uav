#!/usr/bin/env python3
"""Summarize DA GL matched-score results across seeds 42/43/44."""
from __future__ import annotations
import hashlib, json, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; AB=ROOT/'ablation/da/GL'
SCENES=['bike','crossroads','farmland','highway','railway','solar','vehicle']; ORDERS=['GGGG','LLLL','GLLL','LLLG']; FFNS=['gate','gelumlp']; SEEDS=[42,43,44]
def exp(seed,o,f): return AB/f'20261106_{o.lower()}_{f}_s{seed}'
def load():
 d={}
 for seed in SEEDS:
  for o in ORDERS:
   for f in FFNS:
    p=exp(seed,o,f); c=json.loads((p/'config.json').read_text()); cells={s:json.loads((p/'results'/f'{s}.json').read_text()) for s in SCENES}
    d[(seed,o,f)]={'experiment':p.name,'cells':cells,'auc':statistics.mean(x['auc_percent'] for x in cells.values()),'eer':statistics.mean(x['eer'] for x in cells.values()),'params':next(iter({x['trainable_parameters'] for x in cells.values()}))}
 return d
def mean_sd(vals): return statistics.mean(vals), statistics.stdev(vals) if len(vals)>1 else 0.0
def main():
 d=load(); rows=[]
 for o in ORDERS:
  for f in FFNS:
   auc=[d[(s,o,f)]['auc'] for s in SEEDS]; eer=[d[(s,o,f)]['eer'] for s in SEEDS]; am,asd=mean_sd(auc); em,esd=mean_sd(eer)
   rows.append({'block_order':o,'ffn_type':f,'trainable_parameters':d[(42,o,f)]['params'],'seed_auc_percent':dict(zip(map(str,SEEDS),auc)),'seed_eer':dict(zip(map(str,SEEDS),eer)),'mean_auc_percent':am,'sd_auc_percent':asd,'mean_eer':em,'sd_eer':esd})
 def delta(seed,a,b,metric): return d[(seed,*a)][metric]-d[(seed,*b)][metric]
 comps={'ffn_effect_by_order':{},'order_effect_vs_glll_gate':{},'order_effect_vs_glll_gelumlp':{}}
 for o in ORDERS:
  comps['ffn_effect_by_order'][o]={'auc': [delta(s,(o,'gelumlp'),(o,'gate'),'auc') for s in SEEDS], 'eer':[delta(s,(o,'gelumlp'),(o,'gate'),'eer') for s in SEEDS]}
 for o in ORDERS:
  if o!='GLLL':
   comps['order_effect_vs_glll_gate'][o]={'auc':[delta(s,(o,'gate'),('GLLL','gate'),'auc') for s in SEEDS],'eer':[delta(s,(o,'gate'),('GLLL','gate'),'eer') for s in SEEDS]}
   comps['order_effect_vs_glll_gelumlp'][o]={'auc':[delta(s,(o,'gelumlp'),('GLLL','gelumlp'),'auc') for s in SEEDS],'eer':[delta(s,(o,'gelumlp'),('GLLL','gelumlp'),'eer') for s in SEEDS]}
 payload={'schema_version':1,'dataset':'da','family':'GL','seeds':SEEDS,'score':'matched_group_balanced_top5 (alpha=0.5)','aggregation':'mean across seven scene-level metrics; cross-seed mean and sample SD','mother_template':'rd_s4_t12_gl768_h512','scenes':SCENES,'matrix':rows,'comparisons':comps,'limitations':['three seeds only (42,43,44)','each seed has seven independent scene models','cross-seed SD is descriptive, not a formal significance test']}
 (AB/'multiseed_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 lines=['# DA / GL 三 seed 消融矩阵总结','','本汇总合并 seed42、seed43、seed44 的八个 GL 变体。所有配置固定 V2 母模板：DINOv3 ViT-B/16、12 teacher taps、OCBE hidden 512、code_dim 64、student width 768、4 stages、12 heads、30 epoch、batch64、AdamW + OneCycleLR。训练与测试均使用 `matched_score(k=5, alpha=0.5)`。每个 seed 的 AUC/EER 先按七场景等权宏平均，再对三个 seed 计算均值和样本标准差。','','## 三 seed 总体矩阵','','| block order | FFN | params | seed42 AUC | seed43 AUC | seed44 AUC | mean AUC ± SD | mean EER ± SD |','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in sorted(rows,key=lambda x:-x['mean_auc_percent']): lines.append(f"| {r['block_order']} | {r['ffn_type']} | {r['trainable_parameters']:,} | {r['seed_auc_percent']['42']:.4f} | {r['seed_auc_percent']['43']:.4f} | {r['seed_auc_percent']['44']:.4f} | {r['mean_auc_percent']:.4f} ± {r['sd_auc_percent']:.4f} | {r['mean_eer']:.4f} ± {r['sd_eer']:.4f} |")
 lines += ['', '## 三 seed 逐场景 AUC 均值', '', '| scene | GGGG gate | GGGG gelumlp | LLLL gate | LLLL gelumlp | GLLL gate | GLLL gelumlp | LLLG gate | LLLG gelumlp |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for scene in SCENES:
  vals=[]
  for o in ORDERS:
   for f in FFNS: vals.append(statistics.mean(d[(s,o,f)]['cells'][scene]['auc_percent'] for s in SEEDS))
  lines.append('| '+scene+' | '+' | '.join(f'{x:.2f}' for x in vals)+' |')
 best_auc=max(rows,key=lambda x:x['mean_auc_percent']); best_eer=min(rows,key=lambda x:x['mean_eer'])
 def avg(v): return statistics.mean(v)
 lines += ['', '## 矩阵分析', '', f"- 三 seed 宏平均 AUC 最高为 `{best_auc['block_order']} + {best_auc['ffn_type']}`：{best_auc['mean_auc_percent']:.4f}% ± {best_auc['sd_auc_percent']:.4f}；宏平均 EER 最低为 `{best_eer['block_order']} + {best_eer['ffn_type']}`：{best_eer['mean_eer']:.4f} ± {best_eer['sd_eer']:.4f}。", '- 三 seed 下的排序比单 seed 更稳定地反映结构差异，但 AUC 最优和 EER 最优仍不是同一个组合。']
 for o in ORDERS:
  x=comps['ffn_effect_by_order'][o]; lines.append(f"- FFN 从 gate 换成 gelumlp，在 `{o}` 上跨 seed 的 AUC 变化均值为 {avg(x['auc']):+.4f} 个百分点，EER 变化均值为 {avg(x['eer']):+.4f}；该效应随 block order 改变，不能单独解释为 FFN 的普遍收益。")
 for o in ('GGGG','LLLL','LLLG'):
  x=comps['order_effect_vs_glll_gate'][o]; y=comps['order_effect_vs_glll_gelumlp'][o]; lines.append(f"- 相对 GLLL，`{o}` 在 gate 下 AUC 变化均值 {avg(x['auc']):+.4f}、EER {avg(x['eer']):+.4f}；在 gelumlp 下 AUC {avg(y['auc']):+.4f}、EER {avg(y['eer']):+.4f}。")
 lines += ['- 逐场景均值显示 Crossroads 和 Vehicle 仍是主要区分场景；Bike、Farmland、Railway 的 AUC 普遍较高，配置间差异相对小。','- 该结果覆盖三个 seed，但仍只有三种子、每种子七个独立场景模型；标准差用于描述稳定性，不能替代正式显著性检验。','','## 运行证据','','- seed43/44 共 112 个场景任务由混合 seed 双卡动态队列完成，scheduler 报告 `completed=112, skipped=0, failures=[]`。','- 16 个实验目录分别保留场景 JSON、逐帧 CSV、checkpoint 哈希和 scheduler 日志；大权重仍由 `.gitignore` 排除。']
 (AB/'multiseed_summary.md').write_text('\n'.join(lines)+'\n')
 for r in rows:
  for s in SEEDS:
   p=exp(s,r['block_order'],r['ffn_type'])/'results'; local={'schema_version':1,'experiment':exp(s,r['block_order'],r['ffn_type']).name,'seed':s,'ablation':{'block_order':r['block_order'],'ffn_type':r['ffn_type']},'score':payload['score'],'macro_auc_percent':d[(s,r['block_order'],r['ffn_type'])]['auc'],'macro_eer':d[(s,r['block_order'],r['ffn_type'])]['eer'],'cross_seed_mean_auc_percent':r['mean_auc_percent'],'cross_seed_sd_auc_percent':r['sd_auc_percent'],'cross_seed_mean_eer':r['mean_eer'],'cross_seed_sd_eer':r['sd_eer']}; (p/'summary.json').write_text(json.dumps(local,ensure_ascii=False,indent=2)+'\n'); (p/'summary.md').write_text(f"# {local['experiment']}\n\n配置：`{r['block_order']} + {r['ffn_type']}`，seed{s}。\n\nmatched-score 宏平均 AUC：**{local['macro_auc_percent']:.4f}%**；宏平均 EER：**{local['macro_eer']:.4f}**。三 seed 汇总见 [../../multiseed_summary.md](../../multiseed_summary.md)。\n")
 print('wrote',AB/'multiseed_summary.md')
if __name__=='__main__': main()
