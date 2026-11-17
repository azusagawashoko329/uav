#!/usr/bin/env python3
"""Create the explicitly requested publication chain and copy validated source artifacts."""
from __future__ import annotations
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model.publication_rd import SPECS,model_spec
BASE=ROOT/'ablation/da/publication';SCENES=['bike','crossroads','farmland','highway','railway','solar','vehicle']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def source_for(name,seed):
 if name=='pt_rd' and seed==42:return ROOT/'experiments/da/appearance/20261103_variant_matrix_v0_rd_s12_t4_h512_s42'
 if name=='d_ablation':return ROOT/f'ablation/da/GL/20261106_gggg_gelumlp_s{seed}'
 if name=='hap_rd':return ROOT/f'ablation/da/GL/20261106_lllg_gate_s{seed}'
 return None

def finalize_copied_summary(exp):
 cfg=json.loads((exp/'config.json').read_text())
 if not cfg.get('reused_from'):return
 for name in ['summary.md','summary.json']:
  src=exp/'results'/name
  if src.exists() and not (exp/'provenance'/('source_'+name)).exists():shutil.copy2(src,exp/'provenance'/('source_'+name))
 metrics={s:json.loads((exp/'results'/f'{s}.json').read_text()) for s in SCENES}
 summary={'experiment':exp.name,'variant':cfg['variant'],'seed':cfg['seed'],'reused_from':cfg['reused_from'],
          'macro_auc_percent':sum(m['auc_percent'] for m in metrics.values())/7,
          'macro_eer':sum(m['eer'] for m in metrics.values())/7,
          'scenes':{s:{k:m[k] for k in ['auc_percent','eer','checkpoint_sha256']} for s,m in metrics.items()},
          'match':'historical training-matched scores; unchanged scene JSON/CSV and checkpoint bytes'}
 write(exp/'results/summary.json',summary)
 (exp/'results/summary.md').write_text(f"# {exp.name}\n\n{cfg['model']['label']}, seed{cfg['seed']}. Seven DA scenes copied from `{cfg['reused_from']}`; original training provenance is preserved.\n\nMatched macro AUC: {summary['macro_auc_percent']:.4f}%; EER: {summary['macro_eer']:.6f}.\n\n[Publication chain](../../README.md).\n")
 art=json.loads((exp/'artifacts.json').read_text());by={x['path']:x for x in art['artifacts']}
 for rel in ['results/summary.json','results/summary.md','provenance/source_summary.json','provenance/source_summary.md']:
  q=exp/rel
  if q.exists():by[rel]={'path':rel,'producer_script':'scripts/prepare_da_publication_chain.py','run_id':'copy_'+exp.name,
      'status':'present','size_bytes':q.stat().st_size,'sha256':sha(q),'storage':'git','inputs':[cfg['reused_from']]}
 art['artifacts']=list(by.values());write(exp/'artifacts.json',art)

def main():
 entries=[]
 for name in SPECS:
  for seed in [42,43,44]:
   exp=BASE/f'20261117_{name}_s{seed}'
   if exp.exists():raise FileExistsError(exp)
   for sub in ['scripts','results','logs','ckpt','provenance']: (exp/sub).mkdir(parents=True,exist_ok=True)
   spec=model_spec(name);src=source_for(name,seed)
   cfg={'schema_version':2,'id':exp.name,'dataset':'da','family':'appearance','variant':name,'seed':seed,'model':spec,
        'training':{'epochs':30,'batch':64,'workers':8,'image_size':256,'max_lr':.0005,'weight_decay':.05,
                    'optimizer':'AdamW','scheduler':'OneCycleLR','data_root':'/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly',
                    'deterministic':True,'drop_last':True},
        'evaluation':{'match':'publication_match','patches':256,'cosine_channels':768,'topk':5,
                      'alpha':spec['alpha'],'batch':32,'label':'sequence*/test/<video>.npy; 1=anomaly',
                      'score_direction':'larger=more anomalous','aggregation':'concat within scene; macro mean of 7 scenes'},
        'reused_from':str(src.relative_to(ROOT)) if src else None}
   write(exp/'config.json',cfg)
   wrapper='''#!/usr/bin/env python3
from pathlib import Path
import sys
EXP=Path(__file__).resolve().parents[1]
ROOT=EXP.parents[3]
sys.path.insert(0,str(ROOT))
from scripts.da_publication_scene import main
if __name__ == '__main__':
    main(default_experiment=EXP)
'''
   (exp/'scripts/run_scene.py').write_text(wrapper)
   rec={'schema_version':2,'id':exp.name,'path':str(exp.relative_to(ROOT)),'dataset':'da','family':'appearance',
        'status':'completed' if src else 'planned','source_commit':None,'scripts':['scripts/run_scene.py'],'runs':[],
        'result_summary':'results/summary.md' if src else None,
        'limitations':['DA only; seed42/43/44 each has seven independently initialized scene models.',
                      'Fresh training unless explicitly copied; B removes unmatched blocks at architecture level, without warm-start/pruned weights.',
                      'Train uses publication_match.mean(); test retains the same per-frame score.'], 'variant':name}
   artifacts=[]
   if src:
    original=json.loads((src/'record.json').read_text())
    for filename in ['record.json','config.json','artifacts.json']:
     shutil.copy2(src/filename,exp/'provenance'/('source_'+filename))
    for script in (src/'scripts').glob('*.py'):
     shutil.copy2(script,exp/'provenance'/('source_'+script.name))
    for folder in ['logs','results','ckpt']:
     for p in sorted((src/folder).iterdir()):
      if not p.is_file():continue
      # Only the seven scene checkpoints; no unrelated local cache.
      if folder=='ckpt' and p.name not in [f'{s}_final.pth' for s in SCENES]:continue
      shutil.copy2(p,exp/folder/p.name)
    for scene in SCENES:
     metric=json.loads((exp/'results'/f'{scene}.json').read_text())
     assert metric['seed']==seed and metric['epochs']==30 and metric['trainable_parameters']==spec['params']
     assert sha(exp/'ckpt'/f'{scene}_final.pth')==metric['checkpoint_sha256']
    rec['source_commit']=original['source_commit'];rec['runs']=original['runs']
    rec['reused_from']=str(src.relative_to(ROOT));rec['provenance']='provenance/source_record.json'
    rec['limitations'].append('Copied results/checkpoints are historical; their source IDs, guards, source commits and run IDs are preserved, not replaced by the publication ID.')
    rec['source_training_script']= 'scripts/da_variant_runner.py' if name=='pt_rd' or (name=='d_ablation' and seed==42) else 'scripts/da_gl_ablation_runner.py'
    rec['review_status']='pending_preflight'
    for p in sorted(exp.rglob('*')):
     if not p.is_file() or p.name in ['artifacts.json','record.json','config.json','run_scene.py']:continue
     rel=str(p.relative_to(exp));artifacts.append({'path':rel,'producer_script':'scripts/prepare_da_publication_chain.py',
       'run_id':original['runs'][0]['run_id'] if original['runs'] else 'unknown','inputs':[str(src.relative_to(ROOT))],
       'status':'present','size_bytes':p.stat().st_size,'sha256':sha(p),
       'storage':'local ignored; no external backup' if p.suffix=='.pth' else 'git',
       'copied_from':str((src/rel).relative_to(ROOT)) if not rel.startswith('provenance/') else str(src.relative_to(ROOT)),
       'note':'Original producer/run preserved in provenance/source_artifacts.json; this entry records the byte copy.'})
   write(exp/'record.json',rec);write(exp/'artifacts.json',{'schema_version':2,'artifacts':artifacts})
   (exp/'README.md').write_text(f'''# {spec['label']} / DA / seed{seed}

唯一配置：{json.dumps(spec,ensure_ascii=False)}。

数据为DA七场景，每场景独立从头初始化并训练30 epoch，batch64，256×256，AdamW(max_lr=5e-4, weight_decay=0.05) + OneCycleLR；teacher冻结，tap_norm=True，code_dim=64。

逐patch计算1-cosine并取最大5个距离的均值。4-tap模型对四tap平均；12-tap模型对四个相邻三层组计算(1-alpha)*mean + alpha*max，再四组平均。训练仅增加batch mean；测试直接保留同一match逐帧分数，越大越异常。标签为sequence*/test/<video>.npy，1为异常，场景内concat计算AUC×100和离散ROC EER，最后七场景宏平均。每stage的三个头分别对应三个教师tap，不复用同一个头输出。

运行入口为本目录新建scripts/run_scene.py → scripts/da_publication_scene.py → model/publication_rd.py；不调用历史variant/GL/Loss runner。每场景单独logs/<scene>.log，断点重启追加；每epoch保存优化器、调度器和RNG状态。

{('复制自 '+str(src.relative_to(ROOT))+'；原始结果与权重不改写，原配置、record和产物索引保留于provenance/，完成性和模型等价性须通过publication_preflight。') if src else '状态：planned；先提交源码计划并通过preflight，再以精确source_commit运行。'}

B-ablation是从头训练的4-block架构，不是剪枝旧权重后的直接测试。C的alpha=0对四个等大小组取mean，数学上等于12层平均。默认超参数保持一致；不根据test标签选择模型或score方向。
''')
   entries.append({'path':str(exp.relative_to(ROOT)),'variant':name,'seed':seed,'reused_from':cfg['reused_from']})
   if src:finalize_copied_summary(exp)
   print('prepared',exp.name,'copy' if src else 'new',flush=True)
 write(BASE/'publication_chain_plan.json',{'schema_version':2,'dataset':'da','seeds':[42,43,44],'total_experiments':24,
      'reused_experiments':7,'new_experiments':17,'new_scene_jobs':119,'entries':entries,
      'fresh_training_entry':'scripts/da_publication_scene.py','scheduler':'scripts/da_publication_scheduler.py'})
if __name__=='__main__':main()
