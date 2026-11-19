#!/usr/bin/env python3
"""Create three independent, registered DA LGGD alpha0 experiments once."""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.lggd_no_laga import VARIANT, ablation_spec
from scripts.da_laga_scene import save_json


def main():
    paths=[]
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for seed in (42,43,44):
        exp=ROOT/f'experiments/da/aggregation/20261119_laga_lggd_a0_s{seed}'
        if exp.exists():
            raise FileExistsError(exp)
        for sub in ('scripts','logs','results','ckpt'):
            (exp/sub).mkdir(parents=True)
        baseline=f'ablation/da/publication/20261117_hap_rd_s{seed}'
        baseline_cfg=json.loads((ROOT/baseline/'config.json').read_text())
        cfg={'schema_version':2,'id':exp.name,'dataset':'da','family':'aggregation','variant':VARIANT,
             'seed':seed,'model':ablation_spec(),'training':baseline_cfg['training'],
             'evaluation':{**baseline_cfg['evaluation'],'alpha':0.0,'match':'ablation_match','workers':4},
             'baseline':baseline,'aliases':['d-new-ablation','LGGD without LAGA'],
             'changed_factor':'LAGA alpha 0.5 -> 0 in both training and matched evaluation',
             'formal_publication_model':False}
        save_json(exp/'config.json',cfg)
        save_json(exp/'record.json',{'schema_version':2,'id':exp.name,'path':str(exp.relative_to(ROOT)),
                  'dataset':'da','family':'aggregation','status':'planned','source_commit':None,
                  'scripts':['scripts/run_scene.py'],'runs':[],'result_summary':None,
                  'variant':VARIANT,'baseline':baseline,
                  'limitations':['DA only; seven independent scene models; no PODP.',
                                 'Alpha changes both train objective and matched test readout.',
                                 'Three seeds are descriptive; benefits need not agree across decoder architectures.']})
        save_json(exp/'artifacts.json',{'schema_version':2,'artifacts':[]})
        save_json(exp/'architecture_version.json',{'version_id':'pub_lggd_without_laga_20261119',
                  'formal_publication_model':False,'variant':VARIANT,'model':ablation_spec(),
                  'baseline_model':'HAP-RD','only_hyperparameter_change':{'alpha':{'baseline':0.5,'ablation':0.0}},
                  'model_entry':'model/lggd_no_laga.py:build_ablation_model',
                  'score_entry':'model/lggd_no_laga.py:ablation_match',
                  'teacher_student_routes':[[9,10,11],[6,7,8],[3,4,5],[0,1,2]],
                  'note':'Three independent heads per student stage; not one shared prediction matched three times.'})
        (exp/'scripts/run_scene.py').write_text('''#!/usr/bin/env python3
from pathlib import Path
import sys
EXP=Path(__file__).resolve().parents[1]
ROOT=EXP.parents[3]
sys.path.insert(0,str(ROOT))
from scripts.da_laga_scene import main
if __name__=='__main__':
    main(default_experiment=EXP)
''')
        (exp/'README.md').write_text(f'''# HAP-RD w/o LAGA / DA / seed{seed}

过程消融别名：d-new-ablation、LGGD without LAGA。固定版本：pub_lggd_without_laga_20261119；不是新的正式模型。

与同seed的 [HAP-RD](../../../../{baseline}/config.json) 比较。仅改变训练与match测试中的 α：0.5→0；七个场景均从头训练。四个等大小三层组在α=0时等价于12层平均。结果是否改善是待检验假设，不预设与其他decoder模板一致。

冻结DINOv3 ViT-B/16，teacher taps0..11，LLLG+gate ratio2、student width768、12个独立重建头、PDFB hidden512、code_dim64、tap_norm=True，无PODP。可训练参数30,248,000。四stage分别用三个独立头重建teacher9–11、6–8、3–5、0–2。

仅DA；七场景独立训练，30 epoch、batch64、workers8、256×256输入（256 patches）；AdamW(max_lr=5e-4, weight_decay=0.05)+OneCycleLR，deterministic cuDNN、train drop_last。Top-5 patch cosine distance→组内mean→四组mean。训练取同一ablation_match的batch mean，测试保留每帧score。分数越大越异常；sequence*/test/<video>.npy为标签，1=异常；场景内concat计算AUC×100及离散ROC EER，再七场景等权宏平均。

专用入口：本目录scripts/run_scene.py→scripts/da_laga_scene.py→model/lggd_no_laga.py。启动前核验实际模型、梯度、头路由及α=0评分等价性；独立preflight代码哈希固定。初次运行先提交计划，record中的source_commit由调度器绑定实际干净HEAD，历史模型和原结果不覆盖。

动态双卡按训练帧数优先的一条跨seed队列；每卡一个本轮任务，完成立即补位，不做外部占用门控。调度器以独立session运行、stdin关闭，忽略SIGHUP。每场景单一logs/<scene>.log；每epoch保存模型、优化器、调度器和全部RNG，恢复从已完成epoch继续，已有完成结果跳过，失败日志保留。权重由Git忽略，产物索引记录哈希。

[版本配置](architecture_version.json)，[运行记录](record.json)。三seed运行计划及preflight位于seed42目录。
''')
        paths.append(str(exp.relative_to(ROOT)))
        print('prepared',exp.name,flush=True)
    owner=ROOT/paths[0]
    save_json(owner/'run_plan.json',{'schema_version':2,'run_id':stamp+'_da_lggd_no_laga_s42_s43_s44',
              'source_commit':None,'experiments':paths,'gpus':[0,1],'new_experiments':3,'new_scene_jobs':21,
              'scheduler':'scripts/da_laga_scheduler.py','model_entry':'model/lggd_no_laga.py',
              'score_entry':'ablation_match; alpha0 for training and test'})


if __name__=='__main__':
    main()
