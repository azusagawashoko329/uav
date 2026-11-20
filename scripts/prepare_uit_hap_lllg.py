#!/usr/bin/env python3
"""Create six fresh formal UIT-ADrone LLLG experiments and their exact plans."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.hap_lllg import model_spec
from scripts.uit_hap_lllg_runner import TRAINING,EVALUATION,save_json,now


def main():
    experiments=[]
    for variant in ['hap_rd','hap_rd_lite']:
        for seed in [42,43,44]:
            exp=ROOT/f'experiments/uit/appearance/20261120_formal_lllg_{variant}_s{seed}'
            if exp.exists():raise FileExistsError(exp)
            for sub in ['scripts','logs','results','ckpt']:(exp/sub).mkdir(parents=True)
            cfg={'schema_version':2,'id':exp.name,'dataset':'uit','family':'appearance',
                 'variant':variant,'seed':seed,'model':model_spec(variant),'training':TRAINING,'evaluation':EVALUATION,
                 'data_protocol':{'train':'41 normal videos / 59186 frames: train/frames/<video>/*.jpg',
                                  'test':'51 full videos / 147005 frames: test/frames/<video>/*.jpg',
                                  'labels':'test/test_frame_mask/<video>.npy; 63485 anomaly frames',
                                  'official_overlap_retained':True},
                 'initialization':'from scratch; no model checkpoint or result reused',
                 'model_entry':'model/hap_lllg.py:build_hap_lllg','score_entry':'model/hap_lllg.py:pldd_match'}
            save_json(exp/'config.json',cfg)
            save_json(exp/'record.json',{'schema_version':2,'id':exp.name,'path':str(exp.relative_to(ROOT)),
                 'dataset':'uit','family':'appearance','variant':variant,'status':'planned','source_commit':None,
                 'scripts':['scripts/run_uit.py'],'runs':[],'result_summary':None,
                 'limitations':['UIT official train/test overlap retained; no deduplication.',
                                'Whole-dataset model, not independent per-scene training.',
                                'Fresh LLLG+GATE formal model; historical GLLL V2/V5 are separate experiments.']})
            save_json(exp/'artifacts.json',{'schema_version':2,'artifacts':[]})
            save_json(exp/'architecture_version.json',{'version_id':f'formal_uit_lllg_{variant}_20261120',
                       'formal_model':variant,'model':model_spec(variant),
                       'teacher_targets_by_stage':[[9,10,11],[6,7,8],[3,4,5],[0,1,2]],
                       'implementation':'independent model/hap_lllg.py; no legacy builders at runtime',
                       'podp':'shared learnable 2x3, seeded random orthonormal init, forward Gram-Schmidt' if variant=='hap_rd_lite' else None})
            (exp/'scripts/run_uit.py').write_text('''#!/usr/bin/env python3
from pathlib import Path
import sys
EXP=Path(__file__).resolve().parents[1]
ROOT=EXP.parents[3]
sys.path.insert(0,str(ROOT))
from scripts.uit_hap_lllg_runner import main
if __name__=='__main__':
    main(default_experiment=EXP)
''')
            spec=model_spec(variant)
            (exp/'README.md').write_text(f'''# {spec['label']} / UIT-ADrone / seed{seed}

新训练的正式LLLG+GATE配置。数据集为UIT-ADrone；不是DA，也不复用历史GLLL V2/V5结果或权重。

学生固定3个GLMB后接1个GGMB，gated FFN hidden ratio2；student width={spec['student_width']}，PDFB hidden={spec['pdfb_hidden']}，code_dim64，12个独立重建头。可训练参数{spec['parameters']:,}，不含冻结教师。四stage的三个头分别重建teacher9–11、6–8、3–5、0–2。

Teacher为冻结DINOv3 ViT-B/16，taps0..11，输入256×256（256 patches），移除CLS+4 registers。每tap独立LayerNorm；Lite包含一个四组共用、可学习2×3 Q，按seed随机正交初始化，前向Gram-Schmidt保持行正交；完整HAP-RD无PODP。

## 数据与训练

沿用官方目录：train/frames下41个正常训练视频、59186帧；test/frames下51个测试视频、147005帧，标签test/test_frame_mask/<video>.npy，共63485异常帧。保留历史UIT协议中的官方train/test重叠，不去重。每个seed仅训练一个覆盖整个训练集的模型。

45 epoch、batch64、workers8、AdamW(max_lr=5e-4, weight_decay=0.05)+OneCycleLR，deterministic cuDNN、train drop_last=True。checkpoint保存epoch10/20/30/40/45与final；每个epoch另更新可恢复的uit_resume.pth，含optimizer、scheduler、全部RNG和DataLoader generator。

## Match评分

每个教师/重建对按patch计算1-cosine，取最大5个距离平均；四组三层各算0.5mean+0.5max，再四组平均。训练只增加batch mean，测试保留相同pldd_match的逐帧score。分数越大越异常；1为异常标签。测试batch32、workers4，按视频/帧排序将所有147005个测试帧拼接，计算一项全局AUC×100和离散ROC EER，不做场景宏平均、分数反转或测试集择优读出。

结果文件为results/uit.json、results/matched_scores.csv；AUC/EER的EER取fpr[argmin(abs(1-tpr-fpr))]，不插值。训练和测试均由新入口执行。

## 入口与运行

本目录scripts/run_uit.py→新写scripts/uit_hap_lllg_runner.py→独立model/hap_lllg.py。运行代码不调用历史student、bottleneck、variant builder或loss接口；只复用冻结教师骨干。实际模型guard检查模块类型、顺序、维度、FFN、路由和PODP，任何错配均拒绝训练。

先提交计划和preflight，通过后绑定精确干净HEAD为source_commit；每个子任务启动时核验代码/config SHA-256、数据指纹和教师权重SHA-256。完整preflight及六任务run_plan位于同批HAP-RD seed42目录。动态双卡以UUID固定物理卡、每卡一个任务，完成立即补位；不做外部占用等待。

调度器关闭stdin、start_new_session=True，忽略SIGHUP；logs/run.log为唯一训练日志，恢复追加。中断后同目录恢复完整epoch；45epoch已完成时可从resume继续测试。已完成实验跳过，失败/中止证据保留。代码、小型CSV/JSON、日志入Git；权重留本地并由Git忽略，在artifacts.json登记大小与哈希。

[精确配置](config.json)，[结构版本](architecture_version.json)，[运行记录](record.json)。状态以record为准。
''')
            experiments.append(str(exp.relative_to(ROOT)))
            print('prepared',exp.name,flush=True)
    owner=ROOT/experiments[0]
    save_json(owner/'run_plan.json',{'schema_version':2,'dataset':'uit','run_id':now()+'_uit_hap_lllg_s42_s43_s44',
               'source_commit':None,'experiments':experiments,'gpus':[0,1],'new_experiments':6,'new_jobs':6,
               'scheduler':'scripts/uit_hap_lllg_scheduler.py','policy':'fresh whole-UIT,45ep,LLLG+GATE,alpha0.5,force two GPUs'})


if __name__=='__main__':
    main()
