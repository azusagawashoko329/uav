#!/usr/bin/env python3
"""Audit saved formal UIT HAP-RD/Lite results on CPU and regenerate summaries."""
from __future__ import annotations
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics as st
import subprocess
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score,roc_curve

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.hap_lllg import build_hap_lllg,actual_guard
from scripts.uit_hap_lllg_runner import DATA_ROOT,checked_config,data_manifest,read,save_json,sha,state_hash

OWNER=ROOT/'experiments/uit/appearance/20261120_formal_lllg_hap_rd_s42'
SEEDS=(42,43,44)
VARIANTS=('hap_rd','hap_rd_lite')
PRODUCER='scripts/summarize_uit_hap_lllg.py'


def stats(values):
    return {'mean':st.mean(values),'sample_sd':st.stdev(values),'by_seed':dict(zip(map(str,SEEDS),values))}


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def register(exp,paths,rid,inputs):
    payload=read(exp/'artifacts.json');items={x['path']:x for x in payload['artifacts']}
    for p in paths:
        rel=str(p.relative_to(exp))
        items[rel]={'path':rel,'producer_script':PRODUCER,'run_id':rid,'inputs':inputs,'status':'present',
                    'size_bytes':p.stat().st_size,'sha256':sha(p),'storage':'git'}
    payload['artifacts']=list(items.values());save_json(exp/'artifacts.json',payload)


def plot(payload):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(9,4.3),layout='constrained')
    colors=['#286a96','#c1772b','#3b9465']
    for ax,metric,label in zip(axes,['auc_percent','eer'],['Global frame AUC (%)','Global frame EER (fraction)']):
        for index,variant in enumerate(VARIANTS):
            values=payload['models'][variant][metric]
            for k,seed in enumerate(SEEDS):
                ax.scatter(index+(k-1)*.08,values['by_seed'][str(seed)],color=colors[k],s=30,
                           label=f'seed{seed}' if index==0 else None)
            ax.errorbar(index,values['mean'],yerr=values['sample_sd'],fmt='D',color='#333333',
                        capsize=4,markersize=5,label='mean ± sample SD' if index==0 else None)
        ax.set(xticks=[0,1],xticklabels=['HAP-RD','HAP-RD-Lite'],ylabel=label,xlim=(-.4,1.4))
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('UIT-ADrone / LLLG + GATE / 45 epochs / training-matched PLDD')
    paths=[]
    for ext in ('png','svg'):
        path=OWNER/'results'/f'formal_models.{ext}';fig.savefig(path,dpi=180)
        if ext=='svg':path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
        paths.append(path)
    plt.close(fig)
    return paths


def report(payload):
    models=payload['models'];full=models['hap_rd'];lite=models['hap_rd_lite'];pair=payload['lite_minus_full']
    lines=['# UIT-ADrone：正式 HAP-RD / HAP-RD-Lite 三种子汇总','',
        '**本轮六个独立实验全部完成，均为LLLG+GATE、45 epoch、训练匹配的Top-5/四组α=0.5 PLDD。** '
        f"HAP-RD的全局AUC为{full['auc_percent']['mean']:.4f}±{full['auc_percent']['sample_sd']:.4f}%，"
        f"Lite为{lite['auc_percent']['mean']:.4f}±{lite['auc_percent']['sample_sd']:.4f}%。"
        'Lite在三个种子中都明显降低AUC并提高EER，体现较大的压缩代价，不能描述为无损轻量化。','',
        '## 模型与评价口径','',
        '- 两者均为独立新写入口`model/hap_lllg.py`、`scripts/uit_hap_lllg_runner.py`；'
        '实际学生顺序是GLMB→GLMB→GLMB→GGMB。12个独立头按四stage依次重建teacher9–11、6–8、3–5、0–2。',
        '- HAP-RD：width768、PDFB hidden512、无PODP，30,248,000个可训练参数；'
        'Lite：width192、hidden128、共享可学习2×3 PODP Q，3,761,222个可训练参数。冻结DINOv3 ViT-B/16不计入参数量。',
        '- teacher taps0..11，输入256×256，共256个patch；逐patch cosine distance取最大5个平均，'
        '四组三层各算0.5mean+0.5max再平均。训练取同一`pldd_match`的batch mean，测试直接使用逐帧分数。',
        '- UIT官方训练集为41个视频/59,186帧；测试51个视频/147,005帧，其中63,485帧为异常。'
        '保留历史协议中的官方train/test重叠，不去重。训练45 epoch、batch64、AdamW+OneCycleLR，max_lr=5e-4、weight_decay=0.05。',
        '- **将所有测试帧拼接后计算一项全局AUC×100和EER**，不是51个视频AUC的宏平均，也不同于DA的七场景独立训练/宏平均。'
        '分数越大越异常；EER为0–1比例，取离散ROC的`fpr[argmin(abs(1-tpr-fpr))]`，不插值。',
        '- 均值和样本标准差（ddof=1）在seed42/43/44的三项全局指标上计算；SD不是标准误或置信区间。','',
        '## 三种子矩阵','',
        '| 模型 | 参数M | seed42 AUC | seed43 AUC | seed44 AUC | AUC均值 ± SD | EER均值 ± SD |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        r=models[variant];a,e=r['auc_percent'],r['eer']
        lines.append(f"| {r['label']} | {r['parameters']/1e6:.6f} | "+' | '.join(f"{a['by_seed'][str(s)]:.4f}" for s in SEEDS)
                     +f" | {a['mean']:.4f} ± {a['sample_sd']:.4f} | {e['mean']:.6f} ± {e['sample_sd']:.6f} |")
    lines+=['','[全精度矩阵CSV](multiseed_matrix.csv)，[机器可读汇总](multiseed_summary.json)。','',
            '## Lite − HAP-RD 的同seed差值','',
            '| seed | ΔAUC（百分点） | ΔEER（比例） |','|---|---:|---:|']
    for s in map(str,SEEDS):lines.append(f"| {s} | {pair['auc_delta_pp']['by_seed'][s]:+.4f} | {pair['eer_delta']['by_seed'][s]:+.6f} |")
    lines.append(f"| 均值 ± SD | {pair['auc_delta_pp']['mean']:+.4f} ± {pair['auc_delta_pp']['sample_sd']:.4f} | "
                 f"{pair['eer_delta']['mean']:+.6f} ± {pair['eer_delta']['sample_sd']:.6f} |")
    lines+=['',
        f"Lite将可训练参数减少{payload['parameter_reduction_percent']:.2f}%，但全局AUC平均下降"
        f"{abs(pair['auc_delta_pp']['mean']):.4f}个百分点；EER升高{pair['eer_delta']['mean']:.6f}，约6.53个百分点。"
        '三个种子的AUC差值为−6.1189、−6.2203、−6.2747，方向一致，差值SD仅0.0791。'
        '本轮观察到的压缩差距明显大于各模型自身跨seed波动，但n=3仍不作显著性或总体性能保证。','',
        '从完整模型到Lite同时改变学生宽度、瓶颈hidden并加入PODP，因此这不是PODP单因素消融，'
        '不能把6.2个百分点的差距单独归因于PODP、学生宽度或某一个压缩环节；若要区分它们，需在UIT上另做F1/F2等过程对照。'
        'DA上同名模型的结果采用另一评价粒度，不能把两个数据集的绝对AUC直接当成配对对照。','',
        '## 测试视频诊断','',
        '为解释全局AUC差值，另统计每个含异常帧视频的“该视频正例对全体测试负例的胜率”。'
        '它包含跨视频正负样本配对；乘以该视频正例占总正例比例后，对所有视频求和恰好等于全局AUC。'
        '下表列出对Lite−HAP-RD全局AUC变化贡献最负的五个视频，单位为百分点；'
        '这是一种精确的排序分解，不是对单个视频独立AUC或单帧因果关系的归因。','',
        '| 视频 | 正例帧数 | 平均全局AUC差值贡献（百分点） |','|---|---:|---:|']
    for r in payload['largest_negative_video_contributions']:
        lines.append(f"| {r['video']} | {r['positive_frames']} | {r['mean_lite_minus_full_contribution_pp']:+.4f} |")
    lines+=['','完整逐视频、逐seed指标见[video_diagnostics.csv](video_diagnostics.csv)，'
            '全局差值分解见[video_contributions.csv](video_contributions.csv)。'
            '视频若只有单一标签，其独立AUC/EER留空；不会丢弃这些帧或重新定义主指标。','',
            '## 与历史GLLL结果的有限对照','',
            '| 结构规模 | 旧GLLL seed42 AUC | 新LLLG seed42 AUC | ΔAUC（百分点） | 旧EER | 新EER |',
            '|---|---:|---:|---:|---:|---:|']
    for r in payload['historical_seed42']:
        lines.append(f"| {r['label']} | {r['old_auc']:.4f} | {r['new_auc']:.4f} | {r['auc_delta_pp']:+.4f} | {r['old_eer']:.6f} | {r['new_eer']:.6f} |")
    lines+=['',
        '旧结果为20261104 UIT矩阵V2/V5的matched评分，均45 epoch、seed42；新结果全部从头训练。'
        '两个规模的seed42均有小幅正向观察，但旧模型只有一个seed，且是历史入口与硬件分配下的运行，'
        '不足以认定LLLG在UIT上有稳定、多种子优势；不能把旧GLLL改名成新正式模型，也不能用旧单seed替代缺失的多seed基线。','',
        '## 运行与复核','',
        f"调度器于北京时间{payload['scheduler_started_local']}启动，{payload['scheduler_finished_local']}结束，"
        f"总墙钟{payload['scheduler_wall_hours']:.2f}小时；6个新任务完成、0复用、0失败。"
        '每个实验均保存epoch10/20/30/40/45、final及可恢复resume。'
        '运行期间发生过明显的图片读取等待，且两张卡型号不同，故墙钟时间不能作为受控推理速度比较。'
        '本轮没有测端到端延迟、FPS或FLOPs，参数减少不代表相同比例的加速。','',
        '[validation.json](validation.json)核验六份CSV共882,030行预测，每份的帧顺序、标签、有限分数及全局AUC/EER均与数据集/结果文件一致；'
        '六份final权重校验SHA-256并在CPU严格加载，所有42份已登记checkpoint校验大小和哈希。'
        '六个日志均完整包含epoch1–45和51次视频评测，source/config/guard、教师权重指纹及match配置一致。'
        '不同seed的实际瓶颈/解码器参数指纹也不同。','',
        '固定训练源码为`992c9950804c45bbf9aec3302764594a29ed992b`；'
        '统计与归档保留此指针，不把汇总提交冒充训练源码。训练前模型路由、梯度及PLDD等价性证据见同目录上一级的preflight.json。','',
        '复现汇总：`/home/lyx/miniconda3/envs/DMAD/bin/python scripts/summarize_uit_hap_lllg.py`。'
        '仅CPU审计已有产物，不重训、不重新推理或修改逐帧分数。权重留本地并由Git忽略；'
        'Git同步包含代码、小型CSV/JSON、日志、图表和报告。','']
    p=OWNER/'results/multiseed_summary.md';p.write_text('\n'.join(lines));return p


def main():
    torch.set_num_threads(2)
    plan=read(OWNER/'run_plan.json');audit=read(OWNER/'preflight.json');rid=plan['run_id'];source=plan['source_commit']
    events=[json.loads(l) for l in (OWNER/'logs'/f'scheduler_{rid}.jsonl').read_text().splitlines()]
    assert events[-1]['kind']=='all_finished' and events[-1]['completed']==6 and events[-1]['skipped']==0 and events[-1]['failures']==[]
    starts={e['experiment']:e for e in events if e['kind']=='launch'};ends={e['experiment']:e for e in events if e['kind']=='finish'}
    assert len(starts)==len(ends)==6 and audit['status']=='passed'
    for rel,digest in audit['code_hashes'].items():
        assert sha(ROOT/rel)==digest,rel
        assert hashlib.sha256(subprocess.check_output(['git','show',f'{source}:{rel}'],cwd=ROOT)).hexdigest()==digest,rel
    _,videos,split=data_manifest();assert split==audit['data']
    keys=[]
    for video in videos:
        frames=sorted(video.glob('*.jpg'),key=lambda p:int(p.stem))
        y=np.asarray(np.load(DATA_ROOT/'test/test_frame_mask'/f'{video.name}.npy',allow_pickle=True)).astype(int).ravel()
        keys.extend((video.name,p.name,int(label)) for p,label in zip(frames,y))
    assert len(keys)==147005 and len(set((v,f) for v,f,y in keys))==len(keys)
    labels=np.array([x[2] for x in keys]);video_names=np.array([x[0] for x in keys]);video_order=[v.name for v in videos]
    checks=[];data={};video_rows=[];contributions={};checkpoint_count=0
    print('UIT data and committed source fingerprints verified',flush=True)
    for rel in plan['experiments']:
        exp=ROOT/rel;cfg=checked_config(exp);variant=cfg['variant'];seed=cfg['seed'];m=read(exp/'results/uit.json');record=read(exp/'record.json')
        assert record['status']=='completed' and record['source_commit']==source and record['runs'][-1]['status']=='completed'
        assert m['dataset']=='UIT-ADrone' and m['seed']==seed and m['variant']==variant and m['epochs']==45
        assert m['run_id']==rid and m['source_commit']==source and m['code_hashes']==audit['code_hashes']
        assert m['split_fingerprint']==split['split_fingerprint'] and m['teacher_state_sha256']==audit['teacher']['state_sha256']
        assert m['official_overlap_retained'] is True and 'training=batch mean, test=frame score' in m['match']
        with (exp/'results/matched_scores.csv').open() as f:rows=list(csv.DictReader(f))
        assert [(r['video'],r['frame'],int(r['label'])) for r in rows]==keys
        scores=np.array([float(r['anomaly_score']) for r in rows]);assert np.isfinite(scores).all()
        auc=float(roc_auc_score(labels,scores)*100);fpr,tpr,_=roc_curve(labels,scores)
        eer=float(fpr[np.nanargmin(np.abs(1-tpr-fpr))])
        assert math.isclose(auc,m['auc_percent'],abs_tol=1e-9,rel_tol=0) and math.isclose(eer,m['eer'],abs_tol=1e-12,rel_tol=0)
        assert m['frames']==147005 and m['positive_frames']==63485 and m['train_frames']==59186
        assert ends[rel]['returncode']==0
        lines=(exp/'logs/run.log').read_text().splitlines()
        assert [int(re.match(r'epoch=(\d+)',l)[1]) for l in lines if l.startswith('epoch=')]==list(range(1,46))
        assert [int(re.match(r'eval_video=(\d+)',l)[1]) for l in lines if l.startswith('eval_video=')]==list(range(1,52))
        assert json.loads(lines[-1])==m
        guards=[json.loads(l) for l in lines if l.startswith('{') and '"kind": "actual_model_guard"' in l]
        assert len(guards)==1 and guards[0]['guard']==m['guard'] and guards[0]['source_commit']==source
        assert read(exp/'results/status.json')['status']=='completed' and read(exp/'results/eval_status.json')['completed_videos']==51
        for item in read(exp/'artifacts.json')['artifacts']:
            p=exp/item['path'];assert p.is_file() and p.stat().st_size==item['size_bytes'] and sha(p)==item['sha256'],str(p)
            if p.suffix=='.pth':checkpoint_count+=1
        ck=exp/m['checkpoint'];assert sha(ck)==m['checkpoint_sha256']
        weight=torch.load(ck,map_location='cpu',weights_only=False);bn,decoder=build_hap_lllg(variant,seed)
        bn.load_state_dict(weight['bottleneck_state_dict'],strict=True);decoder.load_state_dict(weight['decoder_state_dict'],strict=True)
        guard=json.loads(json.dumps(actual_guard(bn,decoder,variant)))
        assert guard==m['guard']==json.loads(json.dumps(weight['guard']))
        for field in ['source_commit','run_id','variant','seed','code_hashes','split_fingerprint','teacher_state_sha256','match']:
            assert weight[field]==m[field]
        assert weight['epoch']==45 and weight['train_loss']==m['train_loss']
        signatures={'bottleneck':state_hash(bn),'decoder':state_hash(decoder)}
        check={'experiment':exp.name,'variant':variant,'seed':seed,'auc_percent':auc,'eer':eer,'frames':len(rows),
               'scores_sha256':sha(exp/'results/matched_scores.csv'),'checkpoint_sha256':m['checkpoint_sha256'],
               'checkpoint_strict_load':'passed','guard':guard,'trained_state_signatures':signatures,
               'effective_q':bn.effective_depth_basis().detach().tolist() if cfg['model']['podp'] else None}
        checks.append(check);data[variant,seed]=m
        cached=read(exp/'results/summary.json');assert cached['auc_percent']==m['auc_percent'] and cached['eer']==m['eer']
        # Exact global AUC decomposition by the video of each positive frame.
        negative=np.sort(scores[labels==0]);positive_scores=scores[labels==1]
        concordance=(np.searchsorted(negative,positive_scores,side='left')+np.searchsorted(negative,positive_scores,side='right'))/(2*len(negative))
        assert math.isclose(float(concordance.mean()*100),auc,abs_tol=1e-10,rel_tol=0)
        positive_videos=video_names[labels==1]
        for video in video_order:
            mask=video_names==video;y=labels[mask];s=scores[mask];pcount=int(y.sum());mixed=len(np.unique(y))==2
            va=ve=None
            if mixed:
                va=float(roc_auc_score(y,s)*100);fp,tp,_=roc_curve(y,s);ve=float(fp[np.nanargmin(np.abs(1-tp-fp))])
            contribution=float(concordance[positive_videos==video].sum()*100/63485) if pcount else 0.
            contributions[variant,seed,video]=contribution
            video_rows.append({'variant':variant,'seed':seed,'video':video,'frames':len(y),'positive_frames':pcount,
                               'within_video_auc_percent':va,'within_video_eer':ve,
                               'positive_video_global_auc_contribution_pp':contribution})
        assert math.isclose(sum(contributions[variant,seed,v] for v in video_order),auc,abs_tol=1e-10,rel_tol=0)
        record['review_status']='passed_postrun_audit';record['review_report']='../20261120_formal_lllg_hap_rd_s42/results/validation.json'
        record['weight_files_local']=len(list((exp/'ckpt').glob('*.pth')));save_json(exp/'record.json',record)
        del bn,decoder,weight,rows;gc.collect()
        print('audited',exp.name,flush=True)
    assert checkpoint_count==42
    for variant in VARIANTS:
        for kind in ['bottleneck','decoder']:
            assert len({c['trained_state_signatures'][kind] for c in checks if c['variant']==variant})==3
    models={variant:{'label':data[variant,42]['guard']['label'],'parameters':data[variant,42]['guard']['parameters'],
                     'auc_percent':stats([data[variant,s]['auc_percent'] for s in SEEDS]),
                     'eer':stats([data[variant,s]['eer'] for s in SEEDS])} for variant in VARIANTS}
    pair={'auc_delta_pp':stats([data['hap_rd_lite',s]['auc_percent']-data['hap_rd',s]['auc_percent'] for s in SEEDS]),
          'eer_delta':stats([data['hap_rd_lite',s]['eer']-data['hap_rd',s]['eer'] for s in SEEDS])}
    video_deltas=[]
    for video in video_order:
        row=next(x for x in video_rows if x['video']==video)
        video_deltas.append({'video':video,'positive_frames':row['positive_frames'],
          **{f'seed{s}_lite_minus_full_contribution_pp':contributions['hap_rd_lite',s,video]-contributions['hap_rd',s,video] for s in SEEDS},
          'mean_lite_minus_full_contribution_pp':st.mean(contributions['hap_rd_lite',s,video]-contributions['hap_rd',s,video] for s in SEEDS)})
    assert math.isclose(sum(r['mean_lite_minus_full_contribution_pp'] for r in video_deltas),pair['auc_delta_pp']['mean'],abs_tol=1e-10,rel_tol=0)
    historical=[]
    for variant,stem in [('hap_rd','v2_rd_s4_t12_gl768_h512'),('hap_rd_lite','v5_rd_s4_t12_gl192_h128_qr2')]:
        path=ROOT/f'experiments/uit/appearance/20261104_variant_matrix_{stem}_s42/results/uit_matched.json';old=read(path);new=data[variant,42]
        historical_cfg=read(path.parent.parent/'config.json')
        assert old['seed']==42 and historical_cfg['training']['epochs']==45 and old['split_fingerprint']==split['split_fingerprint']
        historical.append({'label':models[variant]['label'],'source':str(path.relative_to(ROOT)),'sha256':sha(path),
                           'old_auc':old['auc_percent'],'new_auc':new['auc_percent'],'auc_delta_pp':new['auc_percent']-old['auc_percent'],
                           'old_eer':old['eer'],'new_eer':new['eer']})
    parse=lambda t:dt.datetime.strptime(t,'%Y%m%dT%H%M%SZ').replace(tzinfo=dt.timezone.utc)
    local=lambda t:parse(t).astimezone(dt.timezone(dt.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    payload={'schema_version':1,'dataset':'uit','seeds':list(SEEDS),'models':models,'lite_minus_full':pair,
             'aggregation':'global concatenated test-frame AUC/EER per seed; three-seed mean and sample SD, ddof=1',
             'run_source_commit':source,'split':split,'parameter_reduction_percent':100*(1-3761222/30248000),
             'historical_seed42':historical,
             'largest_negative_video_contributions':sorted(video_deltas,key=lambda r:r['mean_lite_minus_full_contribution_pp'])[:5],
             'scheduler_started_local':local(events[0]['time']),'scheduler_finished_local':local(events[-1]['time']),
             'scheduler_wall_hours':(parse(events[-1]['time'])-parse(events[0]['time'])).total_seconds()/3600}
    save_json(OWNER/'results/multiseed_summary.json',payload)
    save_json(OWNER/'results/validation.json',{'status':'passed','experiments':6,'prediction_rows':882030,
              'scheduler_final':events[-1],'committed_source_hashes_verified':True,'dataset_labels_and_order_verified':True,
              'final_checkpoints_hash_verified_and_strict_loaded':6,'registered_checkpoints_hash_verified':checkpoint_count,
              'distinct_trained_state_signatures_across_seeds':True,'global_auc_video_decomposition_verified':True,'checks':checks})
    write_csv(OWNER/'results/multiseed_matrix.csv',[{'variant':v,'parameters':models[v]['parameters'],
              **{f'seed{s}_{metric}':models[v][metric]['by_seed'][str(s)] for s in SEEDS for metric in ['auc_percent','eer']},
              **{f'{stat}_{metric}':models[v][metric][stat] for stat in ['mean','sample_sd'] for metric in ['auc_percent','eer']}} for v in VARIANTS])
    write_csv(OWNER/'results/paired_deltas.csv',[{'seed':s,'lite_minus_full_auc_pp':pair['auc_delta_pp']['by_seed'][str(s)],
               'lite_minus_full_eer':pair['eer_delta']['by_seed'][str(s)]} for s in SEEDS])
    write_csv(OWNER/'results/video_diagnostics.csv',video_rows);write_csv(OWNER/'results/video_contributions.csv',video_deltas)
    figures=plot(payload);text=report(payload)
    generated=[OWNER/'results'/x for x in ['multiseed_summary.json','validation.json','multiseed_matrix.csv',
                'paired_deltas.csv','video_diagnostics.csv','video_contributions.csv']]+figures+[text]
    register(OWNER,generated,rid,[str(Path(rel)/'results/uit.json') for rel in plan['experiments']])
    for rel in plan['experiments']:
        exp=ROOT/rel;review=exp/'results/review.md'
        review.write_text('# UIT正式模型复核\n\n全局AUC/EER、147005帧标签/顺序、45 epoch日志、模型guard及权重哈希通过复核。\n\n'
                          '[三种子汇总与模型对比](../../20261120_formal_lllg_hap_rd_s42/results/multiseed_summary.md)。\n')
        register(exp,[review],rid,[str(text.relative_to(ROOT)),'results/uit.json'])
    print('PASSED: six runs, 882030 scores, six strict final loads, 42 checkpoint hashes; report and figures written.',flush=True)


if __name__=='__main__':
    main()
