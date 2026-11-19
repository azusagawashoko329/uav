#!/usr/bin/env python3
"""CPU audit and three-seed LAGA comparison; does not train or rescore images."""
from __future__ import annotations

import csv
import gc
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.lggd_no_laga import build_ablation_model, ablation_guard
from model.publication_rd import build_publication_model, architecture_guard, load_historical_weights
from model.bottleneck import build_bottleneck
from model.student_structural import build_structural_student
from scripts.da_laga_scene import SCENES, checked_config, data_manifest, save_json, sha

OWNER = ROOT/'experiments/da/aggregation/20261119_laga_lggd_a0_s42'
SEEDS = (42, 43, 44)
PRODUCER = 'scripts/summarize_da_laga.py'


def read(path):
    return json.loads(path.read_text())


def stats(values):
    return {'mean': st.mean(values), 'sample_sd': st.stdev(values),
            'by_seed': dict(zip(map(str, SEEDS), values))}


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def register(exp, paths, rid, inputs):
    payload = read(exp/'artifacts.json')
    items = {x['path']: x for x in payload['artifacts']}
    for path in paths:
        rel = str(path.relative_to(exp))
        items[rel] = {'path': rel, 'producer_script': PRODUCER, 'run_id': rid,
                      'inputs': inputs, 'status': 'present', 'size_bytes': path.stat().st_size,
                      'sha256': sha(path), 'storage': 'git'}
    payload['artifacts'] = list(items.values())
    save_json(exp/'artifacts.json', payload)


def historical_context(current):
    sources = [
        ('GGGG + GELU', ROOT/'ablation/da/publication/summary.json', 'c_ablation', 'd_ablation'),
        ('GLLL + GATE', ROOT/'ablation/da/Loss/summary.json', 'top5_12mean', 'top5_group_a05'),
    ]
    rows = []
    for label, path, left, right in sources:
        matrix = {r['id']: r for r in read(path)['matrix']}
        a, b = matrix[left], matrix[right]
        def values(row, metric):
            return row[metric].get('by_seed', row[metric].get('values'))
        rows.append({'template': label, 'alpha0_auc': a['auc_percent']['mean'],
                     'alpha05_auc': b['auc_percent']['mean'],
                     'auc_delta_pp': stats([values(b,'auc_percent')[str(s)]-values(a,'auc_percent')[str(s)] for s in SEEDS]),
                     'eer_delta': stats([values(b,'eer')[str(s)]-values(a,'eer')[str(s)] for s in SEEDS]),
                     'source_summary': str(path.relative_to(ROOT)), 'source_sha256': sha(path),
                     'alpha0_id': left, 'alpha05_id': right,
                     'evidence': 'historical audited summary; no new training or inference'})
    rows.append({'template': 'LLLG + GATE', 'alpha0_auc': current['alpha0']['auc_percent']['mean'],
                 'alpha05_auc': current['alpha05']['auc_percent']['mean'],
                 'auc_delta_pp': current['effect']['auc_delta_pp'], 'eer_delta': current['effect']['eer_delta'],
                 'source_summary': 'this comparison', 'evidence': '42 scene CSVs and final checkpoints audited now'})
    return rows


def plot(payload):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    for i, (scene, row) in enumerate(payload['scenes'].items()):
        vals = list(row['auc_delta_pp']['by_seed'].values())
        axes[0].scatter(vals, [i]*3, s=28, color='#286a96', alpha=.7)
        axes[0].scatter(row['auc_delta_pp']['mean'], i, s=35, color='#b7472a', marker='D')
    axes[0].set(yticks=range(7), yticklabels=list(payload['scenes']),
                xlabel='AUC change (percentage points)', title='LLLG + GATE: scene effects')
    axes[0].invert_yaxis()
    for i, row in enumerate(payload['template_context']):
        axes[1].scatter(list(row['auc_delta_pp']['by_seed'].values()), [i]*3,
                        s=28, color='#286a96', alpha=.7)
        axes[1].scatter(row['auc_delta_pp']['mean'], i, s=35, color='#b7472a', marker='D')
    axes[1].set(yticks=range(3), yticklabels=[r['template'] for r in payload['template_context']],
                xlabel='Macro AUC change (percentage points)', title='Template comparison (descriptive)')
    axes[1].invert_yaxis()
    for axis in axes:
        axis.axvline(0, color='gray', linewidth=.8)
        axis.grid(axis='x', alpha=.2)
    fig.suptitle('LAGA effect: alpha=0.5 minus alpha=0; dots: seeds, diamond: mean')
    paths = []
    for ext in ('png','svg'):
        path = OWNER/'results'/f'laga_effects.{ext}'
        fig.savefig(path, dpi=180)
        if ext == 'svg':
            path.write_text('\n'.join(l.rstrip() for l in path.read_text().splitlines())+'\n')
        paths.append(path)
    plt.close(fig)
    return paths


def report(payload):
    a, b, effect = payload['alpha0'], payload['alpha05'], payload['effect']
    lines = [
        '# DA / LGGD：LAGA 三种子补充消融', '',
        '**本轮没有复现“加入 LAGA 一致提升”的现象。** 在 LLLG+GATE 下，'
        f"α=0 的平均 AUC 为 {a['auc_percent']['mean']:.4f}%，高于 α=0.5 的 {b['auc_percent']['mean']:.4f}%；"
        '不带 LAGA 的平均 EER 也更低。加入 LAGA 后，AUC 三个种子一升两降，EER 三个种子均升高。', '',
        '## 设置与证据', '',
        '新增配置为 **HAP-RD w/o LAGA**（别名 d-new-ablation / LGGD without LAGA）。'
        'LLLG+gate ratio2、学生宽度768、PDFB hidden512、code_dim64、12 taps/12个独立头、无PODP，'
        '可训练参数30,248,000。与同seed的正式HAP-RD比较，仅将训练及match测试的α从0.5改为0。'
        '四组三层等权mean在α=0时等价于12层平均，Top-5 patch cosine计算不变。', '',
        '新增seed42/43/44、每seed七场景，共21个新训练任务，全部30 epoch完成、无失败。'
        '15:12:29启动、21:56:41结束（北京时间2026-11-19）。'
        '冻结DINOv3 ViT-B/16、256×256输入、batch64、AdamW+OneCycleLR、max_lr=5e-4、'
        'weight_decay=0.05等固定设置与基线一致。基线复用已完成的GL LLLG+gate结果，未重新训练或改写。', '',
        '分数越大越异常。场景内拼接全部测试视频后计算AUC×100及离散ROC EER，'
        '再对七场景等权宏平均，最后对三seed求均值和样本SD（ddof=1）。EER为0–1比例，越低越好；'
        '取`fpr[argmin(abs(1-tpr-fpr))]`，不插值。', '',
        '## 种子矩阵', '',
        '| seed | α=0 AUC | α=0.5 AUC | 加入LAGA ΔAUC | α=0 EER | α=0.5 EER | 加入LAGA ΔEER |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for s in map(str, SEEDS):
        lines.append(f"| {s} | {a['auc_percent']['by_seed'][s]:.4f} | {b['auc_percent']['by_seed'][s]:.4f} | "
                     f"{effect['auc_delta_pp']['by_seed'][s]:+.4f} | {a['eer']['by_seed'][s]:.6f} | "
                     f"{b['eer']['by_seed'][s]:.6f} | {effect['eer_delta']['by_seed'][s]:+.6f} |")
    lines.append(f"| 均值 ± SD | {a['auc_percent']['mean']:.4f} ± {a['auc_percent']['sample_sd']:.4f} | "
                 f"{b['auc_percent']['mean']:.4f} ± {b['auc_percent']['sample_sd']:.4f} | "
                 f"{effect['auc_delta_pp']['mean']:+.4f} ± {effect['auc_delta_pp']['sample_sd']:.4f} | "
                 f"{a['eer']['mean']:.6f} ± {a['eer']['sample_sd']:.6f} | "
                 f"{b['eer']['mean']:.6f} ± {b['eer']['sample_sd']:.6f} | "
                 f"{effect['eer_delta']['mean']:+.6f} ± {effect['eer_delta']['sample_sd']:.6f} |")
    lines += ['', '所有差值统一定义为 **α=0.5 − α=0**；AUC正值有利于LAGA，EER负值有利于LAGA。'
              'AUC差值单位为百分点。完整精度见 [comparison_matrix.csv](comparison_matrix.csv)。', '',
              '加入LAGA的平均AUC变化为−0.1682个百分点，平均EER变化为+0.003090。'
              'α=0.5的AUC跨seed样本SD较小（0.2803 vs 0.4714），但均值和EER不占优，'
              '不能单凭SD较小宣称整体更好。三seed不足以证明显著差异或α=0在总体上必然更好。', '',
              '## 场景分解', '',
              '| 场景 | α=0 AUC均值 | α=0.5 AUC均值 | 平均ΔAUC | seed42 Δ | seed43 Δ | seed44 Δ | 平均ΔEER |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for scene, r in payload['scenes'].items():
        lines.append(f"| {SCENES[scene]} | {r['alpha0_auc']['mean']:.4f} | {r['alpha05_auc']['mean']:.4f} | "
                     f"{r['auc_delta_pp']['mean']:+.4f} | "
                     + ' | '.join(f"{r['auc_delta_pp']['by_seed'][str(s)]:+.4f}" for s in SEEDS)
                     + f" | {r['eer_delta']['mean']:+.6f} |")
    lines += ['',
        'LAGA在Bike、Farmland、Railway的三个seed中均提高AUC，其中Railway平均提高1.0653个百分点；'
        '在Crossroads、Highway、Solar的三个seed中均降低AUC，平均分别下降1.4494、0.6763、0.6060。'
        'Vehicle平均提高0.2414，但seed44下降1.6207，表现并不一致。'
        '因此平均退化来自场景收益与损失的抵消，并不是LAGA对所有场景都无效。', '',
        'EER的最大不利变化来自Highway（+0.017650），随后是Solar（+0.009851）和Crossroads（+0.005340）；'
        'Farmland、Railway、Bike则改善。保留AUC和EER两项指标，避免只报告有利方向。', '',
        '## 与先前Loss及C→D结果并列', '',
        '| 模板 | α=0 AUC均值 | α=0.5 AUC均值 | 平均ΔAUC | AUC提升seed数 | 平均ΔEER |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for r in payload['template_context']:
        wins = sum(x>0 for x in r['auc_delta_pp']['by_seed'].values())
        lines.append(f"| {r['template']} | {r['alpha0_auc']:.4f} | {r['alpha05_auc']:.4f} | "
                     f"{r['auc_delta_pp']['mean']:+.4f} | {wins}/3 | {r['eer_delta']['mean']:+.6f} |")
    lines += ['',
        'GLLL+GATE来自先前Loss矩阵的Top-5/12层平均与Top-5/四组α=0.5；'
        'GGGG+GELU来自原publication的C/D。两组引用原已审计汇总，本轮没有重跑，也没有混用GL矩阵里其他seed43/44的V2结果。'
        '各来源路径和SHA-256见 [laga_comparison.json](laga_comparison.json)。', '',
        '**三组证据共同说明：目前不能假定LAGA收益跨decoder模板保持一致。** '
        '先前GLLL模板有正向平均增益，标准GGGG模板增益很小，本次LLLG模板则平均略降。'
        '这支持“收益可能依赖模板或其配对误差分布”的解释方向，但不是decoder×LAGA交互的显著性检验，'
        '也不足以把差异唯一归因于Global block的位置。历史复用、执行入口和物理GPU分配不同，'
        '同seed不是同一张卡上的逐位配对重训。', '',
        '从模型路由看，LLLG的Global位于最后stage，直接输出teacher0–2的三个头；'
        'GLLL的Global位于第一stage，其表示还会被后续Local blocks处理。'
        '组内最大误差加权对不同路由可能产生不同的优化偏好；这是机制假设，'
        '本轮没有记录每层/每patch误差轨迹，不能据此下因果结论。', '',
        '## 对论文表述的影响', '',
        '- 可将本实验作为HAP-RD（LLLG+GATE）下直接移除LAGA的证据，与原C/D并列；'
        '应完整报告混合与负向结果，不把原GLLL的正向增益直接移植为LLLG的结论。',
        '- α=0的平均AUC比现有α=0.5正式HAP-RD高0.1682个百分点，且EER在三个seed均更低；'
        '它是值得保留的候选配置，但本次只归档结果，不自动改变已确认正式模型的定义。',
        '- 原八配置publication矩阵保持原值；本实验是补充行，不覆盖原D或HAP-RD，也不冒充新的正式变体。',
        '- 训练目标和匹配测试读出随α一起改变。因此这是匹配的整体设计比较，不单独隔离训练损失效应。', '',
        '![LAGA场景差值与模板对比](laga_effects.png)', '',
        '图中每个圆点为一个seed，菱形为三seed均值；无显著性或置信区间含义。[SVG](laga_effects.svg)。', '',
        '## 完整性与复现', '',
        '[validation.json](validation.json)记录主比较的42份CSV重算AUC/EER、数据集帧顺序/标签核对、'
        '42份最终checkpoint的SHA-256和严格模型加载。21个新场景均有完整1–30 epoch日志，'
        'config、运行guard、checkpoint、固定source及preflight哈希一致。'
        '新训练源码为`c43f0d23360cce0b574eeeb3589ac793e00fb548`；基线保留各自原始训练来源。', '',
        '旧GL构造器与新ablation构造器在相同RNG状态下的初始参数逐张量相同也经CPU核验；'
        '这验证模型构造差异没有引入额外结构/初始化变化，不保证跨硬件训练逐位相同。'
        '初次preflight因缺少显式evaluation.workers字段失败的记录仍保留在preflight_attempt1.json；'
        '修正后再正式训练，没有训练失败被丢弃。', '',
        '可复现命令：`/home/lyx/miniconda3/envs/DMAD/bin/python scripts/summarize_da_laga.py`。'
        '仅CPU读取已有结果和权重，不训练、不重新推理或改写逐帧分数。权重仍是Git忽略的本地产物。', '',
    ]
    path = OWNER/'results/laga_comparison.md'
    path.write_text('\n'.join(lines))
    return path


def main():
    torch.set_num_threads(2)
    plan = read(OWNER/'run_plan.json')
    preflight = read(OWNER/'preflight.json')
    rid = plan['run_id']
    events = [json.loads(x) for x in (OWNER/'logs'/f'scheduler_{rid}.jsonl').read_text().splitlines()]
    final = events[-1]
    assert final['kind']=='all_finished' and final['completed']==21 and final['skipped']==0 and final['failures']==[]
    launches = {(x['experiment'],x['scene']):x for x in events if x['kind']=='launch'}
    finishes = {(x['experiment'],x['scene']):x for x in events if x['kind']=='finish'}
    assert len(launches)==len(finishes)==21
    assert preflight['status']=='passed'
    for rel,h in preflight['code_hashes'].items():
        assert sha(ROOT/rel)==h, rel
    expected_keys, fingerprints = {}, {}
    for scene,dirname in SCENES.items():
        scene_dir = Path('/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly')/dirname
        _,videos,fingerprints[scene] = data_manifest(scene_dir)
        expected_keys[scene] = []
        for video in videos:
            frames = sorted(video.glob('*.jpg'),key=lambda p:int(p.stem))
            labels = np.asarray(np.load(str(video)+'.npy',allow_pickle=True)).astype(int).ravel()
            expected_keys[scene].extend((str(video.relative_to(scene_dir)),p.name,int(y)) for p,y in zip(frames,labels))
    checks, data, initialization = [], {}, []
    for rel in plan['experiments']:
        exp = ROOT/rel
        cfg = checked_config(exp)
        seed = cfg['seed']
        assert read(exp/'record.json')['status']=='completed'
        # Verify the actual historical GL constructor, beyond the publication wrapper.
        torch.manual_seed(seed)
        new_bn,new_decoder = build_ablation_model()
        torch.manual_seed(seed)
        old_bn = build_bottleneck(n_taps=12,code_dim=64,fuse_hidden=512,use_tap_norm=True)
        old_decoder = build_structural_student(dim=768,code_dim=64,heads=8,ffn_ratio=2.,
                                               teacher_dim=768,block_order='LLLG',ffn_type='gate')
        for actual,reference in [(new_bn,old_bn),(new_decoder,old_decoder)]:
            assert actual.state_dict().keys()==reference.state_dict().keys()
            assert all(torch.equal(v,reference.state_dict()[k]) for k,v in actual.state_dict().items())
        initialization.append({'seed':seed,'legacy_gl_initial_state_equal':'passed'})
        del new_bn,new_decoder,old_bn,old_decoder
        for condition,directory in [('alpha0',exp),('alpha05',ROOT/cfg['baseline'])]:
            metrics = {}
            bn,decoder = build_ablation_model() if condition=='alpha0' else build_publication_model('hap_rd',seed)
            for scene in SCENES:
                m = read(directory/'results'/f'{scene}.json')
                assert m['scene']==scene and m['seed']==seed and m['epochs']==30
                assert m['split_fingerprint']==fingerprints[scene]
                score_path = directory/'results'/f'{scene}_scores.csv'
                with score_path.open() as f:
                    rows = list(csv.DictReader(f))
                keys = [(x['video'],x['frame'],int(x['label'])) for x in rows]
                assert keys==expected_keys[scene] and len(set((v,f) for v,f,_ in keys))==len(keys)
                y = np.array([k[2] for k in keys])
                scores = np.array([float(x['anomaly_score']) for x in rows])
                assert np.isfinite(scores).all()
                auc = float(roc_auc_score(y,scores)*100)
                fpr,tpr,_ = roc_curve(y,scores)
                eer = float(fpr[np.nanargmin(np.abs(1-tpr-fpr))])
                assert math.isclose(auc,m['auc_percent'],abs_tol=1e-9,rel_tol=0)
                assert math.isclose(eer,m['eer'],abs_tol=1e-12,rel_tol=0)
                assert len(y)==m['frames'] and int(y.sum())==m['positive_frames']
                checkpoint = directory/m['checkpoint']
                assert sha(checkpoint)==m['checkpoint_sha256']
                weight = torch.load(checkpoint,map_location='cpu',weights_only=False)
                load_historical_weights(bn,decoder,weight)
                guard = ablation_guard(bn,decoder) if condition=='alpha0' else architecture_guard(bn,decoder,'hap_rd')
                guard = json.loads(json.dumps(guard))
                if condition=='alpha0':
                    assert m['guard']==guard and json.loads(json.dumps(weight['guard']))==guard
                    assert m['code_hashes']==preflight['code_hashes'] and m['source_commit']==plan['source_commit']
                    for field in ('code_hashes','source_commit','seed','scene','run_id','match'):
                        assert weight[field]==m[field]
                    assert m['run_id']==rid and 'alpha0' in m['match'] and 'train=batch mean, test=frame score' in m['match']
                    assert finishes[rel,scene]['returncode']==0
                    lines = (ROOT/launches[rel,scene]['log']).read_text().splitlines()
                    epochs = [int(re.match(r'epoch=(\d+)',l)[1]) for l in lines if l.startswith('epoch=')]
                    assert epochs==list(range(1,31))
                    assert json.loads(lines[-1])==m
                    runtime_guard = [json.loads(l) for l in lines if l.startswith('{') and '"kind": "model_guard"' in l]
                    assert len(runtime_guard)==1 and runtime_guard[0]['guard']==guard
                    state = read(directory/'results'/f'{scene}_status.json')
                    assert state['status']=='completed' and state['epoch']==30
                else:
                    assert m['model_guard']['block_order']=='LLLG' and m['model_guard']['ffn_type']=='gate'
                    assert m['trainable_parameters']==30248000 and m['max_steps']==0
                    historical = read(directory/'provenance/source_config.json')
                    assert historical['fixed']['loss']=='group_balanced_top5_cosine_alpha0.5'
                    source = ROOT/read(directory/'config.json')['reused_from']
                    assert (directory/'results'/f'{scene}.json').read_bytes()==(source/'results'/f'{scene}.json').read_bytes()
                    assert score_path.read_bytes()==(source/'results'/score_path.name).read_bytes()
                checks.append({'experiment':str(directory.relative_to(ROOT)),'condition':condition,'seed':seed,'scene':scene,
                               'auc_percent':auc,'eer':eer,'frames':len(y),'scores_sha256':sha(score_path),
                               'checkpoint_sha256':m['checkpoint_sha256'],'guard':guard,
                               'checkpoint_hash_verified':True,'strict_load':'passed'})
                metrics[scene] = m
                del weight
            data[condition,seed] = {'auc_percent':st.mean(x['auc_percent'] for x in metrics.values()),
                                    'eer':st.mean(x['eer'] for x in metrics.values()),'scenes':metrics}
            del bn,decoder
            gc.collect()
        cached = read(exp/'results/summary.json')
        assert math.isclose(cached['macro_auc_percent'],data['alpha0',seed]['auc_percent'],abs_tol=1e-10)
        assert math.isclose(cached['macro_eer'],data['alpha0',seed]['eer'],abs_tol=1e-12)
        record = read(exp/'record.json')
        record['review_status']='passed_postrun_audit'
        record['review_report']='../20261119_laga_lggd_a0_s42/results/validation.json'
        save_json(exp/'record.json',record)
        print('audited new/baseline seed',seed,flush=True)
    payload = {'schema_version':1,'seeds':list(SEEDS),'scene_names':SCENES,
               'delta_definition':'alpha0.5 minus alpha0; AUC positive favors LAGA; EER negative favors LAGA',
               'run_source_commit':plan['source_commit'],'scene_pairs':21,'parameters_both':30248000}
    for condition in ('alpha0','alpha05'):
        payload[condition] = {metric:stats([data[condition,s][metric] for s in SEEDS]) for metric in ('auc_percent','eer')}
    payload['effect'] = {'auc_delta_pp':stats([data['alpha05',s]['auc_percent']-data['alpha0',s]['auc_percent'] for s in SEEDS]),
                         'eer_delta':stats([data['alpha05',s]['eer']-data['alpha0',s]['eer'] for s in SEEDS])}
    payload['scenes'] = {}
    for scene in SCENES:
        payload['scenes'][scene] = {
            'alpha0_auc':stats([data['alpha0',s]['scenes'][scene]['auc_percent'] for s in SEEDS]),
            'alpha05_auc':stats([data['alpha05',s]['scenes'][scene]['auc_percent'] for s in SEEDS]),
            'alpha0_eer':stats([data['alpha0',s]['scenes'][scene]['eer'] for s in SEEDS]),
            'alpha05_eer':stats([data['alpha05',s]['scenes'][scene]['eer'] for s in SEEDS]),
            'auc_delta_pp':stats([data['alpha05',s]['scenes'][scene]['auc_percent']-data['alpha0',s]['scenes'][scene]['auc_percent'] for s in SEEDS]),
            'eer_delta':stats([data['alpha05',s]['scenes'][scene]['eer']-data['alpha0',s]['scenes'][scene]['eer'] for s in SEEDS])}
    payload['template_context'] = historical_context(payload)
    save_json(OWNER/'results/laga_comparison.json',payload)
    save_json(OWNER/'results/validation.json',{'status':'passed','new_scenes':21,'baseline_scenes':21,'primary_scene_cells':42,
              'scheduler_final':final,'pinned_code_config_hashes_verified':True,
              'dataset_frame_order_and_labels_verified':True,'final_checkpoint_hashes_and_strict_loads':42,
              'legacy_constructor_checks':initialization,'checks':checks,
              'context_scope':'GGGG/GLLL context quotes previously audited summary JSON; not re-evaluated in this audit'})
    matrix = [{'seed':s,**{f'{c}_{metric}':data[c,s][metric] for c in ('alpha0','alpha05') for metric in ('auc_percent','eer')},
               'laga_auc_delta_pp':payload['effect']['auc_delta_pp']['by_seed'][str(s)],
               'laga_eer_delta':payload['effect']['eer_delta']['by_seed'][str(s)]} for s in SEEDS]
    write_csv(OWNER/'results/comparison_matrix.csv',matrix)
    write_csv(OWNER/'results/scene_comparison.csv',[
        {'scene':scene,'seed':s,'alpha0_auc':data['alpha0',s]['scenes'][scene]['auc_percent'],
         'alpha05_auc':data['alpha05',s]['scenes'][scene]['auc_percent'],
         'laga_auc_delta_pp':payload['scenes'][scene]['auc_delta_pp']['by_seed'][str(s)],
         'alpha0_eer':data['alpha0',s]['scenes'][scene]['eer'],
         'alpha05_eer':data['alpha05',s]['scenes'][scene]['eer'],
         'laga_eer_delta':payload['scenes'][scene]['eer_delta']['by_seed'][str(s)]} for scene in SCENES for s in SEEDS])
    write_csv(OWNER/'results/template_context.csv',[
        {'template':r['template'],'alpha0_auc':r['alpha0_auc'],'alpha05_auc':r['alpha05_auc'],
         'laga_auc_delta_mean_pp':r['auc_delta_pp']['mean'],'paired_delta_sample_sd':r['auc_delta_pp']['sample_sd'],
         'laga_eer_delta_mean':r['eer_delta']['mean'],
         **{f'seed{s}_laga_auc_delta_pp':r['auc_delta_pp']['by_seed'][str(s)] for s in SEEDS}}
        for r in payload['template_context']])
    plots = plot(payload)
    report_path = report(payload)
    generated = [OWNER/'results'/name for name in ('laga_comparison.json','validation.json','comparison_matrix.csv',
                  'scene_comparison.csv','template_context.csv')]+plots+[report_path]
    inputs = [str(Path(rel)/'results/summary.json') for rel in plan['experiments']]
    register(OWNER,generated,rid,inputs+['ablation/da/publication/summary.json','ablation/da/Loss/summary.json'])
    for rel in plan['experiments']:
        exp = ROOT/rel
        local = exp/'results/laga_review.md'
        local.write_text(f'# LAGA comparison / seed{read(exp/"config.json")["seed"]}\n\n'
                         '[三种子比较、场景分解及完整核验](../../20261119_laga_lggd_a0_s42/results/laga_comparison.md)。\n')
        register(exp,[local],rid,['config.json',str(report_path.relative_to(ROOT))])
    print('PASSED: 42 CSV metric/label audits and checkpoint hash/strict-load checks; comparison and figures written.',flush=True)


if __name__=='__main__':
    main()
