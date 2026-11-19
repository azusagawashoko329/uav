#!/usr/bin/env python3
"""Audit saved DA publication results on CPU and regenerate summary tables/figures.

Does not train, rerun inference, modify scene results, or change source provenance.
"""
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
from model.publication_rd import (
    SPECS, architecture_guard, build_publication_model, load_historical_weights,
)
from scripts.da_publication_scene import SCENES, config, data_manifest, save_json, sha

BASE = ROOT / 'ablation/da/publication'
SEEDS = (42, 43, 44)
ORDER = tuple(SPECS)
PRODUCER = 'scripts/summarize_da_publication.py'


def stats(values):
    return {'mean': st.mean(values), 'sample_sd': st.stdev(values),
            'by_seed': dict(zip(map(str, SEEDS), values))}


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def register(directory, paths, run_id, inputs, producer=PRODUCER):
    """Update only registered/new small evidence; preserve weight provenance."""
    index = directory / 'artifacts.json'
    payload = json.loads(index.read_text())
    items = {item['path']: item for item in payload['artifacts']}
    for path in paths:
        rel = str(path.relative_to(directory))
        old = items.get(rel, {})
        items[rel] = {**old, 'path': rel, 'producer_script': producer,
                      'run_id': run_id, 'inputs': inputs, 'status': 'present',
                      'size_bytes': path.stat().st_size, 'sha256': sha(path), 'storage': 'git'}
    payload['artifacts'] = list(items.values())
    save_json(index, payload)


def figures(matrix, contrasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    for row in matrix:
        formal = row['id'] in ('pt_rd', 'hap_rd', 'hap_rd_lite')
        axs[0].errorbar(row['parameters']/1e6, row['auc_percent']['mean'],
                        yerr=row['auc_percent']['sample_sd'], fmt='o', capsize=3,
                        color='#b7472a' if formal else '#286a96')
        offset = (5, -15) if row['id'] in ('f2', 'c_ablation') else (5, 7)
        axs[0].annotate(row['label'], (row['parameters']/1e6, row['auc_percent']['mean']),
                        xytext=offset, textcoords='offset points', fontsize=8)
    axs[0].set(xscale='log', xlabel='Trainable parameters (M; log scale)',
               ylabel='DA macro AUC (%)', title='Mean ± sample SD across three seeds')
    axs[0].set_xlim(3, 125)
    axs[0].grid(alpha=.2)
    labels = []
    for i, row in enumerate(contrasts[:7]):
        values = list(row['auc_delta_pp']['by_seed'].values())
        axs[1].scatter(values, [i]*3, color='#286a96', s=20, alpha=.7)
        axs[1].scatter(row['auc_delta_pp']['mean'], i, marker='D', color='#b7472a', s=30)
        labels.append(f"{SPECS[row['from']]['label']} → {SPECS[row['to']]['label']}")
    axs[1].axvline(0, color='gray', linewidth=.8)
    axs[1].set(yticks=range(7), yticklabels=labels, xlabel='Paired AUC change (percentage points)',
               title='Dots: seeds 42/43/44; diamond: mean')
    axs[1].invert_yaxis()
    axs[1].grid(axis='x', alpha=.2)
    for ext in ('png', 'svg'):
        path = BASE / 'results' / f'publication_tradeoffs.{ext}'
        fig.savefig(path, dpi=180)
        if ext == 'svg':
            # Matplotlib emits insignificant spaces at the ends of path lines.
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def write_report(payload):
    matrix, contrasts = payload['matrix'], payload['paired_contrasts']
    rows = {r['id']: r for r in matrix}
    pairs = {(r['from'], r['to']): r for r in contrasts}
    lines = [
        '# DA publication：三种子完整消融链汇总', '',
        '24 个实验（8 种配置 × seed42/43/44）已完成，包含 168 个独立场景模型的结果。'
        '本轮从头训练 17 个实验、119 个场景任务，复制复用 7 个实验、49 个场景结果；调度器记录无失败。'
        '训练于北京时间 2026-11-19 01:35:50 全部结束。', '',
        '**主要观察：HAP-RD 的三种子平均 AUC 最高；PODP 在 F2 上带来三个种子一致的 AUC 提升，'
        '同时进一步减少参数。LAGA 在标准 Transformer 模板上的 AUC 增益很小且 EER 升高，'
        '本轮证据不支持把整条链写成每步都稳定提升。**', '',
        '## 统计与评分口径', '',
        '- 每个场景拼接全部测试视频，计算 AUC×100 和离散 ROC EER；先对七场景等权宏平均，'
        '再对三个种子求均值及样本标准差（ddof=1）。不是把全部场景帧混在一起计算一项 AUC。',
        '- AUC 单位为百分数，差值为百分点；EER 保留 0–1 比例，越低越好。'
        'EER 取 `fpr[argmin(abs(1-tpr-fpr))]`，不作插值。',
        '- 所有模型采用逐 patch cosine distance 的 Top-5。四 tap 模型取四层均值；'
        '十二 tap 模型按四组三层计算 `(1−α)mean+α max` 再平均。'
        '训练取同一 match 的 batch mean，测试使用逐帧 match 分数，无分数翻转、重新归一化或择优读出。',
        '- 统一 DA、冻结 DINOv3 ViT-B/16、256×256 输入（256 patches）、30 epoch、batch64、'
        'AdamW + OneCycleLR、max_lr=5e-4、weight_decay=0.05、code_dim64、tap_norm=True。',
        '- 参数量仅计学生解码器和瓶颈的可训练参数，冻结教师不计入。', '',
        '## 总体矩阵', '',
        '| 配置 | 参数 M | seed42 AUC | seed43 AUC | seed44 AUC | AUC 均值 ± SD | EER 均值 ± SD |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in matrix:
        a, e = r['auc_percent'], r['eer']
        lines.append(f"| {r['label']} | {r['parameters']/1e6:.3f} | "
                     + ' | '.join(f"{a['by_seed'][str(s)]:.4f}" for s in SEEDS)
                     + f" | {a['mean']:.4f} ± {a['sample_sd']:.4f} | {e['mean']:.6f} ± {e['sample_sd']:.6f} |")
    lines += ['', '完整精度及各 seed EER 见 [matrix.csv](results/matrix.csv)；'
              '168 项逐场景结果见 [scene_metrics.csv](results/scene_metrics.csv)。', '',
              '## 相邻步骤的种子配对差值', '',
              '差值均为“后者 − 前者”；AUC 正值更好，EER 负值更好。'
              '同 seed 仅用于配对索引，不表示不同结构具有相同初始化张量。', '',
              '| 步骤 | seed42 ΔAUC | seed43 ΔAUC | seed44 ΔAUC | 平均 ΔAUC ± SD | AUC 提升种子数 | 平均 ΔEER | 参数减少 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in contrasts[:7]:
        a = r['auc_delta_pp']
        lines.append(f"| {rows[r['from']]['label']} → {rows[r['to']]['label']} | "
                     + ' | '.join(f"{a['by_seed'][str(s)]:+.4f}" for s in SEEDS)
                     + f" | {a['mean']:+.4f} ± {a['sample_sd']:.4f} | {r['auc_improved_seeds']}/3 | "
                     + f"{r['eer_delta']['mean']:+.6f} | {r['parameter_reduction_percent']:+.2f}% |")
    lines += ['', '“参数减少”为负表示该步骤参数增加。'
              '可复核数据见 [paired_contrasts.csv](results/paired_contrasts.csv)。', '',
              '## 按场景分析', '',
              '下表为每场景三个种子的 AUC 均值；逐场景 SD、EER、配对差值保存在 [summary.json](summary.json)。', '',
              '| 场景 | PT-RD | B | C | D | HAP-RD | F1 | F2 | Lite |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in SCENES:
        lines.append(f'| {SCENES[c]} | ' + ' | '.join(f"{r['scene_auc_percent'][c]['mean']:.3f}" for r in matrix) + ' |')
    lines += ['',
        'Bike 与 Farmland 在所有配置下已接近高 AUC 区间，压缩的影响主要体现在 Railway、Solar、'
        'Vehicle 和 Crossroads。HAP-RD 相对 PT-RD 在 Solar、Vehicle 分别提高 3.533、3.757 个百分点，'
        '但 Crossroads、Railway 分别降低 0.977、1.288 个百分点；宏平均提升并不代表每个场景都改善。', '',
        'Lite 相对 HAP-RD 的主要退化来自 Railway（−3.963）、Vehicle（−3.243）和 Solar（−2.953），'
        'Bike、Farmland、Highway 则略有提高。F2→Lite 的最大恢复来自 Vehicle（+1.988），'
        '其余 Railway、Solar、Crossroads 分别恢复 0.473、0.400、0.388 个百分点；七场景中六个平均 AUC 提升，'
        'Bike 略降 0.010 个百分点。', '',
        '## 逐步消融解释', '',
        '### PT-RD → B：删除未直接蒸馏的学生 block', '',
        '保持四个 teacher taps `[5,7,9,11]`、四个重建头、768 宽度和 hidden512，只将 12 个标准 '
        'Transformer blocks 减为 4 个，并从头训练。AUC 均值由 84.8582 变为 84.7733（−0.0848），'
        '三个种子的差值一正两负，EER 均值由 0.200161 降到 0.198491。参数量从 89.079M 降为 32.376M。'
        '结果支持删去这些 block 可以显著压缩模型且观测到的平均 AUC 损失较小；'
        '三个种子不足以证明统计等效或这些 block 在任何情况下都无用。', '',
        '### B → C：扩展深度覆盖与一对三监督', '',
        '扩为 12 个 teacher taps 和 12 个独立重建头，四个学生 stage 各有三个头分别预测三层教师；'
        'α=0 等价于十二层直接平均。平均 AUC 增加 0.5043，EER 降低 0.006626，两个种子提升，'
        'seed42 略降 0.0141。Highway、Vehicle、Solar 分别提高 2.853、2.423、2.172，'
        'Railway 和 Crossroads 则降低 2.421、1.025 个百分点。'
        '该步骤同时扩展瓶颈输入（3072→9216）和头数，参数增至 40.259M；'
        '应描述为更完整的深度覆盖与分组配对设计，不能把全部增益归于“只增加重建头”。', '',
        '### C → D：引入 LAGA，α=0 → 0.5', '',
        '结构与参数量不变，只在训练损失和匹配测试评分中引入组内 max 项。'
        '平均 AUC 仅增加 0.0482，配对差值 SD 为 0.7218；三个种子的变化分别为 +0.7165、−0.7172、+0.1453。'
        'EER 均值反而由 0.191865 升至 0.197044。C 的 EER 均值是本轮八种配置中最低的。'
        '**本轮不能宣称 LAGA 在标准 Transformer 模板上稳定改善检测。** '
        '它在 Railway、Solar 上的收益被 Highway 等场景的退化抵消；'
        '若引用先前 GLLL 模板的 Loss 消融，必须明确模板不同，不能替代本轮 C/D 证据。', '',
        '### D → HAP-RD：引入 LLLG + gated decoder', '',
        '将 GGGG + GELU MLP 换为 LLLG + gated FFN。平均 AUC 增加 0.3294，EER 降低 0.004156；'
        '三个种子两升一降，seed44 降低 0.0662。Vehicle 提高 2.953 个百分点，'
        '但 Highway 降低 0.792。参数从 40.259M 减至 30.248M。'
        '该步支持整套 LGGD 在本轮平均效果与参数规模上的优势，'
        '但同时改变 mixer 和 FFN，不能据此单独归因于 local mixing、LLLG 顺序或 gate；'
        '这些因素应引用已有 GL 矩阵。', '',
        '### HAP-RD → F1 → F2：分两步定位压缩代价', '',
        '先将学生宽度 768→192，参数降至 7.718M，平均 AUC 损失 1.2964；'
        '再将 PDFB hidden512→128，参数降至 4.154M，平均 AUC 再损失 0.7876。'
        '两步各自均在三个种子上降低 AUC，EER 也分别增加 0.004308、0.004748。'
        '宽度压缩的损失更大，但两种压缩都产生可观察代价；'
        'F1→F2 的七个场景平均 AUC 均降低，其中 Vehicle 降低 2.738 个百分点最明显。', '',
        '### F2 → HAP-RD-Lite：加入 PODP', '',
        '保持 width192、hidden128、LLLG、12 个独立头和 PLDD，加入四组共享的可学习 2×3 Q，'
        '随机正交初始化并在前向保持行正交。平均 AUC 增加 0.5042，三个种子分别提高 '
        '0.3121、0.9508、0.2497；EER 均值降低 0.002371，AUC 样本 SD 从 0.4940 降至 0.1740。'
        '这为 PODP 在该轻量配置上的收益提供了方向一致的三种子证据。'
        'Q 只有 6 个可学习系数，但把瓶颈输入从 12×768 压为 8×768，'
        '净参数量进一步减少 393,210，降至 3.761M。'
        '该比较体现整套 PODP 的作用，不能分开证明收益来自可学习性、正交约束或降秩中的哪一项。', '',
        '## 正式模型的效果与规模取舍', '',
    ]
    for a, b in [('pt_rd', 'hap_rd'), ('pt_rd', 'hap_rd_lite'), ('hap_rd', 'hap_rd_lite')]:
        p = pairs[a, b]
        lines.append(f"- {rows[a]['label']} → {rows[b]['label']}：参数减少 "
                     f"{p['parameter_reduction_percent']:.2f}%，平均 AUC 变化 "
                     f"{p['auc_delta_pp']['mean']:+.4f} 个百分点，EER 变化 {p['eer_delta']['mean']:+.6f}。")
    lines += ['',
        'HAP-RD 相对 PT-RD 的三个种子 AUC 均提高，是本轮完整方法的主要正向证据。'
        'Lite 相对 PT-RD 的三个种子 AUC 均降低，平均少 0.7827 个百分点；'
        '应明确其是大幅减少可训练参数后的效果取舍。'
        'Lite 的 EER 均值略低于 PT-RD，但差距仅 0.000587，不能表述成检测性能全面超过参考模型。'
        '参数减少也不等于推理延迟或显存按同比例减少，本轮没有测量端到端速度与 FLOPs。', '',
        '![参数与 AUC、各步配对差值](results/publication_tradeoffs.png)', '',
        '图的左侧是三种子均值及样本 SD，右侧是同 seed 的 AUC 差值与其均值；'
        '误差条不是置信区间。[可编辑 SVG](results/publication_tradeoffs.svg)。', '',
        '## 完整性、复用及限制', '',
        '- [核验报告](results/validation.json)：168 份逐帧 CSV 的 AUC/EER 全部重算，'
        '逐视频、逐帧顺序及标签与数据集核对；168 份最终权重校验 SHA-256 并在 CPU 严格加载到目标模型。',
        '- 119 个新场景的配置、模型 guard、训练日志、checkpoint、source_commit 和运行前代码哈希一致；'
        '各有完整 1–30 epoch 日志。所有代码与配置保持运行前哈希。复用部分保留原始指标、评分及 provenance，'
        '没有以本次提交冒充原训练源码。',
        '- PT-RD seed42 复用旧 V0；D 三种子复用 GGGG+GELU / V1 等价结果；'
        'HAP-RD 三种子复用 GL 的 LLLG+gate 结果。复用的 49 份 JSON/CSV 与原目录逐字节相同；'
        '运行前已核验新旧模型前向等价，详见 [preflight](publication_preflight.json)。',
        '- 本轮新训练固定源码为 `aa0f057d834efc9ec30ba1a120b93d9d42aa7028`。'
        '模型入口为 `model/publication_rd.py`，新训练入口为 `scripts/da_publication_scene.py`；'
        '复用实验的原训练提交以各自 record/provenance 为准。',
        '- 三个种子只支持描述性比较，不作显著性或统计等效断言；七个不同场景不等于同一条件下的七次重复实验。'
        '本轮只覆盖 DA，不能外推 UIT。',
        '- B/C/D/F1/F2 保留为[过程结构版本](ARCHITECTURE_VERSIONS.md)，不新增正式模型名，'
        '不覆盖旧 V0–V5；F1/F2 为 LLLG，不能直接当作旧 GLLL 的 V3/V4。',
        '- 复现汇总：`/home/lyx/miniconda3/envs/DMAD/bin/python scripts/summarize_da_publication.py`。'
        '该脚本只读取已保存分数、CPU 核验权重并重建报告，不启动训练或 GPU 推理。',
        '- Git 保存代码、小型结果、日志、报告和产物索引；权重留在本地并由 Git 忽略，'
        '本次 Git 同步不构成权重云端备份。', '',
    ]
    (BASE/'summary.md').write_text('\n'.join(lines))


def main():
    torch.set_num_threads(2)
    (BASE / 'results').mkdir(exist_ok=True)
    plan = json.loads((BASE/'publication_chain_plan.json').read_text())
    manifest = json.loads((BASE/'publication_run.json').read_text())
    preflight = json.loads((BASE/'publication_preflight.json').read_text())
    rid = manifest['run_id']
    events = [json.loads(x) for x in (BASE/f'{rid}.scheduler.jsonl').read_text().splitlines()]
    final = events[-1]
    assert final['kind'] == 'all_finished' and final['completed'] == 119
    assert final['skipped'] == 49 and final['failures'] == []
    launches = {(e['experiment'], e['scene']): e for e in events if e['kind'] == 'launch'}
    finishes = {(e['experiment'], e['scene']): e for e in events if e['kind'] == 'finish'}
    assert len(launches) == len(finishes) == 119
    assert preflight['status'] == 'passed'
    for path, digest in preflight['code_hashes'].items():
        assert sha(ROOT/path) == digest, path

    expected_keys, fingerprints = {}, {}
    for scene, dirname in SCENES.items():
        scene_dir = Path('/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly') / dirname
        _, videos, fingerprints[scene] = data_manifest(scene_dir)
        keys = []
        for video in videos:
            frames = sorted(video.glob('*.jpg'), key=lambda p: int(p.stem))
            labels = np.load(str(video)+'.npy', allow_pickle=True).astype(int).ravel()
            keys.extend((str(video.relative_to(scene_dir)), p.name, int(y)) for p, y in zip(frames, labels))
        expected_keys[scene] = keys

    data, checks, scene_rows = {}, [], []
    for entry in plan['entries']:
        exp = ROOT/entry['path']
        cfg = config(exp)
        name, seed = cfg['variant'], cfg['seed']
        record = json.loads((exp/'record.json').read_text())
        assert record['status'] == 'completed' and cfg['model'] == SPECS[name]
        bn, dec = build_publication_model(name, seed)
        metrics = {}
        for scene in SCENES:
            mp = exp/'results'/f'{scene}.json'
            sp = exp/'results'/f'{scene}_scores.csv'
            m = json.loads(mp.read_text())
            assert m['scene'] == scene and m['seed'] == seed and m['epochs'] == 30
            assert m['split_fingerprint'] == fingerprints[scene]
            with sp.open() as f:
                rows = list(csv.DictReader(f))
            keys = [(r['video'], r['frame'], int(r['label'])) for r in rows]
            assert keys == expected_keys[scene] and len(set((v, f) for v, f, _ in keys)) == len(keys)
            y = np.array([r[2] for r in keys])
            scores = np.array([float(r['anomaly_score']) for r in rows])
            assert np.isfinite(scores).all() and set(y) == {0, 1}
            auc = float(roc_auc_score(y, scores)*100)
            fpr, tpr, _ = roc_curve(y, scores)
            eer = float(fpr[np.nanargmin(np.abs(1-tpr-fpr))])
            assert math.isclose(auc, m['auc_percent'], abs_tol=1e-9, rel_tol=0)
            assert math.isclose(eer, m['eer'], abs_tol=1e-12, rel_tol=0)
            assert len(y) == m['frames'] and int(y.sum()) == m['positive_frames']
            ck = exp/m['checkpoint']
            assert sha(ck) == m['checkpoint_sha256'], str(ck)
            payload = torch.load(ck, map_location='cpu', weights_only=False)
            load_historical_weights(bn, dec, payload)
            guard = architecture_guard(bn, dec, name)
            # JSON roundtrip normalizes routing tuples to arrays.
            guard = json.loads(json.dumps(guard))
            if entry['reused_from']:
                source = ROOT/entry['reused_from']
                assert mp.read_bytes() == (source/'results'/mp.name).read_bytes()
                assert sp.read_bytes() == (source/'results'/sp.name).read_bytes()
                assert m['trainable_parameters'] == SPECS[name]['params']
                if 'scoring' in m:
                    assert m['scoring'] == 'matched_group_balanced_top5'
                else:
                    historical = json.loads((exp/'provenance/source_config.json').read_text())
                    assert historical['fixed']['loss'] == 'group_balanced_top5_cosine_alpha0.5'
                    assert m['model_guard']['block_order'] == SPECS[name]['order']
                    assert m['model_guard']['ffn_type'] == SPECS[name]['ffn']
                if name == 'hap_rd':
                    assert m['model_guard']['block_order'] == 'LLLG'
                evidence = 'unchanged historical matched scores; strict equivalent-model load; preflight forward equivalence'
            else:
                assert m['guard'] == guard and json.loads(json.dumps(payload['guard'])) == guard
                for field in ('source_commit', 'run_id', 'variant', 'seed', 'scene', 'code_hashes', 'match'):
                    assert payload[field] == m[field], field
                assert m['source_commit'] == manifest['source_commit'] == record['source_commit']
                assert m['code_hashes'] == preflight['code_hashes']
                assert m['run_id'] == rid and m['match'] == 'publication_match train=batch mean; test=per-frame score'
                assert finishes[entry['path'], scene]['returncode'] == 0
                lines = (ROOT/launches[entry['path'], scene]['log']).read_text().splitlines()
                epochs = [int(re.match(r'epoch=(\d+)', line)[1]) for line in lines if line.startswith('epoch=')]
                assert epochs == list(range(1, 31))
                guards = [json.loads(line) for line in lines if line.startswith('{') and '"kind": "model_guard"' in line]
                assert len(guards) == 1 and guards[0]['guard'] == guard
                assert json.loads(lines[-1]) == m
                state = json.loads((exp/'results'/f'{scene}_status.json').read_text())
                assert state['status'] == 'completed' and state['epoch'] == 30
                evidence = 'same publication_match for train/test; config/log/result/checkpoint guard and source hashes agree'
            checks.append({'experiment': exp.name, 'scene': scene, 'reused_from': entry['reused_from'],
                           'auc_percent': auc, 'eer': eer, 'frames': len(y),
                           'checkpoint_sha256': m['checkpoint_sha256'], 'checkpoint_hash_verified': True,
                           'checkpoint_strict_load': 'passed', 'guard': guard, 'match_evidence': evidence})
            scene_rows.append({'variant': name, 'seed': seed, 'scene': scene, 'auc_percent': auc,
                               'eer': eer, 'frames': len(y), 'reused': bool(entry['reused_from'])})
            metrics[scene] = m
            del payload
        data[name, seed] = {'experiment': exp.name, 'scenes': metrics,
                            'auc': st.mean(m['auc_percent'] for m in metrics.values()),
                            'eer': st.mean(m['eer'] for m in metrics.values())}
        # Preserve the scheduler/copy summary and its original producer.
        existing = json.loads((exp/'results/summary.json').read_text())
        assert math.isclose(existing['macro_auc_percent'], data[name, seed]['auc'], abs_tol=1e-10)
        assert math.isclose(existing['macro_eer'], data[name, seed]['eer'], abs_tol=1e-12)
        record['review_status'] = 'passed_postrun_audit'
        record['review_report'] = '../results/validation.json'
        save_json(exp/'record.json', record)
        log = exp/'logs'/f'scheduler_{rid}.jsonl'
        assert log.read_bytes() == (BASE/f'{rid}.scheduler.jsonl').read_bytes()
        register(exp, [log], rid, ['config.json'], producer='scripts/da_publication_scheduler.py')
        del bn, dec
        gc.collect()
        print('validated', exp.name, flush=True)

    matrix = []
    for name in ORDER:
        matrix.append({'id': name, 'label': SPECS[name]['label'], 'parameters': SPECS[name]['params'],
                       'model': SPECS[name],
                       'auc_percent': stats([data[name, s]['auc'] for s in SEEDS]),
                       'eer': stats([data[name, s]['eer'] for s in SEEDS]),
                       'scene_auc_percent': {c: stats([data[name, s]['scenes'][c]['auc_percent'] for s in SEEDS]) for c in SCENES},
                       'scene_eer': {c: stats([data[name, s]['scenes'][c]['eer'] for s in SEEDS]) for c in SCENES}})
    pairs = list(zip(ORDER, ORDER[1:])) + [('pt_rd', 'hap_rd'), ('pt_rd', 'hap_rd_lite'), ('hap_rd', 'hap_rd_lite')]
    contrasts = []
    for a, b in pairs:
        values = [data[b, s]['auc']-data[a, s]['auc'] for s in SEEDS]
        contrasts.append({'from': a, 'to': b, 'auc_delta_pp': stats(values),
                          'auc_improved_seeds': sum(x > 0 for x in values),
                          'eer_delta': stats([data[b, s]['eer']-data[a, s]['eer'] for s in SEEDS]),
                          'parameter_change': SPECS[b]['params']-SPECS[a]['params'],
                          'parameter_reduction_percent': 100*(1-SPECS[b]['params']/SPECS[a]['params']),
                          'scene_auc_delta_pp': {c: stats([data[b, s]['scenes'][c]['auc_percent']-data[a, s]['scenes'][c]['auc_percent'] for s in SEEDS]) for c in SCENES}})
    save_json(BASE/'summary.json', {'schema_version': 1, 'seeds': SEEDS, 'scenes': list(SCENES),
                                  'matrix': matrix, 'paired_contrasts': contrasts,
                                  'aggregation': 'scene-concatenated AUC x100 and discrete ROC EER; unweighted seven-scene macro per seed; three-seed mean and sample SD (ddof=1)',
                                  'score_direction': 'larger=more anomalous; training-matched readout; no reversal or renormalization',
                                  'run_source_commit': manifest['source_commit'],
                                  'limitations': ['Only three seeds, descriptive SD without significance claims.',
                                                  'Same seed is an index pairing, not shared initialization across architectures.',
                                                  'C to D changes both training loss and matched test readout.',
                                                  'B to C changes teacher coverage, bottleneck input and head count together.',
                                                  'D to HAP changes both mixing and FFN.',
                                                  'Trainable parameter counts exclude the shared frozen teacher; no latency/FLOP benchmark.']})
    save_json(BASE/'results/validation.json', {'status': 'passed', 'experiments': 24, 'scene_cells': 168,
                                            'new_scene_jobs': 119, 'reused_scene_cells': 49, 'scheduler_final': final,
                                            'source_hashes_verified': True, 'checkpoint_hashes_verified': True,
                                            'all_checkpoints_strict_loaded_on_cpu': True, 'test_labels_verified_against_dataset': True,
                                            'metric_method': 'AUC x100; EER = fpr[argmin(abs(1-tpr-fpr))], no interpolation',
                                            'checks': checks})
    rows = []
    for row in matrix:
        r = {'variant': row['id'], 'parameters': row['parameters']}
        for metric, key in [('auc', 'auc_percent'), ('eer', 'eer')]:
            r.update({f'seed{s}_{metric}': row[key]['by_seed'][str(s)] for s in SEEDS})
            r[f'mean_{metric}'], r[f'sample_sd_{metric}'] = row[key]['mean'], row[key]['sample_sd']
        rows.append(r)
    write_csv(BASE/'results/matrix.csv', rows)
    write_csv(BASE/'results/scene_metrics.csv', scene_rows)
    write_csv(BASE/'results/paired_contrasts.csv', [
        {'from': r['from'], 'to': r['to'],
         **{f'seed{s}_auc_delta_pp': r['auc_delta_pp']['by_seed'][str(s)] for s in SEEDS},
         'mean_auc_delta_pp': r['auc_delta_pp']['mean'], 'sample_sd_auc_delta_pp': r['auc_delta_pp']['sample_sd'],
         'improved_seeds': r['auc_improved_seeds'], 'mean_eer_delta': r['eer_delta']['mean'],
         'parameter_reduction_percent': r['parameter_reduction_percent']} for r in contrasts])
    figures(matrix, contrasts)
    write_report(json.loads((BASE/'summary.json').read_text()))
    paths = [BASE/'summary.json', BASE/'summary.md', *sorted((BASE/'results').glob('*'))]
    register(BASE, [p for p in paths if p.is_file()], rid,
             [str((ROOT/e['path']/'results/summary.json').relative_to(ROOT)) for e in plan['entries']])
    register(BASE, [BASE/f'{rid}.scheduler.jsonl'], rid, ['publication_run.json'],
             producer='scripts/da_publication_scheduler.py')
    print('PASSED: 24 experiments, 168 CSV metrics/label sequences/checkpoint hashes/strict loads; wrote matrix, contrasts and figures.', flush=True)


if __name__ == '__main__':
    main()
