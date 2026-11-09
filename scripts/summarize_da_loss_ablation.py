#!/usr/bin/env python3
"""Audit and summarize the DA Loss matrix; no training or GPU evaluation is run."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import statistics as st
import subprocess
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'ablation/da/Loss'
SEEDS = (42, 43, 44)
SCENES = ('bike', 'crossroads', 'farmland', 'highway', 'railway', 'solar', 'vehicle')
SPECS = {
    'flatten_12mean': ('flatten', 'mean12', None, None),
    'patchmean_12mean': ('patchmean', 'mean12', None, None),
    'top5_12mean': ('topk', 'mean12', 5, None),
    'top5_group_a05': ('topk', 'group', 5, .5),
    'top5_group_a10': ('topk', 'group', 5, 1.),
    'top1_group_a05': ('topk', 'group', 1, .5),
    'top10_group_a05': ('topk', 'group', 10, .5),
    'patchmean_group_a05': ('patchmean', 'group', None, .5),
}
RUN = '20261107T150000Z_da_loss_multiseed_s42_s43_s44'
SOURCE = subprocess.check_output(['git', 'rev-parse', '141ff4f'], cwd=ROOT, text=True).strip()
LABELS = {
    'flatten_12mean': 'Flatten / 12层平均',
    'patchmean_12mean': 'Patch Mean / 12层平均',
    'top5_12mean': 'Top-5 / 12层平均',
    'top5_group_a05': 'Top-5 / 4组 α=0.5（V2）',
    'top5_group_a10': 'Top-5 / 4组 α=1',
    'top1_group_a05': 'Top-1 / 4组 α=0.5',
    'top10_group_a05': 'Top-10 / 4组 α=0.5',
    'patchmean_group_a05': 'Patch Mean / 4组 α=0.5',
}


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stats(values):
    return {'mean': st.mean(values), 'sample_sd': st.stdev(values),
            'sample_variance': st.variance(values), 'values': dict(zip(map(str, SEEDS), values))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify-checkpoints', action='store_true')
    args = ap.parse_args()
    ev = [json.loads(x) for x in (BASE / f'{RUN}.stdout.log').read_text().splitlines()]
    assert ev[-1]['kind'] == 'all_finished' and ev[-1]['completed'] == 161
    assert ev[-1]['skipped'] == 7 and ev[-1]['failures'] == []
    launches = {(x['experiment_id'], x['scene']): x for x in ev if x['kind'] == 'launch'}
    finishes = {(x['experiment_id'], x['scene']): x for x in ev if x['kind'] == 'finish'}
    assert len(launches) == len(finishes) == 161
    data, checks, reference_keys, fingerprints = {}, [], {}, {}
    for key, (error, agg, k, alpha) in SPECS.items():
        for seed in SEEDS:
            exp = BASE / f'20261107_{key}_s{seed}'
            cfg = json.loads((exp / 'config.json').read_text())
            lc = cfg['loss_ablation']
            assert cfg['seed'] == seed and lc['error_mode'] == error and lc['aggregation'] == agg
            assert k is None or lc['k'] == k
            assert alpha is None or lc['alpha'] == alpha
            assert cfg['ablation']['block_order'] == 'GLLL' and cfg['ablation']['ffn_type'] == 'gate'
            assert cfg['fixed']['teacher_taps'] == list(range(12))
            metrics = {}
            for scene in SCENES:
                m = json.loads((exp / 'results' / f'{scene}.json').read_text())
                assert m['seed'] == seed and m['epochs'] == 30 and m['scene'] == scene
                assert m['trainable_parameters'] == 30248000 and m.get('max_steps', 0) == 0
                ck = exp / m['checkpoint']
                assert ck.is_file()
                if args.verify_checkpoints:
                    assert sha(ck) == m['checkpoint_sha256'], str(ck)
                fingerprints.setdefault(scene, m['split_fingerprint'])
                assert fingerprints[scene] == m['split_fingerprint']
                with (exp / 'results' / f'{scene}_scores.csv').open() as f:
                    rows = list(csv.DictReader(f))
                keys = [(r['video'], r['frame'], int(r['label'])) for r in rows]
                assert len(set((v, f) for v, f, _ in keys)) == len(keys)
                reference_keys.setdefault(scene, keys)
                assert keys == reference_keys[scene], f'frame/label mismatch {exp.name}/{scene}'
                y = np.array([r[2] for r in keys])
                scores = np.array([float(r['anomaly_score']) for r in rows])
                assert np.isfinite(scores).all()
                auc = roc_auc_score(y, scores) * 100
                fpr, tpr, _ = roc_curve(y, scores)
                eer = float(fpr[np.nanargmin(np.abs(1 - tpr - fpr))])
                assert math.isclose(auc, m['auc_percent'], abs_tol=1e-9)
                assert math.isclose(eer, m['eer'], abs_tol=1e-12)
                assert len(y) == m['frames'] and int(y.sum()) == m['positive_frames']
                migrated = seed == 42 and key == 'top5_group_a05'
                if not migrated:
                    assert m['loss_ablation'] == lc
                    guard = m['model_guard']
                    for field, value in [('block_order','GLLL'),('ffn_type','gate'),('student_dim',768),('ocbe_fuse_hidden',512),('teacher_taps',12),('code_dim',64)]:
                        assert guard[field] == value
                    assert guard['loss_ablation'] == lc
                    launch, finish = launches[(exp.name, scene)], finishes[(exp.name, scene)]
                    assert finish['returncode'] == 0
                    log = ROOT / launch['log']
                    lines = log.read_text().splitlines()
                    assert [int(re.match(r'epoch=(\d+)', x)[1]) for x in lines if x.startswith('epoch=')] == list(range(1,31))
                    assert json.loads(lines[0])['model_guard'] == guard
                    assert json.loads(lines[-1]) == m
                else:
                    assert m['scoring'] == 'matched_group_balanced_top5'
                    source_dir = ROOT / cfg.get('migrated_from', 'ablation/da/GL/20261106_glll_gate_s42')
                    assert (source_dir / 'results' / f'{scene}.json').read_bytes() == (exp / 'results' / f'{scene}.json').read_bytes()
                checks.append({'experiment':exp.name, 'scene':scene, 'auc_percent':auc, 'eer':eer,
                               'frames':len(rows), 'split_fingerprint':m['split_fingerprint'],
                               'checkpoint_sha256':m['checkpoint_sha256'],
                               'checkpoint_hash_verified':args.verify_checkpoints,
                               'matched_score_evidence':'migrated matched re-evaluation' if migrated else 'same loss_score config in train/eval; config, result and log guard agree'})
                metrics[scene] = m
            data[(key, seed)] = {'experiment':exp.name, 'scenes':metrics,
                                'auc':st.mean(x['auc_percent'] for x in metrics.values()),
                                'eer':st.mean(x['eer'] for x in metrics.values())}
            print('validated', exp.name, flush=True)
    matrix = []
    for key in SPECS:
        matrix.append({'id':key, 'label':LABELS[key], 'auc_percent':stats([data[key,s]['auc'] for s in SEEDS]),
                       'eer':stats([data[key,s]['eer'] for s in SEEDS]),
                       'scene_auc_percent':{c:stats([data[key,s]['scenes'][c]['auc_percent'] for s in SEEDS]) for c in SCENES},
                       'scene_eer':{c:stats([data[key,s]['scenes'][c]['eer'] for s in SEEDS]) for c in SCENES}})
    pairs = [
        ('flatten_vs_patchmean', 'flatten_12mean', 'patchmean_12mean', '相同12层平均：Flatten − Patch Mean'),
        ('top5_vs_patchmean', 'top5_12mean', 'patchmean_12mean', '相同12层平均：Top-5 − Patch Mean'),
        ('group_vs_mean12', 'top5_group_a05', 'top5_12mean', '相同Top-5：4组α=0.5 − 12层平均'),
        ('max_vs_balanced', 'top5_group_a10', 'top5_group_a05', '相同Top-5：α=1 − α=0.5'),
        ('top1_vs_top5', 'top1_group_a05', 'top5_group_a05', '相同4组α=0.5：Top-1 − Top-5'),
        ('top10_vs_top5', 'top10_group_a05', 'top5_group_a05', '相同4组α=0.5：Top-10 − Top-5'),
        ('patchmean_vs_top5', 'patchmean_group_a05', 'top5_group_a05', '相同4组α=0.5：Patch Mean − Top-5'),
        ('patchmean_group_vs_mean12', 'patchmean_group_a05', 'patchmean_12mean', '相同Patch Mean：4组α=0.5 − 12层平均'),
    ]
    contrasts = [{'id':i,'left':a,'right':b,'label':label,
                  'auc_delta_pp':stats([data[a,s]['auc']-data[b,s]['auc'] for s in SEEDS]),
                  'eer_delta':stats([data[a,s]['eer']-data[b,s]['eer'] for s in SEEDS])} for i,a,b,label in pairs]
    payload = {'schema_version':1,'seeds':SEEDS,'scenes':SCENES,'matrix':matrix,'paired_contrasts':contrasts,
               'aggregation':'per-scene concat AUC x100 and discrete ROC EER; macro mean across seven scenes, then mean/sample SD across three seeds',
               'score_direction':'larger is more anomalous; no sign flip or test-selected readout',
               'patch_count':256,'feature_dim':768,'flatten_dim':196608,
               'limitations':['n=3 seeds; descriptive sample SD, not standard error or significance',
                              'train objective and matched test readout both change; these results do not isolate training-only effects',
                              'seed42 baseline migrated; seeds43/44 freshly trained for this Loss matrix',
                              'mean12 equals mean of four equal-sized group means; aggregation change adds a within-group max term',
                              'k=196 in non-topk config is unused; Patch Mean takes all actual 256 patches']}
    write_json(BASE/'summary.json',payload)
    write_json(BASE/'results/validation.json',{'status':'passed','scene_cells':len(checks), 'new_scene_jobs':161,
               'reused_scene_cells':7,'scheduler_final':ev[-1],'source_commit_new_runs':SOURCE,
               'checks':checks,'checkpoint_hashes_verified':args.verify_checkpoints})
    with (BASE/'results/matrix.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['variant','seed42_auc','seed43_auc','seed44_auc','mean_auc','sd_auc','seed42_eer','seed43_eer','seed44_eer','mean_eer','sd_eer'])
        for r in matrix:
            w.writerow([r['id']]+list(r['auc_percent']['values'].values())+[r['auc_percent']['mean'],r['auc_percent']['sample_sd']]+list(r['eer']['values'].values())+[r['eer']['mean'],r['eer']['sample_sd']])
    for (key,seed), d in data.items():
        p=BASE/d['experiment']; local={'experiment':d['experiment'],'seed':seed,'loss_ablation':json.loads((p/'config.json').read_text())['loss_ablation'],
                                     'macro_auc_percent':d['auc'],'macro_eer':d['eer'],
                                     'scenes':{s:{k:d['scenes'][s][k] for k in ['auc_percent','eer','frames','checkpoint_sha256']} for s in SCENES}}
        write_json(p/'results/summary.json',local)
        (p/'results/summary.md').write_text(f"# {d['experiment']}\n\n{LABELS[key]}，seed{seed}，七场景完成。训练和测试采用相同配置的 match 分数。\n\n宏平均 AUC：**{d['auc']:.4f}%**；宏平均 EER：**{d['eer']:.6f}**。\n\n[三 seed 矩阵与分析](../../summary.md)。\n")
    print('summary JSON, CSV, per-experiment summaries and validation written',flush=True)

if __name__ == '__main__':
    main()
