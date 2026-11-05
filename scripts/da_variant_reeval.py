#!/usr/bin/env python3
"""Re-evaluate one DA variant/scene checkpoint with the training-matched score.

No training. The original runner scored V1-V5 with the plain mean-over-taps top-k
readout, while training used the group-balanced 0.5*mean + 0.5*max objective.
This script loads the existing final checkpoint and writes matched scores to
results_matched/ so the original result files remain available.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import build_appearance_teacher, matched_score
from scripts.da_variant_runner import SCENES, VARIANT_SPECS, dataset_manifest, evaluate
from scripts.uit_appearance_runner import build_variant


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--scene', required=True, choices=sorted(SCENES))
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    exp = a.experiment.resolve()
    cfg = json.loads((exp / 'config.json').read_text())
    if cfg.get('dataset') != 'da' or cfg.get('family') != 'appearance':
        raise ValueError(f'wrong dataset/family: {exp}')
    vid = cfg['variant_id']
    spec = VARIANT_SPECS[vid]
    cfg_builder = cfg.get('runner_variant', cfg.get('variant'))
    if cfg_builder != spec['builder']:
        raise ValueError(f'builder mismatch: {cfg_builder} != {spec["builder"]}')
    if list(cfg.get('teacher_layers', spec['taps'])) != list(spec['taps']):
        raise ValueError('teacher taps mismatch with canonical variant')

    ckpt_path = exp / 'ckpt' / f'{a.scene}_final.pth'
    out_dir = exp / 'results_matched'
    out_json = out_dir / f'{a.scene}.json'
    out_csv = out_dir / f'{a.scene}_scores.csv'
    if out_json.exists() and not a.overwrite:
        print(f'SKIP {exp.name} {a.scene}: {out_json} exists')
        return
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    device = torch.device(f'cuda:{a.gpu}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    for key in ('variant_id', 'runner_variant', 'teacher_taps', 'trainable_parameters', 'scene'):
        if key in ckpt and key == 'scene' and ckpt[key] != a.scene:
            raise RuntimeError(f'checkpoint scene {ckpt[key]} != {a.scene}')
        if key in ckpt and key == 'variant_id' and ckpt[key] != vid:
            raise RuntimeError(f'checkpoint variant_id {ckpt[key]} != {vid}')
        if key in ckpt and key == 'runner_variant' and ckpt[key] != spec['builder']:
            raise RuntimeError(f'checkpoint runner_variant {ckpt[key]} != {spec["builder"]}')
        if key in ckpt and key == 'teacher_taps' and list(ckpt[key]) != list(spec['taps']):
            raise RuntimeError(f'checkpoint teacher_taps {ckpt[key]} != {spec["taps"]}')

    data_root = Path(cfg['training'].get('data_root', '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly'))
    scene_dir = data_root / SCENES[a.scene]
    _, videos, fingerprint = dataset_manifest(scene_dir)
    teacher = build_appearance_teacher(layers=spec['taps']).to(device).eval()
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(),
                    T.Normalize(teacher.img_mean, teacher.img_std)])
    ocbe, student = build_variant({'variant': spec['builder'], 'seed': int(cfg['seed'])}, int(cfg['seed']))
    ocbe.load_state_dict(ckpt['ocbe_state_dict'])
    student.load_state_dict(ckpt['student_state_dict'])
    ocbe = ocbe.to(device).eval()
    student = student.to(device).eval()
    actual = sum(p.numel() for p in ocbe.parameters()) + sum(p.numel() for p in student.parameters())
    expected = int(cfg['resources']['expected_trainable_parameters'])
    if actual != expected:
        raise RuntimeError(f'parameter mismatch: {actual} != {expected}')
    if int(ckpt.get('trainable_parameters', actual)) != expected:
        raise RuntimeError('checkpoint parameter count mismatch')

    # Reuse the runner's evaluate(), which now calls matched_score.
    auc, eer, n, pos = evaluate(teacher, ocbe, student, scene_dir, videos, tf, device,
                                out_csv, a.batch)
    old = {}
    old_path = exp / 'results' / f'{a.scene}.json'
    if old_path.is_file():
        old_metric = json.loads(old_path.read_text())
        old = {'auc_percent': old_metric.get('auc_percent'),
               'eer': old_metric.get('eer'),
               'scoring': 'legacy_topk_mean'}
    metric = {
        'dataset': 'Drone-Anomaly',
        'variant_id': vid, 'variant_label': spec['label'],
        'runner_variant': spec['builder'], 'scene': a.scene,
        'seed': int(cfg['seed']), 'epochs': int(cfg['training']['epochs']),
        'auc_percent': auc, 'eer': eer, 'frames': n, 'positive_frames': pos,
        'trainable_parameters': actual, 'teacher_taps': spec['taps'],
        'split_fingerprint': fingerprint,
        'checkpoint': str(ckpt_path.relative_to(exp)),
        'checkpoint_sha256': __import__('hashlib').sha256(ckpt_path.read_bytes()).hexdigest(),
        'scoring': 'matched_group_balanced_top5',
        'scoring_note': ('V0: mean over 4 taps of top5 mean; V1-V5: mean over 4 groups of '
                         '0.5*mean_tap(top5 mean)+0.5*max_tap(top5 mean)'),
        'previous': old,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metric, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(metric, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
