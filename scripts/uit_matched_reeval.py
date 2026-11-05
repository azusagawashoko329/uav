#!/usr/bin/env python3
"""Re-evaluate one UIT final checkpoint with the training-matched score."""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path
import torch
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import build_appearance_teacher
from scripts.uit_appearance_runner import build_variant, evaluate, manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--max-videos', type=int, default=0)
    a = p.parse_args()
    exp = a.experiment.resolve()
    cfg = json.loads((exp / 'config.json').read_text())
    ckpt_path = exp / 'ckpt' / 'uit_final.pth'
    out_json = exp / 'results' / 'uit_matched.json'
    out_csv = exp / 'results' / 'uit_matched_scores.csv'
    if out_json.exists() and not a.overwrite:
        print(f'SKIP {exp.name}: {out_json} exists')
        return
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if ckpt.get('variant') != cfg.get('variant'):
        raise RuntimeError(f"checkpoint variant {ckpt.get('variant')} != {cfg.get('variant')}")
    if int(ckpt.get('trainable_parameters', -1)) != int(cfg['resources']['expected_trainable_parameters']):
        raise RuntimeError('checkpoint parameter count mismatch')
    if ckpt.get('split_fingerprint') is None:
        raise RuntimeError('checkpoint missing split_fingerprint')
    device = torch.device(f'cuda:{a.gpu}')
    data_root = Path(cfg['training'].get('data_root', '/data1/data8tb/liangyuxuan/dataset/UIT-ADrone'))
    train, test_dirs, fingerprint = manifest(data_root)
    if a.max_videos:
        test_dirs = test_dirs[:a.max_videos]
    if fingerprint != ckpt['split_fingerprint']:
        raise RuntimeError(f'split fingerprint mismatch: {fingerprint} != {ckpt["split_fingerprint"]}')
    teacher_layers = cfg.get('teacher_layers', list(range(12)))
    teacher = build_appearance_teacher(layers=teacher_layers).to(device).eval()
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(),
                    T.Normalize(teacher.img_mean, teacher.img_std)])
    ocbe, student = build_variant({'variant': cfg['variant'], 'seed': int(cfg['seed'])}, int(cfg['seed']))
    ocbe.load_state_dict(ckpt['ocbe_state_dict'])
    student.load_state_dict(ckpt['student_state_dict'])
    ocbe = ocbe.to(device).eval()
    student = student.to(device).eval()
    actual = sum(p.numel() for p in ocbe.parameters()) + sum(p.numel() for p in student.parameters())
    expected = int(cfg['resources']['expected_trainable_parameters'])
    if actual != expected:
        raise RuntimeError(f'parameter mismatch: {actual} != {expected}')
    auc, eer, n, pos = evaluate(teacher, ocbe, student, test_dirs, tf, device, out_csv, a.batch)
    metric = {
        'dataset': 'UIT-ADrone', 'variant_id': cfg['variant_id'], 'variant_label': cfg['variant_label'],
        'variant': cfg['variant'], 'seed': int(cfg['seed']), 'epochs': int(cfg['training']['epochs']),
        'auc_percent': auc, 'eer': eer, 'frames': n, 'positive_frames': pos,
        'train_frames': len(train), 'test_videos': len(test_dirs),
        'split_fingerprint': fingerprint, 'trainable_parameters': actual,
        'checkpoint': 'ckpt/uit_final.pth',
        'checkpoint_sha256': hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        'scoring': 'matched_group_balanced_top5',
        'scoring_note': ('V0: mean over 4 taps of top5 mean; V1-V5: mean over 4 groups of '
                         '0.5*mean_tap(top5 mean)+0.5*max_tap(top5 mean).'),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metric, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(metric, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
