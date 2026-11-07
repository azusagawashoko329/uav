#!/usr/bin/env python3
"""Train/evaluate one DA V2 loss ablation.

All runs use the V2 mother template and vary only the reconstruction error and
12-layer/group aggregation specified in config.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from model import build_appearance_teacher, build_bottleneck
from model.loss import loss_score
from model.student_structural import build_structural_student

SCENES = {
    'bike': 'Bike Roundabout',
    'crossroads': 'Crossroads',
    'farmland': 'Farmland Inspection',
    'highway': 'Highway',
    'railway': 'Railway Inspection',
    'solar': 'Solar Panel Inspection',
    'vehicle': 'Vehicle Roundabout',
}


class Frames(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert('RGB'))


def frame_key(path):
    return int(Path(path).stem)


def dataset_manifest(scene_dir):
    train = sorted(str(p) for p in scene_dir.glob('sequence*/train/*/*.jpg'))
    test = sorted(p for p in scene_dir.glob('sequence*/test/*') if p.is_dir())
    if not train or not test:
        raise FileNotFoundError(f'missing train/test frames in {scene_dir}')
    lines = [f'{Path(p).relative_to(scene_dir)}:{Path(p).stat().st_size}' for p in train]
    for video in test:
        frames = sorted(video.glob('*.jpg'), key=frame_key)
        label = Path(str(video) + '.npy')
        if not frames or not label.is_file():
            raise FileNotFoundError(f'missing frames or labels: {video}')
        lines.extend(f'{p.relative_to(scene_dir)}:{p.stat().st_size}' for p in frames)
        lines.append(f'{label.relative_to(scene_dir)}:{hashlib.sha256(label.read_bytes()).hexdigest()}')
    return train, test, hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def eer_of(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    return float(fpr[np.nanargmin(np.abs(fnr - fpr))])


@torch.no_grad()
def evaluate(teacher, ocbe, student, scene_dir, videos, tf, device, out_csv, batch, loss_cfg):
    rows, scores, labels = [], [], []
    for video in videos:
        paths = sorted(video.glob('*.jpg'), key=frame_key)
        label_path = Path(str(video) + '.npy')
        y = np.asarray(np.load(label_path, allow_pickle=True)).astype(int).ravel()
        if len(paths) != len(y):
            raise ValueError(f'label/frame count mismatch: {video}: {len(y)} vs {len(paths)}')
        loader = DataLoader(Frames(paths, tf), batch_size=batch, shuffle=False,
                            num_workers=4)
        vid_scores = []
        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            taps = teacher(xb)['patches']
            rec = student(ocbe(taps))
            vid_scores.extend(loss_score(taps, rec, error=loss_cfg['error_mode'], k=int(loss_cfg['k']), aggregation=loss_cfg['aggregation'], alpha=float(loss_cfg['alpha'])).cpu().tolist())
        for path, label, score in zip(paths, y, vid_scores):
            rows.append((str(video.relative_to(scene_dir)), path.name, int(label), float(score)))
        scores.extend(vid_scores)
        labels.extend(y.tolist())
    auc = roc_auc_score(labels, scores) * 100
    eer = eer_of(labels, scores)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video', 'frame', 'label', 'anomaly_score'])
        writer.writerows(rows)
    return auc, eer, len(labels), int(sum(labels))


def load_config(exp: Path):
    cfg = json.loads((exp / 'config.json').read_text())
    required = {
        'dataset': 'da', 'family': 'appearance', 'seed': 42,
        'block_order': cfg.get('ablation', {}).get('block_order'),
        'ffn_type': cfg.get('ablation', {}).get('ffn_type'),
    }
    if cfg.get('dataset') != 'da' or cfg.get('family') != 'appearance':
        raise ValueError(f'wrong dataset/family in {exp / "config.json"}')
    ab = cfg.get('ablation', {})
    if ab.get('mother_template') != 'rd_s4_t12_gl768_h512':
        raise ValueError('ablation mother template must be rd_s4_t12_gl768_h512')
    if ab.get('block_order') != 'GLLL' or ab.get('ffn_type') != 'gate':
        raise ValueError('loss ablation must use the V2 GLLL + gate student')
    loss_cfg = cfg.get('loss_ablation', {})
    if loss_cfg.get('error_mode') not in {'flatten', 'patchmean', 'topk'}:
        raise ValueError(f'invalid loss error_mode: {loss_cfg.get("error_mode")}')
    if loss_cfg.get('aggregation') not in {'mean12', 'group'}:
        raise ValueError(f'invalid loss aggregation: {loss_cfg.get("aggregation")}')
    if int(loss_cfg.get('k', 0)) < 1 or int(loss_cfg.get('k', 0)) > 196:
        raise ValueError(f'invalid loss k: {loss_cfg.get("k")}')
    if not 0 <= float(loss_cfg.get('alpha', -1)) <= 1:
        raise ValueError(f'invalid loss alpha: {loss_cfg.get("alpha")}')
    fixed = cfg.get('fixed', {})
    checks = {
        'teacher_taps': list(range(12)), 'student_dim': 768, 'ocbe_fuse_hidden': 512,
        'code_dim': 64, 'epochs': 30, 'batch': 64,
    }
    for key, expected in checks.items():
        actual = fixed.get(key) if key in fixed else cfg.get('training', {}).get(key)
        if actual != expected:
            raise ValueError(f'{key} mismatch: expected {expected}, got {actual}')
    return cfg


def main(default_experiment: Path | None = None):
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, default=default_experiment,
                   required=default_experiment is None)
    p.add_argument('--scene', required=True, choices=sorted(SCENES))
    p.add_argument('--data-root', type=Path, default=None)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch', type=int, default=None)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    exp = a.experiment.resolve()
    cfg = load_config(exp)
    ab = cfg['ablation']
    loss_cfg = cfg['loss_ablation']
    training = cfg['training']
    seed = a.seed if a.seed is not None else int(cfg['seed'])
    epochs = a.epochs if a.epochs is not None else int(training['epochs'])
    batch = a.batch if a.batch is not None else int(training['batch'])
    workers = a.workers if a.workers is not None else int(training['workers'])
    if seed != int(cfg['seed']) or epochs != int(training['epochs']):
        raise ValueError('CLI seed/epochs do not match config')
    ckpt_path = exp / 'ckpt' / f'{a.scene}_final.pth'
    metric_path = exp / 'results' / f'{a.scene}.json'
    score_path = exp / 'results' / f'{a.scene}_scores.csv'
    if not a.overwrite and (ckpt_path.exists() or metric_path.exists()):
        raise FileExistsError(f'outputs already exist for {a.scene}; use --overwrite')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this experiment')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device(f'cuda:{a.gpu}')
    data_root = a.data_root or Path(training.get('data_root') or os.environ.get(
        'UAV_DA_ROOT', '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly'))
    scene_dir = data_root / SCENES[a.scene]
    train, videos, fingerprint = dataset_manifest(scene_dir)
    teacher = build_appearance_teacher(layers=list(range(12))).to(device).eval()
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(),
                    T.Normalize(teacher.img_mean, teacher.img_std)])
    loader = DataLoader(Frames(train, tf), batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=workers,
                        pin_memory=True, drop_last=True)
    ocbe = build_bottleneck(n_taps=12, code_dim=64, fuse_hidden=512,
                            use_tap_norm=True).to(device)
    student = build_structural_student(dim=768, code_dim=64, heads=8, ffn_ratio=2.0,
                                       teacher_dim=768, block_order=ab['block_order'],
                                       ffn_type=ab['ffn_type']).to(device)
    params = list(ocbe.parameters()) + list(student.parameters())
    trainable_parameters = sum(p.numel() for p in params)
    expected = cfg.get('resources', {}).get('expected_trainable_parameters')
    if expected is not None and int(expected) != trainable_parameters:
        raise RuntimeError(f'parameter mismatch: config={expected} actual={trainable_parameters}')
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=5e-2)
    steps_per_epoch = min(len(loader), a.max_steps) if a.max_steps else len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4,
                                                steps_per_epoch=steps_per_epoch, epochs=epochs)
    guard = {'block_order': student.block_order, 'ffn_type': student.ffn_type,
             'student_dim': 768, 'ocbe_fuse_hidden': getattr(ocbe.fuse[0], 'out_features', None),
             'teacher_taps': 12, 'code_dim': getattr(ocbe, 'code_dim', None), 'loss_ablation': loss_cfg}
    print(json.dumps({'dataset': 'Drone-Anomaly', 'experiment': cfg['id'], 'scene': a.scene,
                      'seed': seed, 'epochs': epochs, 'ablation': ab, 'loss_ablation': loss_cfg, 'model_guard': guard,
                      'train_frames': len(train), 'test_videos': len(videos),
                      'split_fingerprint': fingerprint, 'trainable_parameters': trainable_parameters,
                      'cuda': torch.cuda.get_device_name(device), 'max_steps': a.max_steps},
                     ensure_ascii=False), flush=True)
    t0 = time.monotonic(); last_loss = 0.0
    for ep in range(epochs):
        ocbe.train(); student.train(); total = count = 0
        for step, xb in enumerate(loader):
            if a.max_steps and step >= a.max_steps:
                break
            xb = xb.to(device, non_blocking=True)
            with torch.no_grad():
                taps = teacher(xb)['patches']
            rec = student(ocbe(taps))
            loss = loss_score(taps, rec, error=loss_cfg['error_mode'], k=int(loss_cfg['k']), aggregation=loss_cfg['aggregation'], alpha=float(loss_cfg['alpha'])).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += loss.item() * len(xb); count += len(xb)
        last_loss = total / max(count, 1)
        print(f'epoch={ep + 1} loss={last_loss:.6f} elapsed_s={time.monotonic() - t0:.1f}', flush=True)
    ocbe.eval(); student.eval(); ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format_version': 1, 'dataset': 'Drone-Anomaly', 'experiment_id': cfg['id'],
                'ablation': ab, 'loss_ablation': loss_cfg, 'scene': a.scene, 'seed': seed, 'epochs': epochs,
                'split_fingerprint': fingerprint, 'trainable_parameters': trainable_parameters,
                'train_loss': last_loss, 'model_guard': guard,
                'ocbe_state_dict': {k: v.detach().cpu() for k, v in ocbe.state_dict().items()},
                'student_state_dict': {k: v.detach().cpu() for k, v in student.state_dict().items()}}, ckpt_path)
    ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    auc, eer, n, pos = evaluate(teacher, ocbe, student, scene_dir, videos, tf, device,
                                score_path, 32, loss_cfg)
    metric = {'dataset': 'Drone-Anomaly', 'experiment_id': cfg['id'], 'ablation': ab, 'loss_ablation': loss_cfg,
              'scene': a.scene, 'seed': seed, 'epochs': epochs, 'auc_percent': auc,
              'eer': eer, 'frames': n, 'positive_frames': pos, 'train_frames': len(train),
              'test_videos': len(videos), 'split_fingerprint': fingerprint,
              'trainable_parameters': trainable_parameters, 'model_guard': guard,
              'elapsed_s': time.monotonic() - t0, 'max_steps': a.max_steps,
              'checkpoint': str(ckpt_path.relative_to(exp)), 'checkpoint_sha256': ckpt_sha,
              'cuda': torch.cuda.get_device_name(device)}
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps(metric, indent=2) + '\n')
    print(json.dumps(metric), flush=True)


if __name__ == '__main__':
    main()
