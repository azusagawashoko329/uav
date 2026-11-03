"""Run one canonical reverse-distillation variant on one Drone-Anomaly scene.

The six variants are defined in experiments/VARIANTS.md / VARIANTS.csv. Model construction
and the training loss are imported from scripts/uit_appearance_runner.py so the DA matrix
uses exactly the same architecture code as the formal UIT six-variant matrix.
"""
from __future__ import annotations

import argparse
import csv
import glob
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

from model import anomaly_readouts, build_appearance_teacher
from scripts.uit_appearance_runner import build_variant, loss_for

SCENES = {
    'bike': 'Bike Roundabout',
    'crossroads': 'Crossroads',
    'farmland': 'Farmland Inspection',
    'highway': 'Highway',
    'railway': 'Railway Inspection',
    'solar': 'Solar Panel Inspection',
    'vehicle': 'Vehicle Roundabout',
}

# Canonical variant ID -> frozen UIT builder key + teacher taps + display label.
VARIANT_SPECS = {
    'rd_s12_t4_h512': {
        'label': 'V0',
        'builder': 'uit_12block_dense512',
        'taps': [5, 7, 9, 11],
    },
    'rd_s4_t12_h512': {
        'label': 'V1',
        'builder': 'uit_4block_plain',
        'taps': list(range(12)),
    },
    'rd_s4_t12_gl768_h512': {
        'label': 'V2',
        'builder': 'uit_4block_gl768',
        'taps': list(range(12)),
    },
    'rd_s4_t12_gl192_h512': {
        'label': 'V3',
        'builder': 'uit_4block_gl192',
        'taps': list(range(12)),
    },
    'rd_s4_t12_gl192_h128': {
        'label': 'V4',
        'builder': 'uit_gl192_dense128',
        'taps': list(range(12)),
    },
    'rd_s4_t12_gl192_h128_qr2': {
        'label': 'V5',
        'builder': 'uit_gl192_dense128_rank2',
        'taps': list(range(12)),
    },
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
    train = sorted(glob.glob(str(scene_dir / 'sequence*' / 'train' / '*' / '*.jpg')))
    test = sorted(Path(p) for p in glob.glob(str(scene_dir / 'sequence*' / 'test' / '*'))
                  if Path(p).is_dir())
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
def evaluate(teacher, ocbe, student, scene_dir, videos, tf, device, out_csv, batch):
    rows, scores, labels = [], [], []
    for video in videos:
        paths = sorted(video.glob('*.jpg'), key=frame_key)
        label_path = Path(str(video) + '.npy')
        y = np.asarray(np.load(label_path, allow_pickle=True)).astype(int).ravel()
        if len(paths) != len(y):
            raise ValueError(f'label/frame count mismatch: {video}: {len(y)} vs {len(paths)}')
        loader = DataLoader(Frames(paths, tf), batch_size=batch, shuffle=False, num_workers=4)
        vid_scores = []
        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            taps = teacher(xb)['patches']
            rec = student(ocbe(taps))
            vid_scores.extend(anomaly_readouts(taps, rec, k=5)['topk'].cpu().tolist())
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


def main(default_experiment: Path | None = None):
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, default=default_experiment, required=default_experiment is None)
    p.add_argument('--scene', required=True, choices=sorted(SCENES))
    p.add_argument('--data-root', type=Path, default=None)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch', type=int, default=None)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--overwrite', action='store_true',
                   help='allow replacing an existing checkpoint/result in this experiment directory')
    a = p.parse_args()
    a.experiment = a.experiment.resolve()
    cfg = json.loads((a.experiment / 'config.json').read_text())
    variant_id = cfg['variant_id']
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f'unknown variant_id: {variant_id}')
    spec = VARIANT_SPECS[variant_id]
    if cfg.get('dataset') != 'da' or cfg.get('family') != 'appearance':
        raise ValueError(f'wrong dataset/family in {a.experiment}/config.json')
    if cfg.get('runner_variant') and cfg['runner_variant'] != spec['builder']:
        raise ValueError(f"runner_variant mismatch: config={cfg['runner_variant']} canonical={spec['builder']}")

    training = cfg.get('training', {})
    seed = a.seed if a.seed is not None else int(cfg['seed'])
    epochs = a.epochs if a.epochs is not None else int(training.get('epochs', 30))
    batch = a.batch if a.batch is not None else int(training.get('batch', 64))
    workers = a.workers if a.workers is not None else int(training.get('workers', 8))
    if seed != int(cfg['seed']):
        raise ValueError(f'seed mismatch: config={cfg["seed"]} CLI={seed}')
    if epochs != int(training.get('epochs', epochs)):
        raise ValueError(f'epochs mismatch: config={training.get("epochs")} CLI={epochs}')

    ckpt_path = a.experiment / 'ckpt' / f'{a.scene}_final.pth'
    metric_path = a.experiment / 'results' / f'{a.scene}.json'
    score_path = a.experiment / 'results' / f'{a.scene}_scores.csv'
    if not a.overwrite:
        for path in (ckpt_path, metric_path):
            if path.exists():
                raise FileExistsError(f'{path} already exists; use a new experiment or --overwrite')

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this experiment')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(f'cuda:{a.gpu}')

    data_root = a.data_root or Path(
        training.get('data_root')
        or os.environ.get('UAV_DA_ROOT')
        or '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly'
    )
    scene_dir = data_root / SCENES[a.scene]
    train, videos, fingerprint = dataset_manifest(scene_dir)
    teacher = build_appearance_teacher(layers=spec['taps']).to(device).eval()
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(),
                    T.Normalize(teacher.img_mean, teacher.img_std)])
    loader = DataLoader(Frames(train, tf), batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed),
                        num_workers=workers, pin_memory=True, drop_last=True)
    ocbe, student = build_variant({'variant': spec['builder'], 'seed': seed}, seed)
    ocbe = ocbe.to(device)
    student = student.to(device)
    params = list(ocbe.parameters()) + list(student.parameters())
    trainable_parameters = sum(p.numel() for p in params)
    expected = cfg.get('resources', {}).get('expected_trainable_parameters')
    if expected is not None and int(expected) != trainable_parameters:
        raise RuntimeError(f'parameter mismatch: config={expected} actual={trainable_parameters}')
    if getattr(ocbe, 'code_dim', None) != 64:
        raise RuntimeError(f'code_dim must be 64, got {getattr(ocbe, "code_dim", None)}')

    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=5e-2)
    steps_per_epoch = min(len(loader), a.max_steps) if a.max_steps else len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4,
                                               steps_per_epoch=steps_per_epoch, epochs=epochs)
    print(json.dumps({
        'dataset': 'Drone-Anomaly', 'variant_id': variant_id, 'variant_label': spec['label'],
        'scene': a.scene, 'seed': seed, 'epochs': epochs, 'data_root': str(data_root),
        'train_frames': len(train), 'test_videos': len(videos), 'split_fingerprint': fingerprint,
        'trainable_parameters': trainable_parameters, 'teacher_taps': spec['taps'],
        'cuda': torch.cuda.get_device_name(device), 'max_steps': a.max_steps,
    }, ensure_ascii=False), flush=True)

    t0 = time.monotonic()
    last_loss = 0.0
    for ep in range(epochs):
        ocbe.train()
        student.train()
        total = count = 0
        for step, xb in enumerate(loader):
            if a.max_steps and step >= a.max_steps:
                break
            xb = xb.to(device, non_blocking=True)
            with torch.no_grad():
                taps = teacher(xb)['patches']
            rec = student(ocbe(taps))
            loss = loss_for(taps, rec)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(xb)
            count += len(xb)
        last_loss = total / max(count, 1)
        print(f'epoch={ep + 1} loss={last_loss:.6f} elapsed_s={time.monotonic() - t0:.1f}', flush=True)

    ocbe.eval()
    student.eval()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'format_version': 1,
        'dataset': 'Drone-Anomaly',
        'variant_id': variant_id,
        'variant_label': spec['label'],
        'runner_variant': spec['builder'],
        'scene': a.scene,
        'seed': seed,
        'epochs': epochs,
        'split_fingerprint': fingerprint,
        'trainable_parameters': trainable_parameters,
        'train_loss': last_loss,
        'teacher_taps': spec['taps'],
        'ocbe_state_dict': {k: v.detach().cpu() for k, v in ocbe.state_dict().items()},
        'student_state_dict': {k: v.detach().cpu() for k, v in student.state_dict().items()},
    }, ckpt_path)
    ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    out = a.experiment / 'results'
    auc, eer, n, pos = evaluate(teacher, ocbe, student, scene_dir, videos, tf, device,
                                score_path, 32)
    metric = {
        'dataset': 'Drone-Anomaly',
        'variant_id': variant_id,
        'variant_label': spec['label'],
        'runner_variant': spec['builder'],
        'scene': a.scene,
        'seed': seed,
        'epochs': epochs,
        'auc_percent': auc,
        'eer': eer,
        'frames': n,
        'positive_frames': pos,
        'train_frames': len(train),
        'test_videos': len(videos),
        'split_fingerprint': fingerprint,
        'trainable_parameters': trainable_parameters,
        'teacher_taps': spec['taps'],
        'elapsed_s': time.monotonic() - t0,
        'max_steps': a.max_steps,
        'checkpoint': str(ckpt_path.relative_to(a.experiment)),
        'checkpoint_sha256': ckpt_sha,
        'cuda': torch.cuda.get_device_name(device),
    }
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps(metric, indent=2) + '\n')
    print(json.dumps(metric), flush=True)


if __name__ == '__main__':
    main()
