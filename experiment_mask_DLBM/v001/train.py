#!/usr/bin/env python3
"""Train one scene's DLBM v001 and make a frozen decision on its fixed manifest.

``boundary_train`` is deterministically split inside every subvideo into representation-fit and
representation-validation frames.  ``boundary_decision`` is never used for optimization, epoch
selection, density initialization, or calibration.  It is read exactly once after both the masked
representation and ordered density have been frozen.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


VERSION_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = VERSION_ROOT.parent
ROOT = EXPERIMENT_ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_DLBM.common import ensure_all_layer_probe, set_seed
from experiment_mask_DLBM.v001.dlbm import (
    MaskedRelationDLBM, OrderedChangePointDensity, policy_from_scene_logits,
    representation_loss,
)
from experiment_mask_DLBM.prepare_splits import DEFAULT_DATA_ROOT, SCENES
from model import build_appearance_teacher, build_bottleneck, build_student


VERSION = 'v001_relation_changepoint'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path, data_root, scene):
    path = Path(path).resolve()
    data_root = Path(data_root).resolve()
    scene_root = (data_root / SCENES[scene]).resolve()
    values = []
    with path.open(encoding='utf-8') as handle:
        for line_number, raw in enumerate(handle, 1):
            entry = raw.strip()
            if not entry or entry.startswith('#'):
                continue
            value = Path(entry)
            value = value.resolve() if value.is_absolute() else (data_root / value).resolve()
            if not value.is_file():
                raise FileNotFoundError(f'{path}:{line_number}: missing frame {value}')
            try:
                value.relative_to(scene_root)
            except ValueError as exc:
                raise ValueError(
                    f'{path}:{line_number}: frame outside scene root {scene_root}: {value}') from exc
            values.append(value)
    if not values or len(values) != len(set(values)):
        raise ValueError(f'manifest is empty or contains duplicates: {path}')
    return values


def stable_key(seed, path):
    value = f'{seed}\0{Path(path).as_posix()}'.encode('utf-8')
    return hashlib.sha256(value).digest()


def split_fit_validation(paths, seed=42, validation_fraction=0.2):
    """Keep every subvideo represented in both subsets without consulting decision frames."""
    by_video = defaultdict(list)
    for path in paths:
        by_video[Path(path).parent].append(Path(path))
    fit, validation = [], []
    records = {}
    for video, frames in sorted(by_video.items(), key=lambda item: item[0].as_posix()):
        ordered = sorted(frames, key=lambda value: stable_key(seed, value))
        if len(ordered) < 2:
            raise ValueError(f'cannot split one-frame subvideo: {video}')
        validation_count = max(1, min(len(ordered) - 1, round(validation_fraction * len(ordered))))
        validation.extend(ordered[:validation_count])
        fit.extend(ordered[validation_count:])
        records[video.as_posix()] = {
            'total': len(ordered),
            'fit': len(ordered) - validation_count,
            'validation': validation_count,
        }
    fit, validation = sorted(fit), sorted(validation)
    if set(fit) & set(validation) or set(fit) | set(validation) != set(paths):
        raise RuntimeError('internal boundary_train split is not a disjoint partition')
    return fit, validation, records


class PathDataset(Dataset):
    def __init__(self, paths, image_size, mean, std, return_index=False):
        self.paths = [Path(path) for path in paths]
        self.return_index = bool(return_index)
        self.transform = transforms.Compose((
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = self.transform(Image.open(self.paths[index]).convert('RGB'))
        return (image, index) if self.return_index else image


def make_loader(dataset, batch_size, workers, shuffle, seed):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=torch.cuda.is_available(), drop_last=False, generator=generator,
        persistent_workers=bool(workers),
    )


def load_probe(scene, probe_dir, device):
    path = Path(probe_dir) / f'{scene}.pth'
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location='cpu')
    cfg = ensure_all_layer_probe(checkpoint, scene)
    if int(cfg.get('img', -1)) != 256:
        raise ValueError(f'{scene}: v001 requires img=256, found {cfg.get("img")}')
    teacher = build_appearance_teacher(layers=list(range(12))).to(device).eval()
    ocbe = build_bottleneck(
        n_taps=12, code_dim=cfg['code_dim'],
        use_tap_norm=not cfg.get('no_tap_norm', False),
    ).to(device).eval()
    student = build_student(
        n_taps=12, code_dim=cfg['code_dim'], depth=cfg['depth'],
        tap_indices=list(range(12)), tap_align='absolute',
    ).to(device).eval()
    ocbe.load_state_dict(checkpoint['ocbe'])
    student.load_state_dict(checkpoint['student'])
    teacher.requires_grad_(False)
    ocbe.requires_grad_(False)
    student.requires_grad_(False)
    return teacher, ocbe, student, cfg, path


def probe_features(images, teacher, ocbe, student, amp):
    with torch.no_grad():
        with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=amp):
            teacher_taps = teacher(images)['patches']
            student_taps = student(ocbe(teacher_taps))
    return teacher_taps, student_taps


def metric_accumulator():
    return {'total': 0.0, 'prediction': 0.0, 'transition': 0.0,
            'relation_residual_ratio': 0.0, 'count': 0}


def add_metrics(total, parts, output, count, accumulator):
    accumulator['total'] += float(total.detach()) * count
    accumulator['prediction'] += float(parts['prediction'].detach()) * count
    accumulator['transition'] += float(parts['transition'].detach()) * count
    accumulator['relation_residual_ratio'] += (
        float(output['relation_residual_ratio'].detach()) * count
    )
    accumulator['count'] += count


def final_metrics(accumulator):
    count = max(1, accumulator.pop('count'))
    return {key: value / count for key, value in accumulator.items()}


def train_representation_epoch(
    model, loader, teacher, ocbe, student, optimizer, scaler, device, amp, grad_clip,
    max_steps=0,
):
    model.train()
    totals = metric_accumulator()
    gradient_total, steps = 0.0, 0
    for step, images in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        images = images.to(device, non_blocking=True)
        teacher_taps, student_taps = probe_features(images, teacher, ocbe, student, amp)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(teacher_taps, student_taps)
            loss, parts = representation_loss(output)
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite DLBM representation loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(gradient):
            raise FloatingPointError('non-finite DLBM gradient')
        scaler.step(optimizer)
        scaler.update()
        add_metrics(loss, parts, output, len(images), totals)
        gradient_total += float(gradient)
        steps += 1
    result = final_metrics(totals)
    result['grad_norm'] = gradient_total / max(1, steps)
    return result


@torch.no_grad()
def evaluate_representation(
    model, loader, teacher, ocbe, student, device, amp, use_relation=True,
):
    model.eval()
    totals = metric_accumulator()
    token_sum = None
    token_square_sum = None
    token_count = 0
    for images in loader:
        images = images.to(device, non_blocking=True)
        teacher_taps, student_taps = probe_features(images, teacher, ocbe, student, amp)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(
                teacher_taps, student_taps, use_relation=use_relation,
            )
            loss, parts = representation_loss(output)
        add_metrics(loss, parts, output, len(images), totals)
        # Variance is measured at fixed layer/patch coordinates across frames.  A nearly-zero
        # value would expose a scene-mean shortcut before the ordered density is fitted.
        tokens = output['transition_set'].float()
        batch_sum = tokens.sum(dim=0)
        batch_square_sum = tokens.square().sum(dim=0)
        token_sum = batch_sum if token_sum is None else token_sum + batch_sum
        token_square_sum = (
            batch_square_sum if token_square_sum is None
            else token_square_sum + batch_square_sum
        )
        token_count += len(images)
    result = final_metrics(totals)
    mean = token_sum / token_count
    variance = token_square_sum / token_count - mean.square()
    result['transition_token_variance'] = float(variance.clamp_min(0).mean())
    return result


def roll_taps(taps, shifts):
    outputs = []
    for layer, values in enumerate(taps):
        grid = values.reshape(len(values), 16, 16, values.shape[-1])
        rolled = [
            torch.roll(grid[index], shifts=shifts[index][layer], dims=(0, 1))
            for index in range(len(grid))
        ]
        outputs.append(torch.stack(rolled).flatten(1, 2))
    return outputs


@torch.no_grad()
def representation_interventions(model, dataset, teacher, ocbe, student, device, amp, count, seed):
    model.eval()
    count = min(int(count), len(dataset))
    indices = sorted(random.Random(seed).sample(range(len(dataset)), count))
    images = torch.stack([dataset[index] for index in indices]).to(device)
    teacher_taps, student_taps = probe_features(images, teacher, ocbe, student, amp)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        aligned, _ = representation_loss(model(teacher_taps, student_taps))
        no_relation, _ = representation_loss(
            model(teacher_taps, student_taps, use_relation=False),
        )
        generator = torch.Generator().manual_seed(seed + 1)
        shifts = [[(
            int(torch.randint(0, 16, (), generator=generator)),
            int(torch.randint(0, 16, (), generator=generator)),
        ) for _ in range(12)] for _ in range(count)]
        rolled_teacher = roll_taps(teacher_taps, shifts)
        rolled_student = roll_taps(student_taps, shifts)
        rolled, _ = representation_loss(model(rolled_teacher, rolled_student))
    return {
        'frames': count,
        'aligned_loss': float(aligned),
        'independent_layer_roll_loss': float(rolled),
        'roll_over_aligned': float(rolled / aligned.clamp_min(1e-12)),
        'without_relation_loss': float(no_relation),
        'without_relation_over_full': float(no_relation / aligned.clamp_min(1e-12)),
    }


@torch.no_grad()
def transition_moments(model, loader, teacher, ocbe, student, device, amp):
    model.eval()
    totals = {
        'global_sum': None, 'global_square_sum': None,
        'local_sum': None, 'local_square_sum': None,
    }
    frame_count = 0
    for images in loader:
        images = images.to(device, non_blocking=True)
        teacher_taps, student_taps = probe_features(images, teacher, ocbe, student, amp)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(teacher_taps, student_taps)
        tokens = output['transition_set'].float()
        global_tokens = tokens[:, :, 0]
        local_tokens = tokens[:, :, 1:]
        values = {
            'global_sum': global_tokens.sum(dim=0),
            'global_square_sum': global_tokens.square().sum(dim=0),
            'local_sum': local_tokens.sum(dim=(0, 2)),
            'local_square_sum': local_tokens.square().sum(dim=(0, 2)),
        }
        for key, value in values.items():
            value = value.cpu()
            totals[key] = value if totals[key] is None else totals[key] + value
        frame_count += len(tokens)
    if frame_count == 0:
        raise ValueError('cannot estimate transition moments from an empty loader')
    moments = {
        'global_mean': totals['global_sum'] / frame_count,
        'global_second': totals['global_square_sum'] / frame_count,
        'local_mean': totals['local_sum'] / (frame_count * model.patch_count),
        'local_second': totals['local_square_sum'] / (frame_count * model.patch_count),
    }
    if not all(torch.isfinite(value).all() for value in moments.values()):
        raise FloatingPointError('non-finite transition-set moments')
    return moments


def moments_to_device(moments, device):
    return {key: value.to(device) for key, value in moments.items()}


@torch.no_grad()
def density_nll(density, moments):
    density.eval()
    loss, _ = density.scene_marginal_nll_from_moments(moments)
    return float(loss)


def train_density(density, fit_moments, validation_moments, args):
    density.initialize_from_moments(fit_moments)
    optimizer = torch.optim.AdamW(
        density.parameters(), lr=args.density_lr, weight_decay=0.0,
    )
    best_state = copy.deepcopy(density.state_dict())
    best_validation = density_nll(density, validation_moments)
    best_epoch, stale = 0, 0
    history = []
    for epoch in range(1, args.density_epochs + 1):
        density.train()
        loss, _ = density.scene_marginal_nll_from_moments(fit_moments)
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite ordered-density loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(density.parameters(), 5.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError('non-finite ordered-density gradient')
        optimizer.step()
        validation = density_nll(density, validation_moments)
        row = {
            'epoch': epoch,
            'train_nll': float(loss.detach()),
            'validation_nll': validation,
        }
        history.append(row)
        if validation < best_validation - args.density_min_delta:
            best_validation = validation
            best_epoch = epoch
            best_state = copy.deepcopy(density.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= args.density_min_epochs and stale >= args.density_patience:
            break
    density.load_state_dict(best_state)
    density.eval().requires_grad_(False)
    return history, best_epoch, best_validation


@torch.no_grad()
def decision(
    model, density, dataset, loader, teacher, ocbe, student, device, amp,
):
    model.eval()
    density.eval()
    by_video = defaultdict(lambda: {'sum': None, 'count': 0})
    for images, indices in loader:
        images = images.to(device, non_blocking=True)
        teacher_taps, student_taps = probe_features(images, teacher, ocbe, student, amp)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            tokens = model(teacher_taps, student_taps)['transition_set']
        logits = density.boundary_logits(tokens.float()).cpu()
        for row, index in zip(logits, indices):
            video = dataset.paths[int(index)].parent.as_posix()
            record = by_video[video]
            record['sum'] = row.clone() if record['sum'] is None else record['sum'] + row
            record['count'] += 1
    per_video = {}
    video_logits = []
    for video, record in sorted(by_video.items()):
        logits = record['sum'] / record['count']
        video_logits.append(logits)
        per_video[video] = {
            'frames': record['count'],
            'policy': policy_from_scene_logits(logits),
        }
    # Equal subvideo weighting prevents long videos from defining the scene boundary alone.
    scene_logits = torch.stack(video_logits).mean(dim=0)
    return policy_from_scene_logits(scene_logits), per_video, [
        float(value) for value in scene_logits
    ]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, choices=sorted(SCENES))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--split_root', type=Path,
                        default=ROOT / 'experiment_mask_DLBM' / 'splits' / 'seed42')
    parser.add_argument('--probe_dir', type=Path,
                        default=ROOT / 'experiment_mask_DLBM' / 'run' / 'probe' / 'ckpt')
    parser.add_argument('--out_root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--token_dim', type=int, default=64)
    parser.add_argument('--state_dim', type=int, default=32)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--min_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--min_relative_improvement', type=float, default=0.002)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--density_epochs', type=int, default=300)
    parser.add_argument('--density_min_epochs', type=int, default=50)
    parser.add_argument('--density_patience', type=int, default=35)
    parser.add_argument('--density_min_delta', type=float, default=1e-5)
    parser.add_argument('--density_batch_size', type=int, default=256)
    parser.add_argument('--density_lr', type=float, default=2e-3)
    parser.add_argument('--intervention_frames', type=int, default=16)
    parser.add_argument('--max_train_steps', type=int, default=0,
                        help='smoke only: cap representation batches per epoch')
    args = parser.parse_args()

    if args.min_epochs > args.max_epochs:
        raise ValueError('min_epochs may not exceed max_epochs')
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda'
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    train_manifest = args.split_root / args.scene / 'boundary_train.txt'
    decision_manifest = args.split_root / args.scene / 'boundary_decision.txt'
    boundary_train_paths = read_manifest(
        train_manifest, args.data_root, args.scene,
    )
    decision_paths = read_manifest(
        decision_manifest, args.data_root, args.scene,
    )
    if set(boundary_train_paths) & set(decision_paths):
        raise RuntimeError('boundary_train and boundary_decision overlap')
    fit_paths, validation_paths, internal_records = split_fit_validation(
        boundary_train_paths, seed=args.seed,
    )

    teacher, ocbe, student, probe_cfg, probe_path = load_probe(
        args.scene, args.probe_dir, device,
    )
    fit_dataset = PathDataset(
        fit_paths, probe_cfg['img'], teacher.img_mean, teacher.img_std,
    )
    validation_dataset = PathDataset(
        validation_paths, probe_cfg['img'], teacher.img_mean, teacher.img_std,
    )
    decision_dataset = PathDataset(
        decision_paths, probe_cfg['img'], teacher.img_mean, teacher.img_std,
        return_index=True,
    )
    fit_loader = make_loader(
        fit_dataset, args.batch_size, args.workers, True, args.seed + 100,
    )
    validation_loader = make_loader(
        validation_dataset, args.batch_size, args.workers, False, args.seed + 200,
    )
    token_fit_loader = make_loader(
        fit_dataset, args.batch_size, args.workers, False, args.seed + 300,
    )
    decision_loader = make_loader(
        decision_dataset, args.batch_size, args.workers, False, args.seed + 400,
    )

    model = MaskedRelationDLBM(
        feature_dim=teacher.embed_dim, token_dim=args.token_dim,
        state_dim=args.state_dim, heads=args.heads, patch_count=256,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, threshold=0.001,
        min_lr=1e-5,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    print(
        f'[{args.scene}] v001 boundary_train={len(boundary_train_paths)} '
        f'fit={len(fit_paths)} validation={len(validation_paths)} '
        f'decision={len(decision_paths)} batch={args.batch_size} '
        f'max_epochs={args.max_epochs} patience={args.patience} device={device}',
        flush=True,
    )
    history = []
    best_state = None
    best_validation = math.inf
    best_epoch, stale = 0, 0
    for epoch in range(1, args.max_epochs + 1):
        started = time.time()
        train = train_representation_epoch(
            model, fit_loader, teacher, ocbe, student, optimizer, scaler,
            device, amp, args.grad_clip, args.max_train_steps,
        )
        validation = evaluate_representation(
            model, validation_loader, teacher, ocbe, student, device, amp,
        )
        scheduler.step(validation['total'])
        row = {
            'epoch': epoch,
            'train': train,
            'validation': validation,
            'lr': optimizer.param_groups[0]['lr'],
            'seconds': time.time() - started,
        }
        history.append(row)
        threshold = best_validation * (1.0 - args.min_relative_improvement)
        if validation['total'] < threshold:
            best_validation = validation['total']
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        print(
            f'  ep{epoch:02d}/{args.max_epochs} '
            f'train={train["total"]:.6g} val={validation["total"]:.6g} '
            f'pred={validation["prediction"]:.6g} trans={validation["transition"]:.6g} '
            f'rel={validation["relation_residual_ratio"]:.3f} '
            f'var={validation["transition_token_variance"]:.3e} '
            f'lr={optimizer.param_groups[0]["lr"]:.2e} ({row["seconds"]:.0f}s)',
            flush=True,
        )
        if epoch >= args.min_epochs and stale >= args.patience:
            print(
                f'  early stop at epoch={epoch}; best={best_epoch} '
                f'val={best_validation:.6g}',
                flush=True,
            )
            break
    if best_state is None:
        raise RuntimeError('representation training never produced a valid best checkpoint')
    model.load_state_dict(best_state)
    model.eval()

    interventions = representation_interventions(
        model, validation_dataset, teacher, ocbe, student, device, amp,
        args.intervention_frames, args.seed + 500,
    )
    print(
        f'  intervention roll/full={interventions["roll_over_aligned"]:.3f} '
        f'no_relation/full={interventions["without_relation_over_full"]:.3f}',
        flush=True,
    )

    fit_moments = transition_moments(
        model, token_fit_loader, teacher, ocbe, student, device, amp,
    )
    validation_moments = transition_moments(
        model, validation_loader, teacher, ocbe, student, device, amp,
    )
    density = OrderedChangePointDensity(
        token_dim=args.token_dim, num_layers=12,
    ).to(device)
    fit_moments = moments_to_device(fit_moments, device)
    validation_moments = moments_to_device(validation_moments, device)
    density_history, density_best_epoch, density_best_validation = train_density(
        density, fit_moments, validation_moments, args,
    )
    print(
        f'  density epochs={len(density_history)} best={density_best_epoch} '
        f'validation_nll={density_best_validation:.6g}',
        flush=True,
    )

    ckpt_dir = args.out_root / 'ckpt'
    decision_dir = args.out_root / 'decision'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    decision_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f'{args.scene}.pth'
    checkpoint = {
        'schema_version': 1,
        'version': VERSION,
        'scene': args.scene,
        'model': model.state_dict(),
        'density': density.state_dict(),
        'model_config': {
            'feature_dim': teacher.embed_dim,
            'num_layers': 12,
            'token_dim': args.token_dim,
            'state_dim': args.state_dim,
            'heads': args.heads,
            'patch_count': 256,
        },
        'args': vars(args) | {
            'data_root': str(args.data_root),
            'split_root': str(args.split_root),
            'probe_dir': str(args.probe_dir),
            'out_root': str(args.out_root),
        },
        'probe': {
            'path': str(probe_path),
            'sha256': sha256_file(probe_path),
            'cfg': probe_cfg,
        },
        'manifests': {
            'boundary_train': {
                'path': str(train_manifest),
                'sha256': sha256_file(train_manifest),
                'frames': len(boundary_train_paths),
            },
            'boundary_decision': {
                'path': str(decision_manifest),
                'sha256': sha256_file(decision_manifest),
                'frames': len(decision_paths),
            },
        },
        'internal_boundary_train_split': {
            'fit_frames': len(fit_paths),
            'validation_frames': len(validation_paths),
            'per_subvideo': internal_records,
        },
        'representation_history': history,
        'representation_best_epoch': best_epoch,
        'representation_best_validation': best_validation,
        'representation_interventions': interventions,
        'density_history': density_history,
        'density_best_epoch': density_best_epoch,
        'density_best_validation': density_best_validation,
    }
    torch.save(checkpoint, checkpoint_path)

    policy, per_video, scene_logits = decision(
        model, density, decision_dataset, decision_loader,
        teacher, ocbe, student, device, amp,
    )
    decision_result = {
        'schema_version': 1,
        'version': VERSION,
        'scene': args.scene,
        'decision_role': (
            'frozen same-distribution scene inference; no train-decision gap is computed or used'
        ),
        'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': sha256_file(checkpoint_path),
        'manifest': {
            'path': str(decision_manifest),
            'sha256': sha256_file(decision_manifest),
            'frames': len(decision_paths),
        },
        'aggregation': 'mean frames within subvideo, then equal-weight mean across subvideos',
        'scene_logits': scene_logits,
        'policy': policy,
        'subvideos': per_video,
    }
    decision_path = decision_dir / f'{args.scene}.json'
    write_json(decision_path, decision_result)
    if device.type == 'cuda':
        decision_result['peak_gpu_gib'] = (
            torch.cuda.max_memory_allocated(device) / 2**30
        )
        write_json(decision_path, decision_result)
    print(
        f'  decision boundary=L{policy["boundary"]} taps={policy["selected_taps"]} '
        f'confidence={policy["confidence"]:.3f} '
        f'entropy={policy["normalized_entropy"]:.3f}',
        flush=True,
    )
    print(f'  saved {checkpoint_path}', flush=True)
    print(f'  saved {decision_path}', flush=True)


if __name__ == '__main__':
    main()
