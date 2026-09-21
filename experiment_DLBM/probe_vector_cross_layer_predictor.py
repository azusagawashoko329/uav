"""Validate channel-preserving Teacher/Student cross-layer prediction."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_DLBM.common import SCENES, set_seed
from experiment_DLBM.run_boundary_v001 import (
    PathFrameDataset, all_scene_paths, load_probe, split_video_balanced,
    within_frame_cutpaste,
)
from experiment_DLBM.train_boundary_v006 import _loader
from experiment_DLBM.vector_cross_layer_predictor import (
    VectorCrossLayerPredictor, vector_prediction_loss, vector_reconstruction_loss,
)


@torch.no_grad()
def cache_patch_pairs(paths, teacher, ocbe, student, cfg, args, device):
    values = defaultdict(lambda: defaultdict(list))
    torch.manual_seed(args.seed + 61000)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed + 61000)
    for images in _loader(paths, teacher, cfg, args.cache_bs, args.workers):
        images = images.to(device, non_blocking=True)
        pseudo = within_frame_cutpaste(
            images, args.pseudo_min_area, args.pseudo_max_area,
        )
        for branch, view in (('normal', images), ('pseudo', pseudo)):
            teacher_taps = teacher(view)['patches']
            student_taps = student(ocbe(teacher_taps))
            values[branch]['teacher'].append(
                torch.stack(teacher_taps, dim=1).half().cpu(),
            )
            values[branch]['student'].append(
                torch.stack(student_taps, dim=1).half().cpu(),
            )
    return {
        branch: {name: torch.cat(chunks) for name, chunks in pair.items()}
        for branch, pair in values.items()
    } | {'paths': list(paths)}


def independent_pair_roll(teacher, student, seed):
    """Roll T/S together within layers, but independently across layers."""
    generator = torch.Generator(device=teacher.device).manual_seed(int(seed))
    side = int(round(teacher.shape[2] ** 0.5))
    outputs = [teacher.clone(), student.clone()]
    for frame in range(len(teacher)):
        for layer in range(teacher.shape[1]):
            dy = int(torch.randint(0, side, (), generator=generator, device=teacher.device))
            dx = int(torch.randint(0, side, (), generator=generator, device=teacher.device))
            for output in outputs:
                grid = output[frame, layer].reshape(side, side, output.shape[-1])
                output[frame, layer] = torch.roll(grid, (dy, dx), dims=(0, 1)).flatten(0, 1)
    return tuple(outputs)


@torch.no_grad()
def evaluate(model, cache, device, batch_size, seed):
    model.eval(); result = {}
    for branch in ('normal', 'pseudo'):
        aligned_teacher = cache[branch]['teacher']
        aligned_student = cache[branch]['student']
        totals = defaultdict(float); count = 0
        for start in range(0, len(aligned_teacher), batch_size):
            teacher = aligned_teacher[start:start + batch_size].to(device)
            student = aligned_student[start:start + batch_size].to(device)
            rolled = independent_pair_roll(teacher, student, seed + start)
            for name, pair in (('aligned', (teacher, student)), ('roll', rolled)):
                loss, output = vector_prediction_loss(model, *pair)
                totals[f'{name}_loss'] += float(loss) * len(teacher)
                totals[f'{name}_cosine'] += float(
                    torch.nn.functional.cosine_similarity(
                        output['prediction'], output['target'], dim=-1,
                    ).mean()
                ) * len(teacher)
                if name == 'aligned':
                    totals['zero_loss'] += 1.0 * len(teacher)
            count += len(teacher)
        for key, value in totals.items():
            result[f'{branch}_{key}'] = value / count
    result['normal_roll_ratio'] = result['normal_roll_loss'] / max(
        result['normal_aligned_loss'], 1e-12,
    )
    result['pseudo_over_normal'] = result['pseudo_aligned_loss'] / max(
        result['normal_aligned_loss'], 1e-12,
    )
    result['normal_over_zero'] = result['normal_aligned_loss'] / max(
        result['normal_zero_loss'], 1e-12,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, choices=sorted(SCENES))
    parser.add_argument('--probe_dir', default='experiment_DLBM/run/probe/ckpt')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_frames_per_video', type=int, default=48)
    parser.add_argument('--cache_bs', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pseudo_min_area', type=float, default=0.02)
    parser.add_argument('--pseudo_max_area', type=float, default=0.15)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--encoder_epochs', type=int, default=5)
    parser.add_argument('--encoder_steps', type=int, default=44)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps_per_epoch', type=int, default=44)
    parser.add_argument('--frame_batch', type=int, default=4)
    parser.add_argument('--eval_bs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    teacher, ocbe, student, cfg = load_probe(args.scene, args.probe_dir, device)
    paths, videos = split_video_balanced(
        all_scene_paths(args.scene), args.max_frames_per_video, args.seed,
    )
    support = cache_patch_pairs(paths['support'], teacher, ocbe, student, cfg, args, device)
    query = cache_patch_pairs(paths['query'], teacher, ocbe, student, cfg, args, device)
    del teacher, ocbe, student
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    feature_dim = support['normal']['teacher'].shape[-1]
    model = VectorCrossLayerPredictor(
        feature_dim=feature_dim, channels=args.channels,
    ).to(device)
    rng = np.random.RandomState(args.seed + 62000)
    encoder_parameters = (
        list(model.pair_projection.parameters()) + list(model.spatial_stem.parameters())
        + list(model.decoder.parameters())
    )
    optimizer = torch.optim.AdamW(encoder_parameters, lr=args.lr, weight_decay=args.wd)
    encoder_history = []
    for epoch in range(1, args.encoder_epochs + 1):
        model.train(); losses = []
        for _ in range(args.encoder_steps):
            index = torch.as_tensor(
                rng.randint(0, len(support['normal']['teacher']), size=args.frame_batch),
                dtype=torch.long,
            )
            teacher = support['normal']['teacher'].index_select(0, index).to(device)
            student = support['normal']['student'].index_select(0, index).to(device)
            loss = vector_reconstruction_loss(model, teacher, student)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(encoder_parameters, args.grad_clip)
            if not torch.isfinite(loss) or not torch.isfinite(gradient):
                raise FloatingPointError('non-finite relation autoencoder value')
            optimizer.step(); losses.append(float(loss.detach()))
        encoder_history.append({'epoch': epoch, 'loss': float(np.mean(losses))})
        if epoch in {1, args.encoder_epochs}:
            print(f'[{args.scene}] encoder ep{epoch:02d} loss={np.mean(losses):.6g}', flush=True)
    for parameter in encoder_parameters:
        parameter.requires_grad_(False)
    recurrent_parameters = (
        list(model.recurrent.parameters()) + [model.observed_layer_embedding,
                                               model.target_layer_embedding]
    )
    optimizer = torch.optim.AdamW(recurrent_parameters, lr=args.lr, weight_decay=args.wd)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses, gradients = [], []
        for _ in range(args.steps_per_epoch):
            index = torch.as_tensor(
                rng.randint(0, len(support['normal']['teacher']), size=args.frame_batch),
                dtype=torch.long,
            )
            teacher = support['normal']['teacher'].index_select(0, index).to(device)
            student = support['normal']['student'].index_select(0, index).to(device)
            loss, _ = vector_prediction_loss(model, teacher, student)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(recurrent_parameters, args.grad_clip)
            if not torch.isfinite(loss) or not torch.isfinite(gradient):
                raise FloatingPointError('non-finite vector predictor training value')
            optimizer.step()
            losses.append(float(loss.detach())); gradients.append(float(gradient.detach()))
        row = {'epoch': epoch, 'train_loss': float(np.mean(losses)),
               'grad_norm': float(np.mean(gradients))}
        if epoch in {1, 5, args.epochs}:
            row['query'] = evaluate(model, query, device, args.eval_bs, args.seed + 63000)
            print(
                f'[{args.scene}] ep{epoch:02d} loss={row["train_loss"]:.6g} '
                f'query={row["query"]["normal_aligned_loss"]:.6g} '
                f'roll={row["query"]["normal_roll_ratio"]:.3f} '
                f'p/n={row["query"]["pseudo_over_normal"]:.3f}', flush=True,
            )
        history.append(row)
    result = {
        'scene': args.scene, 'method': 'channel_preserving_vector_cross_layer_predictor',
        'args': vars(args),
        'split': {'support_frames': len(support['paths']), 'query_frames': len(query['paths']),
                  'videos': videos},
        'encoder_history': encoder_history, 'history': history,
        'final_query': evaluate(model, query, device, args.eval_bs, args.seed + 63000),
        'parameters': sum(p.numel() for p in model.parameters()),
        'peak_gpu_gib': (torch.cuda.max_memory_allocated(device) / 2**30
                         if device.type == 'cuda' else 0.0),
    }
    torch.save({'predictor': model.state_dict(), 'result': result}, out_dir / 'predictor.pth')
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
