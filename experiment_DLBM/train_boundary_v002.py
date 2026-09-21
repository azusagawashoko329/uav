"""Train one per-scene DLBM v002 from cached exact-suffix Probe responses."""
import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_DLBM.common import SCENES, probe_path, set_seed, write_json
from experiment_DLBM.dlbm_v001 import boundary_from_adjacent_gains
from experiment_DLBM.dlbm_v002 import ExactSuffixGainPredictor, exact_suffix_adjacent_gains
from experiment_DLBM.run_boundary_v001 import (PathFrameDataset, all_scene_paths, load_probe,
                                                split_video_balanced, within_frame_cutpaste)
from model import all_boundary_topk_scores


@torch.no_grad()
def cache_split(paths, teacher, ocbe, student, cfg, args, device):
    dataset = PathFrameDataset(paths, cfg['img'], teacher.img_mean, teacher.img_std)
    loader = DataLoader(
        dataset, batch_size=args.cache_bs, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == 'cuda', drop_last=False,
    )
    values = defaultdict(list)
    for images in loader:
        images = images.to(device, non_blocking=True)
        pseudo = within_frame_cutpaste(images, args.pseudo_min_area, args.pseudo_max_area)
        normal_taps = teacher(images)['patches']
        pseudo_taps = teacher(pseudo)['patches']
        normal_boundary, normal_input = all_boundary_topk_scores(
            ocbe, student, normal_taps, topk=args.topk,
        )
        pseudo_boundary, pseudo_input = all_boundary_topk_scores(
            ocbe, student, pseudo_taps, topk=args.topk,
        )
        values['normal_input'].append(normal_input.cpu())
        values['pseudo_input'].append(pseudo_input.cpu())
        values['normal_boundary'].append(normal_boundary.cpu())
        values['pseudo_boundary'].append(pseudo_boundary.cpu())
    result = {key: torch.cat(chunks) for key, chunks in values.items()}
    result['paths'] = list(paths)
    return result


def load_or_create_cache(scene, out_dir, teacher, ocbe, student, cfg, paths, videos, args, device):
    cache_path = out_dir / 'exact_suffix_cache.pth'
    expected = {
        'scene': scene, 'topk': args.topk, 'max_frames_per_video': args.max_frames_per_video,
        'seed': args.seed, 'pseudo_min_area': args.pseudo_min_area,
        'pseudo_max_area': args.pseudo_max_area, 'probe': str(probe_path(args.probe_dir, scene)),
    }
    if args.reuse_cache and cache_path.exists():
        cache = torch.load(cache_path, map_location='cpu')
        if cache.get('config') != expected:
            raise ValueError(f'cache config mismatch: {cache.get("config")} vs {expected}')
        return cache
    torch.manual_seed(args.seed + 3000)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed + 3000)
    cache = {
        'config': expected,
        'videos': videos,
        'support': cache_split(paths['support'], teacher, ocbe, student, cfg, args, device),
        'query': cache_split(paths['query'], teacher, ocbe, student, cfg, args, device),
    }
    torch.save(cache, cache_path)
    return cache


def balanced_indices(paths, count, rng):
    by_video = defaultdict(list)
    for index, path in enumerate(paths):
        by_video[str(Path(path).parent)].append(index)
    videos = sorted(by_video)
    return np.asarray([
        rng.choice(by_video[videos[int(rng.randint(len(videos)))]])
        for _ in range(count)
    ], dtype=np.int64)


def make_episodes(split, count, frames, seed):
    rng = np.random.RandomState(seed)
    return [balanced_indices(split['paths'], frames, rng) for _ in range(count)]


def episode_tensors(split, indices, device):
    index = torch.as_tensor(indices, dtype=torch.long)
    return tuple(split[key].index_select(0, index).to(device) for key in (
        'normal_input', 'pseudo_input', 'normal_boundary', 'pseudo_boundary',
    ))


def pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


@torch.no_grad()
def evaluate(model, split, episodes, device, shuffled=False, zeroed=False):
    model.eval()
    predictions, targets, boundaries = [], [], []
    permutation = torch.arange(11, -1, -1, device=device)
    for indices in episodes:
        normal_input, pseudo_input, normal_boundary, pseudo_boundary = episode_tensors(
            split, indices, device,
        )
        if shuffled:
            normal_input = normal_input.index_select(1, permutation)
            pseudo_input = pseudo_input.index_select(1, permutation)
        if zeroed:
            normal_input = torch.zeros_like(normal_input)
            pseudo_input = torch.zeros_like(pseudo_input)
        prediction = model(normal_input, pseudo_input)
        target = exact_suffix_adjacent_gains(normal_boundary, pseudo_boundary)
        predictions.append(prediction.cpu())
        targets.append(target.cpu())
        boundaries.append(boundary_from_adjacent_gains(prediction).boundary)
    predictions = torch.stack(predictions)
    targets = torch.stack(targets)
    mean_prediction = predictions.mean(0)
    mean_target = targets.mean(0)
    decision = boundary_from_adjacent_gains(mean_prediction)
    oracle = boundary_from_adjacent_gains(mean_target)
    return {
        'mse': float(F.mse_loss(predictions, targets)),
        'mae': float(F.l1_loss(predictions, targets)),
        'mean_gain_pearson': pearson(mean_prediction.tolist(), mean_target.tolist()),
        'boundary': decision.boundary,
        'oracle_boundary': oracle.boundary,
        'boundary_histogram': {str(i): boundaries.count(i) for i in range(12)},
        'mean_predicted_gains': [float(value) for value in mean_prediction],
        'mean_target_gains': [float(value) for value in mean_target],
        'predicted_cumulative_benefits': [float(value) for value in decision.cumulative_benefits],
        'target_cumulative_benefits': [float(value) for value in oracle.cumulative_benefits],
        '_predictions': predictions,
    }


def public_metrics(metrics):
    return {key: value for key, value in metrics.items() if not key.startswith('_')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, choices=sorted(SCENES))
    parser.add_argument('--probe_dir', default='experiment_DLBM/run/probe/ckpt')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--max_frames_per_video', type=int, default=48)
    parser.add_argument('--cache_bs', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pseudo_min_area', type=float, default=0.02)
    parser.add_argument('--pseudo_max_area', type=float, default=0.15)
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--layer_depth', type=int, default=1)
    parser.add_argument('--pair_depth', type=int, default=1)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--steps_per_epoch', type=int, default=64)
    parser.add_argument('--episode_frames', type=int, default=8)
    parser.add_argument('--validation_episodes', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--milestones', type=int, nargs='+', default=[5, 10, 20, 30])
    parser.add_argument('--reuse_cache', action='store_true')
    args = parser.parse_args()

    if max(args.milestones) > args.epochs:
        raise ValueError('milestones cannot exceed epochs')
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    teacher, ocbe, student, cfg = load_probe(args.scene, args.probe_dir, device)
    if args.topk != int(cfg.get('k', args.topk)):
        raise ValueError(f"top-k mismatch: boundary={args.topk}, Probe={cfg.get('k')}")
    paths, videos = split_video_balanced(
        all_scene_paths(args.scene), args.max_frames_per_video, args.seed,
    )
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    print(f"[{args.scene}] cache support={len(paths['support'])}/{len(videos['support'])} videos "
          f"query={len(paths['query'])}/{len(videos['query'])} videos", flush=True)
    cache = load_or_create_cache(
        args.scene, out_dir, teacher, ocbe, student, cfg, paths, videos, args, device,
    )
    cache_peak = (torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0.0)
    del teacher, ocbe, student
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    model = ExactSuffixGainPredictor(
        num_layers=12, model_dim=args.model_dim, layer_depth=args.layer_depth,
        pair_depth=args.pair_depth, heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    validation_episodes = make_episodes(
        cache['query'], args.validation_episodes, args.episode_frames, args.seed + 5000,
    )
    history = []
    milestone_metrics = {}
    train_rng = np.random.RandomState(args.seed + 4000)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, grad_norms = [], []
        for _ in range(args.steps_per_epoch):
            indices = balanced_indices(cache['support']['paths'], args.episode_frames, train_rng)
            normal_input, pseudo_input, normal_boundary, pseudo_boundary = episode_tensors(
                cache['support'], indices, device,
            )
            prediction = model(normal_input, pseudo_input)
            target = exact_suffix_adjacent_gains(normal_boundary, pseudo_boundary).detach()
            loss = F.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss epoch={epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f'non-finite grad epoch={epoch}')
            optimizer.step()
            losses.append(float(loss.detach()))
            grad_norms.append(float(grad_norm.detach()))
        row = {'epoch': epoch, 'train_mse': float(np.mean(losses)),
               'grad_norm': float(np.mean(grad_norms))}
        if epoch in args.milestones:
            metrics = evaluate(model, cache['query'], validation_episodes, device)
            row['validation'] = public_metrics(metrics)
            milestone_metrics[str(epoch)] = public_metrics(metrics)
            torch.save({'dlbm': model.state_dict(), 'args': vars(args), 'epoch': epoch,
                        'validation': public_metrics(metrics)}, checkpoint_dir / f'ep{epoch:02d}.pth')
            print(f"  ep{epoch:02d}/{args.epochs} train={row['train_mse']:.3e} "
                  f"val={metrics['mse']:.3e} corr={metrics['mean_gain_pearson']} "
                  f"boundary=L{metrics['boundary']} oracle=L{metrics['oracle_boundary']}", flush=True)
        elif epoch == 1 or epoch % 5 == 0:
            print(f"  ep{epoch:02d}/{args.epochs} train={row['train_mse']:.3e}", flush=True)
        history.append(row)

    final_metrics = evaluate(model, cache['query'], validation_episodes, device)
    shuffled = evaluate(model, cache['query'], validation_episodes, device, shuffled=True)
    zeroed = evaluate(model, cache['query'], validation_episodes, device, zeroed=True)
    input_sensitivity = {
        'reverse_layer_mean_abs_prediction_change': float(
            (final_metrics['_predictions'] - shuffled['_predictions']).abs().mean()),
        'zero_input_mean_abs_prediction_change': float(
            (final_metrics['_predictions'] - zeroed['_predictions']).abs().mean()),
        'reverse_layer_boundary': shuffled['boundary'],
        'zero_input_boundary': zeroed['boundary'],
    }
    result = {
        'format_version': 2,
        'scene': args.scene,
        'method': 'DLBM_v002_exact_suffix_gain',
        'status': 'candidate_pending_cross_scene_review',
        'probe_checkpoint': str(probe_path(args.probe_dir, args.scene)),
        'probe_cfg': cfg,
        'args': vars(args),
        'cache': {'support_frames': len(cache['support']['paths']),
                  'query_frames': len(cache['query']['paths']), 'videos': cache['videos']},
        'milestones': milestone_metrics,
        'history': history,
        'final': public_metrics(final_metrics),
        'input_sensitivity': input_sensitivity,
        'boundary': final_metrics['boundary'],
        'oracle_boundary': final_metrics['oracle_boundary'],
        'selected_taps': list(range(final_metrics['boundary'], 12)),
        'cache_peak_gpu_gib': cache_peak,
        'model_parameters': sum(parameter.numel() for parameter in model.parameters()),
    }
    torch.save({'dlbm': model.state_dict(), 'args': vars(args), 'result': result}, out_dir / 'dlbm_final.pth')
    write_json(out_dir / 'result.json', result)
    print(f"  final boundary=L{result['boundary']} oracle=L{result['oracle_boundary']} "
          f"sensitivity={input_sensitivity['reverse_layer_mean_abs_prediction_change']:.3e}", flush=True)
    print(f'  cache_peak_gpu_memory={cache_peak:.2f} GiB', flush=True)
    print(f'saved {out_dir}', flush=True)


if __name__ == '__main__':
    main()
