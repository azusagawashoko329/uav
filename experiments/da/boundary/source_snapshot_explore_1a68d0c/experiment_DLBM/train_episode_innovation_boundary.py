"""Train a scene-level episode DLBM on a frozen cross-layer predictor."""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_DLBM.common import SCENES, probe_path, set_seed
from experiment_DLBM.cross_layer_predictor import CrossLayerPredictor
from experiment_DLBM.episode_innovation_boundary import (
    SCALE_TRANSITIONS, EpisodeInnovationBoundary, boundary_from_scale_margins,
    permute_added_layers,
)
from experiment_DLBM.probe_cross_layer_predictor import (
    evaluate as evaluate_predictor, independent_layer_roll, prediction_loss,
)
from experiment_DLBM.run_boundary_v001 import (
    all_scene_paths, load_probe, split_video_balanced,
)
from experiment_DLBM.train_boundary_v006 import cache_relation_maps
from experiment_DLBM.train_boundary_v007 import (
    derangement, sample_video_batch, video_indices,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def train_predictor(model, support, args, device):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.predictor_lr, weight_decay=args.wd,
    )
    rng = np.random.RandomState(args.seed + 41000)
    history = []
    for epoch in range(1, args.predictor_epochs + 1):
        model.train(); losses = []
        for _ in range(args.predictor_steps):
            index = torch.as_tensor(
                rng.randint(0, len(support['normal']), size=args.predictor_batch),
                dtype=torch.long,
            )
            maps = support['normal'].index_select(0, index).to(
                device=device, dtype=torch.float32,
            )
            loss, _ = prediction_loss(model, maps)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(gradient):
                raise FloatingPointError('non-finite predictor gradient')
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({'epoch': epoch, 'loss': float(np.mean(losses))})
    model.eval().requires_grad_(False)
    return history


def opening_pair(predictor, normal, pseudo, base, candidate):
    with torch.no_grad():
        return (
            predictor.opening_features(normal, base, candidate),
            predictor.opening_features(pseudo, base, candidate),
        )


def contrastive_episode_loss(predictor, head, normal, pseudo, scale, null_repeats):
    base, candidate = SCALE_TRANSITIONS[scale]
    real_normal, real_pseudo = opening_pair(
        predictor, normal, pseudo, base, candidate,
    )
    real = head.scores(real_normal, real_pseudo)
    losses, margins = [], defaultdict(list)
    for _ in range(null_repeats):
        permutation = derangement(len(normal), normal.device)
        null_normal = permute_added_layers(normal, base, candidate, permutation)
        null_pseudo = permute_added_layers(pseudo, base, candidate, permutation)
        null_open_normal, null_open_pseudo = opening_pair(
            predictor, null_normal, null_pseudo, base, candidate,
        )
        null = head.scores(null_open_normal, null_open_pseudo)
        for branch in ('normal', 'pseudo', 'joint'):
            # Absolute balanced real/null calibration makes evidence comparable
            # across D/M/S.  A ranking-only loss leaves each scale's logit norm
            # unidentified and therefore cannot support scale competition.
            margin = torch.sigmoid(real[branch]) - torch.sigmoid(null[branch])
            margins[branch].append(margin.detach())
            losses.append(0.5 * (
                F.softplus(-real[branch]) + F.softplus(null[branch])
            ))
    return torch.stack(losses).mean(), {
        branch: float(torch.stack(values).mean()) for branch, values in margins.items()
    }


def deterministic_derangement(size, device, seed):
    if size < 2:
        raise ValueError('derangement needs at least two frames')
    generator = torch.Generator(device=device).manual_seed(int(seed))
    original = torch.arange(size, device=device)
    for _ in range(32):
        permutation = torch.randperm(size, generator=generator, device=device)
        if not torch.eq(permutation, original).any():
            return permutation
    return original.roll(1)


def train_head(predictor, head, support, groups, args, device,
               query=None, query_groups=None):
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=args.wd)
    rng = np.random.RandomState(args.seed + 42000)
    history, milestones = [], {}
    scales = tuple(SCALE_TRANSITIONS)
    for epoch in range(1, args.head_epochs + 1):
        head.train(); totals = defaultdict(list)
        for step in range(args.head_steps):
            scale = scales[step % len(scales)]
            (normal, pseudo), _ = sample_video_batch(
                support, groups, args.episode_frames, rng, device,
            )
            if len(normal) < 2:
                raise ValueError('episode needs at least two frames')
            loss, margins = contrastive_episode_loss(
                predictor, head, normal, pseudo, scale, args.train_null_repeats,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite episode boundary loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            if not torch.isfinite(gradient):
                raise FloatingPointError('non-finite episode boundary gradient')
            optimizer.step()
            totals['loss'].append(float(loss.detach()))
            totals['grad_norm'].append(float(gradient.detach()))
            for branch, value in margins.items():
                totals[f'{scale}_{branch}_margin'].append(value)
        row = {'epoch': epoch, **{
            key: float(np.mean(values)) for key, values in totals.items()
        }}
        if epoch in args.milestones and query is not None:
            metrics = evaluate_policy(
                predictor, head, query, query_groups, args, device,
                null_repeats=args.milestone_null_repeats,
            )
            row['query'] = metrics
            milestones[str(epoch)] = metrics
            print(
                f'  head ep{epoch:02d} loss={row["loss"]:.4f} '
                f'query=L{metrics["boundary"]} '
                f'margins={[round(x, 4) for x in metrics["joint_margins"]]}',
                flush=True,
            )
        history.append(row)
    head.eval()
    return history, milestones


@torch.no_grad()
def evaluate_scale(predictor, head, cache, groups, scale, args, device, null_repeats):
    base, candidate = SCALE_TRANSITIONS[scale]
    rows = []
    scale_offset = {'D': 0, 'M': 100000, 'S': 200000}[scale]
    for video_index, video in enumerate(sorted(groups)):
        index = torch.as_tensor(groups[video], dtype=torch.long)
        normal = cache['normal'].index_select(0, index).to(device=device, dtype=torch.float32)
        pseudo = cache['pseudo'].index_select(0, index).to(device=device, dtype=torch.float32)
        if len(normal) < 2:
            continue
        real_open = opening_pair(predictor, normal, pseudo, base, candidate)
        real = head.scores(*real_open)
        null_values = defaultdict(list)
        for repeat in range(null_repeats):
            permutation = deterministic_derangement(
                len(normal), device,
                args.seed + 45000 + scale_offset + 1000 * video_index + repeat,
            )
            null_normal = permute_added_layers(normal, base, candidate, permutation)
            null_pseudo = permute_added_layers(pseudo, base, candidate, permutation)
            null = head.scores(*opening_pair(
                predictor, null_normal, null_pseudo, base, candidate,
            ))
            for branch, value in null.items():
                null_values[branch].append(value)
        row = {'video': video}
        for branch in ('normal', 'pseudo', 'joint'):
            null_mean = torch.stack(null_values[branch]).mean()
            row[f'{branch}_real'] = float(real[branch])
            row[f'{branch}_null'] = float(null_mean)
            row[f'{branch}_raw_margin'] = float(real[branch] - null_mean)
            row[f'{branch}_margin'] = float(
                torch.sigmoid(real[branch])
                - torch.stack([torch.sigmoid(x) for x in null_values[branch]]).mean()
            )
        rows.append(row)
    if not rows:
        raise ValueError(f'no valid video episodes for scale {scale}')
    result = {'videos': len(rows), 'video_rows': rows}
    for branch in ('normal', 'pseudo', 'joint'):
        values = np.asarray([row[f'{branch}_margin'] for row in rows])
        result[f'{branch}_margin'] = float(values.mean())
        result[f'{branch}_margin_std'] = float(values.std())
        result[f'{branch}_positive_fraction'] = float((values > 0).mean())
    return result


@torch.no_grad()
def evaluate_policy(predictor, head, cache, groups, args, device, null_repeats=None):
    predictor.eval(); head.eval()
    repeats = args.eval_null_repeats if null_repeats is None else int(null_repeats)
    scales = {
        scale: evaluate_scale(
            predictor, head, cache, groups, scale, args, device, repeats,
        ) for scale in SCALE_TRANSITIONS
    }
    margins = {
        scale: torch.tensor(scales[scale]['joint_margin']) for scale in SCALE_TRANSITIONS
    }
    boundary, values = boundary_from_scale_margins(margins)
    per_video_winners = []
    videos = sorted(set.intersection(*[
        {row['video'] for row in scales[scale]['video_rows']}
        for scale in SCALE_TRANSITIONS
    ]))
    for video in videos:
        video_margins = {}
        for scale in SCALE_TRANSITIONS:
            row = next(x for x in scales[scale]['video_rows'] if x['video'] == video)
            video_margins[scale] = row['joint_margin']
        winner = max(video_margins, key=video_margins.get)
        per_video_winners.append({'video': video, 'scale': winner,
                                  'boundary': {'D': 8, 'M': 4, 'S': 0}[winner],
                                  'margins': video_margins})
    return {
        'boundary': boundary, 'selected_taps': list(range(boundary, 12)),
        'joint_margins': [float(value) for value in values],
        'scale_order': list(SCALE_TRANSITIONS), 'scales': scales,
        'video_winners': per_video_winners,
    }


def rolled_cache(cache, seed):
    return {
        'paths': cache['paths'],
        'normal': independent_layer_roll(cache['normal'].float(), seed),
        'pseudo': independent_layer_roll(cache['pseudo'].float(), seed + 1),
    }


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
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--pool_seeds', type=int, default=2)
    parser.add_argument('--predictor_epochs', type=int, default=10)
    parser.add_argument('--predictor_steps', type=int, default=44)
    parser.add_argument('--predictor_batch', type=int, default=24)
    parser.add_argument('--head_epochs', type=int, default=30)
    parser.add_argument('--head_steps', type=int, default=45)
    parser.add_argument('--episode_frames', type=int, default=48)
    parser.add_argument('--train_null_repeats', type=int, default=1)
    parser.add_argument('--eval_null_repeats', type=int, default=16)
    parser.add_argument('--milestone_null_repeats', type=int, default=4)
    parser.add_argument('--predictor_lr', type=float, default=1e-3)
    parser.add_argument('--head_lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--milestones', type=int, nargs='+', default=[5, 10, 20, 30])
    args = parser.parse_args()
    if max(args.milestones) > args.head_epochs:
        raise ValueError('milestones cannot exceed head epochs')
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = probe_path(args.probe_dir, args.scene)
    probe_hash_before = file_sha256(checkpoint)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    teacher, ocbe, student, cfg = load_probe(args.scene, args.probe_dir, device)
    if args.topk != int(cfg.get('k', args.topk)):
        raise ValueError('DLBM and Probe top-k must match')
    paths, videos = split_video_balanced(
        all_scene_paths(args.scene), args.max_frames_per_video, args.seed,
    )
    support = cache_relation_maps(paths['support'], teacher, ocbe, student, cfg, args, device)
    query = cache_relation_maps(paths['query'], teacher, ocbe, student, cfg, args, device)
    del teacher, ocbe, student
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    support_groups = video_indices(support['paths'])
    query_groups = video_indices(query['paths'])
    predictor = CrossLayerPredictor(channels=args.channels).to(device)
    predictor_history = train_predictor(predictor, support, args, device)
    predictor_query = evaluate_predictor(
        predictor, query, device, args.predictor_batch, args.seed + 43000,
    )
    print(
        f'[{args.scene}] predictor={predictor_query["normal_aligned_smooth_l1"]:.6g} '
        f'roll={predictor_query["normal_roll_loss_ratio"]:.3f}', flush=True,
    )
    head = EpisodeInnovationBoundary(
        predictor_channels=args.channels, topk=args.topk, model_dim=args.model_dim,
        heads=args.heads, pool_seeds=args.pool_seeds,
    ).to(device)
    head_history, milestones = train_head(
        predictor, head, support, support_groups, args, device,
        query=query, query_groups=query_groups,
    )
    query_policy = evaluate_policy(predictor, head, query, query_groups, args, device)
    support_policy = evaluate_policy(predictor, head, support, support_groups, args, device)
    shifted_policy = evaluate_policy(
        predictor, head, rolled_cache(query, args.seed + 44000), query_groups,
        args, device,
    )
    predictor_after = evaluate_predictor(
        predictor, query, device, args.predictor_batch, args.seed + 43000,
    )
    probe_hash_after = file_sha256(checkpoint)
    if probe_hash_before != probe_hash_after:
        raise RuntimeError('Probe checkpoint changed during boundary training')
    result = {
        'format_version': 9, 'scene': args.scene,
        'method': 'frozen_predictive_innovation_episode_scale_competition',
        'probe_checkpoint': str(checkpoint), 'probe_sha256': probe_hash_before,
        'probe_mutated': False, 'args': vars(args),
        'split': {'support_frames': len(support['paths']), 'query_frames': len(query['paths']),
                  'videos': videos},
        'predictor_history': predictor_history, 'predictor_query': predictor_query,
        'predictor_query_after_head': predictor_after,
        'head_history': head_history, 'milestones': milestones,
        'query_policy': query_policy, 'support_policy': support_policy,
        'shifted_query_policy': shifted_policy,
        'predictor_parameters': sum(p.numel() for p in predictor.parameters()),
        'head_parameters': sum(p.numel() for p in head.parameters()),
        'peak_gpu_gib': (torch.cuda.max_memory_allocated(device) / 2**30
                         if device.type == 'cuda' else 0.0),
    }
    torch.save({'predictor': predictor.state_dict(), 'head': head.state_dict(),
                'result': result}, out_dir / 'model.pth')
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(
        f'[{args.scene}] query=L{query_policy["boundary"]} '
        f'support=L{support_policy["boundary"]} shifted=L{shifted_policy["boundary"]}',
        flush=True,
    )


if __name__ == '__main__':
    main()
