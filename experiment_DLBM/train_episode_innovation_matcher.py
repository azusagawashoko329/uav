"""Train the scene-level same-frame predictive-innovation matcher."""
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

from experiment_DLBM.common import SCENES, probe_path, set_seed
from experiment_DLBM.cross_layer_predictor import CrossLayerPredictor
from experiment_DLBM.episode_innovation_boundary import (
    SCALE_TRANSITIONS, permute_added_layers,
)
from experiment_DLBM.episode_innovation_matcher import (
    EpisodeInnovationMatcher, boundary_from_retrieval, retrieval_metrics,
    symmetric_retrieval_loss,
)
from experiment_DLBM.probe_cross_layer_predictor import (
    evaluate as evaluate_predictor, prediction_loss,
)
from experiment_DLBM.run_boundary_v001 import (
    all_scene_paths, load_probe, split_video_balanced,
)
from experiment_DLBM.train_boundary_v006 import cache_relation_maps
from experiment_DLBM.train_boundary_v007 import sample_video_batch, video_indices
from experiment_DLBM.train_episode_innovation_boundary import (
    deterministic_derangement, file_sha256, rolled_cache, train_predictor,
)


def opening(predictor, maps, base, candidate):
    with torch.no_grad():
        return predictor.opening_features(maps, base, candidate)


def train_matcher(predictor, matcher, support, groups, args, device,
                  query=None, query_groups=None):
    optimizer = torch.optim.AdamW(
        matcher.parameters(), lr=args.matcher_lr, weight_decay=args.wd,
    )
    rng = np.random.RandomState(args.seed + 51000)
    scales = tuple(SCALE_TRANSITIONS)
    history, milestones = [], {}
    for epoch in range(1, args.matcher_epochs + 1):
        matcher.train(); totals = defaultdict(list)
        for step in range(args.matcher_steps):
            scale = scales[step % len(scales)]
            base, candidate = SCALE_TRANSITIONS[scale]
            (normal, pseudo), _ = sample_video_batch(
                support, groups, args.episode_frames, rng, device,
            )
            logits_normal = matcher.logits(opening(predictor, normal, base, candidate))
            logits_pseudo = matcher.logits(opening(predictor, pseudo, base, candidate))
            loss_normal = symmetric_retrieval_loss(logits_normal)
            loss_pseudo = symmetric_retrieval_loss(logits_pseudo)
            loss = 0.5 * (loss_normal + loss_pseudo)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite matcher loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(matcher.parameters(), args.grad_clip)
            if not torch.isfinite(gradient):
                raise FloatingPointError('non-finite matcher gradient')
            optimizer.step()
            totals['loss'].append(float(loss.detach()))
            totals[f'{scale}_normal_loss'].append(float(loss_normal.detach()))
            totals[f'{scale}_pseudo_loss'].append(float(loss_pseudo.detach()))
            totals['grad_norm'].append(float(gradient.detach()))
        row = {'epoch': epoch, **{
            key: float(np.mean(values)) for key, values in totals.items()
        }}
        if epoch in args.milestones and query is not None:
            policy = evaluate_policy(predictor, matcher, query, query_groups, args, device)
            row['query'] = policy
            milestones[str(epoch)] = policy
            print(
                f'  matcher ep{epoch:02d} loss={row["loss"]:.4f} '
                f'query=L{policy["boundary"]} '
                f'mrr={[round(x, 4) for x in policy["evidence"]]}', flush=True,
            )
        history.append(row)
    matcher.eval()
    return history, milestones


@torch.no_grad()
def evaluate_scale(predictor, matcher, cache, groups, scale, args, device,
                   permuted=False):
    base, candidate = SCALE_TRANSITIONS[scale]
    rows = []
    scale_offset = {'D': 0, 'M': 100000, 'S': 200000}[scale]
    for video_index, video in enumerate(sorted(groups)):
        index = torch.as_tensor(groups[video], dtype=torch.long)
        branch_metrics = {}
        for branch_index, branch in enumerate(('normal', 'pseudo')):
            maps = cache[branch].index_select(0, index).to(
                device=device, dtype=torch.float32,
            )
            if len(maps) < 2:
                continue
            if permuted:
                permutation = deterministic_derangement(
                    len(maps), device,
                    args.seed + 52000 + scale_offset + 1000 * video_index + branch_index,
                )
                maps = permute_added_layers(maps, base, candidate, permutation)
            logits = matcher.logits(opening(predictor, maps, base, candidate))
            branch_metrics[branch] = retrieval_metrics(logits)
        if set(branch_metrics) != {'normal', 'pseudo'}:
            continue
        rows.append({
            'video': video, 'normal': branch_metrics['normal'],
            'pseudo': branch_metrics['pseudo'],
            'joint_mrr': 0.5 * (
                branch_metrics['normal']['mrr'] + branch_metrics['pseudo']['mrr']
            ),
        })
    if not rows:
        raise ValueError(f'no valid retrieval episode for scale {scale}')
    result = {'videos': len(rows), 'video_rows': rows}
    for branch in ('normal', 'pseudo'):
        for metric in ('loss', 'top1', 'mrr', 'normalized_information'):
            values = [row[branch][metric] for row in rows]
            result[f'{branch}_{metric}'] = float(np.mean(values))
            result[f'{branch}_{metric}_std'] = float(np.std(values))
    result['joint_mrr'] = float(np.mean([row['joint_mrr'] for row in rows]))
    return result


@torch.no_grad()
def evaluate_policy(predictor, matcher, cache, groups, args, device, permuted=False):
    predictor.eval(); matcher.eval()
    scales = {
        scale: evaluate_scale(
            predictor, matcher, cache, groups, scale, args, device, permuted=permuted,
        ) for scale in SCALE_TRANSITIONS
    }
    evidence = {scale: scales[scale]['joint_mrr'] for scale in SCALE_TRANSITIONS}
    boundary, values = boundary_from_retrieval(evidence)
    video_winners = []
    videos = sorted(set.intersection(*[
        {row['video'] for row in scales[scale]['video_rows']}
        for scale in SCALE_TRANSITIONS
    ]))
    for video in videos:
        scores = {}
        for scale in SCALE_TRANSITIONS:
            row = next(x for x in scales[scale]['video_rows'] if x['video'] == video)
            scores[scale] = row['joint_mrr']
        winner = max(scores, key=scores.get)
        video_winners.append({
            'video': video, 'scale': winner,
            'boundary': {'D': 8, 'M': 4, 'S': 0}[winner], 'evidence': scores,
        })
    return {
        'boundary': boundary, 'selected_taps': list(range(boundary, 12)),
        'scale_order': list(SCALE_TRANSITIONS),
        'evidence': [float(value) for value in values], 'scales': scales,
        'video_winners': video_winners, 'permuted_opening': bool(permuted),
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
    parser.add_argument('--predictor_epochs', type=int, default=10)
    parser.add_argument('--predictor_steps', type=int, default=44)
    parser.add_argument('--predictor_batch', type=int, default=24)
    parser.add_argument('--matcher_epochs', type=int, default=30)
    parser.add_argument('--matcher_steps', type=int, default=45)
    parser.add_argument('--episode_frames', type=int, default=48)
    parser.add_argument('--predictor_lr', type=float, default=1e-3)
    parser.add_argument('--matcher_lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--milestones', type=int, nargs='+', default=[5, 10, 20, 30])
    args = parser.parse_args()
    if max(args.milestones) > args.matcher_epochs:
        raise ValueError('milestones cannot exceed matcher epochs')
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = probe_path(args.probe_dir, args.scene)
    probe_hash_before = file_sha256(checkpoint)
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
    support_groups, query_groups = video_indices(support['paths']), video_indices(query['paths'])
    predictor = CrossLayerPredictor(channels=args.channels).to(device)
    predictor_history = train_predictor(predictor, support, args, device)
    predictor_query = evaluate_predictor(
        predictor, query, device, args.predictor_batch, args.seed + 53000,
    )
    print(
        f'[{args.scene}] predictor={predictor_query["normal_aligned_smooth_l1"]:.6g} '
        f'roll={predictor_query["normal_roll_loss_ratio"]:.3f}', flush=True,
    )
    matcher = EpisodeInnovationMatcher(
        predictor_channels=args.channels, topk=args.topk,
        model_dim=args.model_dim, heads=args.heads,
    ).to(device)
    history, milestones = train_matcher(
        predictor, matcher, support, support_groups, args, device,
        query=query, query_groups=query_groups,
    )
    query_policy = evaluate_policy(predictor, matcher, query, query_groups, args, device)
    support_policy = evaluate_policy(predictor, matcher, support, support_groups, args, device)
    permuted_policy = evaluate_policy(
        predictor, matcher, query, query_groups, args, device, permuted=True,
    )
    shifted_policy = evaluate_policy(
        predictor, matcher, rolled_cache(query, args.seed + 54000), query_groups,
        args, device,
    )
    predictor_after = evaluate_predictor(
        predictor, query, device, args.predictor_batch, args.seed + 53000,
    )
    probe_hash_after = file_sha256(checkpoint)
    if probe_hash_before != probe_hash_after:
        raise RuntimeError('Probe checkpoint changed during boundary training')
    result = {
        'format_version': 10, 'scene': args.scene,
        'method': 'frozen_predictive_innovation_same_frame_retrieval',
        'probe_checkpoint': str(checkpoint), 'probe_sha256': probe_hash_before,
        'probe_mutated': False, 'args': vars(args),
        'split': {'support_frames': len(support['paths']), 'query_frames': len(query['paths']),
                  'videos': videos},
        'predictor_history': predictor_history, 'predictor_query': predictor_query,
        'predictor_query_after_matcher': predictor_after,
        'matcher_history': history, 'milestones': milestones,
        'query_policy': query_policy, 'support_policy': support_policy,
        'permuted_query_policy': permuted_policy,
        'shifted_query_policy': shifted_policy,
        'predictor_parameters': sum(p.numel() for p in predictor.parameters()),
        'matcher_parameters': sum(p.numel() for p in matcher.parameters()),
        'peak_gpu_gib': (torch.cuda.max_memory_allocated(device) / 2**30
                         if device.type == 'cuda' else 0.0),
    }
    torch.save({'predictor': predictor.state_dict(), 'matcher': matcher.state_dict(),
                'result': result}, out_dir / 'model.pth')
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(
        f'[{args.scene}] query=L{query_policy["boundary"]} '
        f'support=L{support_policy["boundary"]} '
        f'permuted=L{permuted_policy["boundary"]} shifted=L{shifted_policy["boundary"]}',
        flush=True,
    )


if __name__ == '__main__':
    main()
