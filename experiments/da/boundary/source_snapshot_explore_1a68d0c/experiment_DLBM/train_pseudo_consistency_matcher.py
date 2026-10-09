"""DLBM study: retrieve scene frames across two pseudo opening views."""
import argparse
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

from experiment_DLBM.common import SCENES, multi_mode_pseudo_stress, probe_path, set_seed
from experiment_DLBM.cross_layer_predictor import CrossLayerPredictor
from experiment_DLBM.episode_innovation_boundary import SCALE_TRANSITIONS
from experiment_DLBM.episode_innovation_matcher import (
    EpisodeInnovationMatcher, boundary_from_retrieval, retrieval_metrics,
    symmetric_retrieval_loss,
)
from experiment_DLBM.probe_cross_layer_predictor import evaluate as evaluate_predictor
from experiment_DLBM.run_boundary_v001 import all_scene_paths, load_probe, split_video_balanced
from experiment_DLBM.train_boundary_v006 import _loader, discrepancy_maps
from experiment_DLBM.train_boundary_v007 import sample_video_batch, video_indices
from experiment_DLBM.train_episode_innovation_boundary import (
    deterministic_derangement, file_sha256, train_predictor,
)


@torch.no_grad()
def cache_views(paths, teacher, ocbe, student, cfg, args, device):
    values = defaultdict(list)
    set_seed(args.seed + 71000)
    for images in _loader(paths, teacher, cfg, args.cache_bs, args.workers):
        images = images.to(device, non_blocking=True)
        views = {
            'normal': images,
            'pseudo_first': multi_mode_pseudo_stress(images),
            'pseudo_second': multi_mode_pseudo_stress(images),
        }
        for name, view in views.items():
            taps = teacher(view)['patches']
            reconstruction = student(ocbe(taps))
            values[name].append(
                discrepancy_maps(taps, reconstruction).half().cpu(),
            )
    return {name: torch.cat(chunks) for name, chunks in values.items()} | {'paths': list(paths)}


def predictor_cache(cache):
    # Existing predictor helpers expect normal and pseudo keys.  The second pseudo
    # view is used only by the boundary matcher.
    return {'normal': cache['normal'], 'pseudo': cache['pseudo_first'],
            'paths': cache['paths']}


def opening(predictor, maps, base, candidate):
    with torch.no_grad():
        return predictor.opening_features(maps, base, candidate)


def sample_views(cache, groups, count, rng, device):
    video = sorted(groups)[int(rng.randint(len(groups)))]
    available = groups[video]
    chosen = rng.choice(available, min(count, len(available)), replace=False)
    index = torch.as_tensor(chosen, dtype=torch.long)
    return tuple(cache[name].index_select(0, index).to(device=device, dtype=torch.float32)
                 for name in ('pseudo_first', 'pseudo_second'))


def train_matcher(predictor, matcher, support, groups, args, device,
                  query=None, query_groups=None):
    optimizer = torch.optim.AdamW(
        matcher.parameters(), lr=args.matcher_lr, weight_decay=args.wd,
    )
    rng = np.random.RandomState(args.seed + 72000)
    scales = tuple(SCALE_TRANSITIONS); history, milestones = [], {}
    for epoch in range(1, args.matcher_epochs + 1):
        matcher.train(); totals = defaultdict(list)
        for step in range(args.matcher_steps):
            scale = scales[step % len(scales)]
            base, candidate = SCALE_TRANSITIONS[scale]
            first, second = sample_views(
                support, groups, args.episode_frames, rng, device,
            )
            logits = matcher.cross_opening_logits(
                opening(predictor, first, base, candidate),
                opening(predictor, second, base, candidate),
            )
            loss = symmetric_retrieval_loss(logits)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(matcher.parameters(), args.grad_clip)
            if not torch.isfinite(loss) or not torch.isfinite(gradient):
                raise FloatingPointError('non-finite pseudo consistency value')
            optimizer.step()
            totals['loss'].append(float(loss.detach()))
            totals[f'{scale}_loss'].append(float(loss.detach()))
        row = {'epoch': epoch, **{key: float(np.mean(value))
                                  for key, value in totals.items()}}
        if epoch in args.milestones and query is not None:
            policy = evaluate_policy(predictor, matcher, query, query_groups, args, device)
            row['query'] = policy; milestones[str(epoch)] = policy
            print(f'  matcher ep{epoch:02d} loss={row["loss"]:.4f} '
                  f'query=L{policy["boundary"]} '
                  f'mrr={[round(x,4) for x in policy["evidence"]]}', flush=True)
        history.append(row)
    return history, milestones


@torch.no_grad()
def evaluate_policy(predictor, matcher, cache, groups, args, device, permuted=False):
    matcher.eval(); predictor.eval(); scales = {}; scale_names = tuple(SCALE_TRANSITIONS)
    for scale_index, scale in enumerate(scale_names):
        base, candidate = SCALE_TRANSITIONS[scale]; rows = []
        for video_index, video in enumerate(sorted(groups)):
            index = torch.as_tensor(groups[video], dtype=torch.long)
            first = cache['pseudo_first'].index_select(0, index).to(device=device).float()
            second = cache['pseudo_second'].index_select(0, index).to(device=device).float()
            if len(first) < 2:
                continue
            if permuted:
                permutation = deterministic_derangement(
                    len(second), device,
                    args.seed + 73000 + 10000 * scale_index + video_index,
                )
                second = second.index_select(0, permutation)
            logits = matcher.cross_opening_logits(
                opening(predictor, first, base, candidate),
                opening(predictor, second, base, candidate),
            )
            rows.append({'video': video, **retrieval_metrics(logits)})
        scales[scale] = {
            'videos': len(rows), 'video_rows': rows,
            **{metric: float(np.mean([row[metric] for row in rows]))
               for metric in ('loss', 'top1', 'mrr', 'normalized_information')},
        }
    evidence = {scale: scales[scale]['mrr'] for scale in scale_names}
    boundary, values = boundary_from_retrieval(evidence)
    return {'boundary': boundary, 'selected_taps': list(range(boundary, 12)),
            'scale_order': list(scale_names), 'evidence': [float(x) for x in values],
            'scales': scales, 'permuted_pairing': bool(permuted)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, choices=sorted(SCENES))
    parser.add_argument('--probe_dir', default='experiment_DLBM/run/probe/ckpt')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_frames_per_video', type=int, default=48)
    parser.add_argument('--cache_bs', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
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
    parser.add_argument('--predictor_checkpoint', default='')
    # Compatibility with shared cache/predictor helpers.
    parser.add_argument('--pseudo_min_area', type=float, default=0.02)
    parser.add_argument('--pseudo_max_area', type=float, default=0.15)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda': torch.cuda.set_device(device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = probe_path(args.probe_dir, args.scene); before = file_sha256(checkpoint)
    teacher, ocbe, student, cfg = load_probe(args.scene, args.probe_dir, device)
    paths, videos = split_video_balanced(all_scene_paths(args.scene),
                                         args.max_frames_per_video, args.seed)
    support = cache_views(paths['support'], teacher, ocbe, student, cfg, args, device)
    query = cache_views(paths['query'], teacher, ocbe, student, cfg, args, device)
    del teacher, ocbe, student
    support_groups, query_groups = video_indices(support['paths']), video_indices(query['paths'])
    # Match the validated scalar-predictor initialization stream; pseudo-view
    # caching must not silently choose model weights through its RNG consumption.
    set_seed(args.seed + 16000)
    predictor = CrossLayerPredictor(channels=args.channels).to(device)
    if args.predictor_checkpoint:
        saved = torch.load(args.predictor_checkpoint, map_location='cpu')
        predictor.load_state_dict(saved['predictor'])
        predictor.eval().requires_grad_(False)
        predictor_history = saved.get('result', {}).get('history', [])
    else:
        args.steps_per_epoch = args.predictor_steps
        predictor_history = train_predictor(predictor, predictor_cache(support), args, device)
    predictor_query = evaluate_predictor(predictor, predictor_cache(query), device,
                                         args.predictor_batch, args.seed + 74000)
    print(f'[{args.scene}] predictor={predictor_query["normal_aligned_smooth_l1"]:.6g} '
          f'roll={predictor_query["normal_roll_loss_ratio"]:.3f}', flush=True)
    matcher = EpisodeInnovationMatcher(predictor_channels=args.channels,
                                       model_dim=args.model_dim, heads=args.heads).to(device)
    history, milestones = train_matcher(predictor, matcher, support, support_groups,
                                        args, device, query, query_groups)
    query_policy = evaluate_policy(predictor, matcher, query, query_groups, args, device)
    support_policy = evaluate_policy(predictor, matcher, support, support_groups, args, device)
    permuted = evaluate_policy(predictor, matcher, query, query_groups, args, device,
                               permuted=True)
    if before != file_sha256(checkpoint): raise RuntimeError('Probe checkpoint changed')
    result = {'format_version': 11, 'scene': args.scene,
              'method': 'pseudo_branch_cross_view_opening_retrieval',
              'probe_checkpoint': str(checkpoint), 'probe_sha256': before,
              'probe_mutated': False, 'args': vars(args), 'videos': videos,
              'predictor_history': predictor_history, 'predictor_query': predictor_query,
              'matcher_history': history, 'milestones': milestones,
              'query_policy': query_policy, 'support_policy': support_policy,
              'permuted_query_policy': permuted}
    torch.save({'predictor': predictor.state_dict(), 'matcher': matcher.state_dict(),
                'result': result}, out_dir / 'model.pth')
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'[{args.scene}] query=L{query_policy["boundary"]} '
          f'support=L{support_policy["boundary"]} permuted=L{permuted["boundary"]}', flush=True)


if __name__ == '__main__': main()
