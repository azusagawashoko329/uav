"""Two-stage predictive-innovation boundary study on one independently trained scene."""
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

from experiment_DLBM.common import SCENES, probe_path, set_seed
from experiment_DLBM.cross_layer_predictor import CrossLayerPredictor
from experiment_DLBM.dlbm_v008 import cumulative_coarse_policy
from experiment_DLBM.innovation_boundary import InnovationOpeningHead
from experiment_DLBM.probe_cross_layer_predictor import (evaluate as evaluate_predictor,
                                                          independent_layer_roll,
                                                          prediction_loss)
from experiment_DLBM.run_boundary_v001 import all_scene_paths, load_probe, split_video_balanced
from experiment_DLBM.train_boundary_v006 import cache_relation_maps
from experiment_DLBM.train_boundary_v007 import sample_video_batch, video_indices
from experiment_DLBM.train_boundary_v008 import (REAL_TRANSITIONS, balanced_flip_loss,
                                                  choose_training_transition, group_score,
                                                  sampled_pairs, structural_nulls,
                                                  transition_targets)


def train_predictor(model, support, args, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.predictor_lr,
                                  weight_decay=args.wd)
    rng = np.random.RandomState(args.seed + 32000)
    history = []
    for epoch in range(1, args.predictor_epochs + 1):
        model.train(); losses = []
        for _ in range(args.steps_per_epoch):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({'epoch': epoch, 'loss': float(np.mean(losses))})
    model.eval().requires_grad_(False)
    return history


def branch_loss(predictor, head, maps, base_layers, candidate_layers, args):
    with torch.set_grad_enabled(args.joint_finetune):
        opening = predictor.opening_features(maps, base_layers, candidate_layers)
    embedding = head.encode(opening)
    first, second = sampled_pairs(len(maps), args.pair_samples, maps.device)
    base, candidate, flip = transition_targets(
        maps, base_layers, candidate_layers, args.topk, first, second,
    )
    predicted_base, predicted_candidate = head.order_scores(embedding)
    base_label = (base.index_select(0, first) > base.index_select(0, second)).to(maps.dtype)
    candidate_label = (
        candidate.index_select(0, first) > candidate.index_select(0, second)
    ).to(maps.dtype)
    rank = 0.5 * (
        F.binary_cross_entropy_with_logits(
            predicted_base.index_select(0, first) - predicted_base.index_select(0, second),
            base_label,
        )
        + F.binary_cross_entropy_with_logits(
            predicted_candidate.index_select(0, first)
            - predicted_candidate.index_select(0, second), candidate_label,
        )
    )
    flip_loss = balanced_flip_loss(
        head.flip_logits(embedding, first, second), flip, args.pair_samples,
    )
    return rank, flip_loss


def train_head(predictor, head, support, support_groups, args, device):
    if args.joint_finetune:
        predictor.requires_grad_(True)
        parameters = list(head.parameters()) + list(predictor.parameters())
    else:
        parameters = list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.head_lr, weight_decay=args.wd)
    rng = np.random.RandomState(args.seed + 33000)
    history = []
    for epoch in range(1, args.head_epochs + 1):
        head.train(); totals = defaultdict(list)
        for step in range(args.steps_per_epoch):
            _, base_layers, candidate_layers, is_null = choose_training_transition(step, rng)
            (normal, pseudo), _ = sample_video_batch(
                support, support_groups, args.frame_batch, rng, device,
            )
            if args.joint_finetune:
                # Alternating optimization keeps the predictive representation anchored
                # without introducing another weighted sum of hand-tuned losses.
                prediction_anchor, _ = prediction_loss(predictor, normal)
                optimizer.zero_grad(set_to_none=True)
                prediction_anchor.backward()
                torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                optimizer.step()
                totals['prediction_anchor'].append(float(prediction_anchor.detach()))
            normal_rank, normal_flip = branch_loss(
                predictor, head, normal, base_layers, candidate_layers, args,
            )
            pseudo_rank, pseudo_flip = branch_loss(
                predictor, head, pseudo, base_layers, candidate_layers, args,
            )
            loss = 0.25 * (normal_rank + normal_flip + pseudo_rank + pseudo_flip)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite boundary-head loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()
            for key, value in {
                    'loss': loss, 'normal_rank': normal_rank, 'normal_flip': normal_flip,
                    'pseudo_rank': pseudo_rank, 'pseudo_flip': pseudo_flip,
                    'grad_norm': gradient}.items():
                totals[key].append(float(value.detach()))
            totals['null_fraction'].append(float(is_null))
        row = {'epoch': epoch, **{key: float(np.mean(values))
                                  for key, values in totals.items()}}
        history.append(row)
        if epoch in {1, 5, 10, args.head_epochs}:
            print(f'  head ep{epoch:02d} loss={row["loss"]:.4f} '
                  f'rankN={row["normal_rank"]:.4f} flipN={row["normal_flip"]:.4f}',
                  flush=True)
    return history


@torch.no_grad()
def transition_metric(predictor, head, cache, groups, base_layers, candidate_layers,
                      args, device):
    output = defaultdict(list)
    for video in sorted(groups):
        index = torch.as_tensor(groups[video], dtype=torch.long)
        for branch in ('normal', 'pseudo'):
            maps = cache[branch].index_select(0, index).to(device=device, dtype=torch.float32)
            opening = predictor.opening_features(maps, base_layers, candidate_layers)
            embedding = head.encode(opening)
            first, second = sampled_pairs(len(maps), args.eval_pairs, device)
            _, _, target = transition_targets(
                maps, base_layers, candidate_layers, args.topk, first, second,
            )
            predicted = torch.sigmoid(head.flip_logits(embedding, first, second))
            output[f'predicted_{branch}_flip'].append(predicted.mean())
            output[f'actual_{branch}_flip'].append(target.mean())
    return {key: float(torch.stack(values).mean()) for key, values in output.items()}


@torch.no_grad()
def evaluate_gate(predictor, head, cache, groups, gate, args, device):
    real = transition_metric(
        predictor, head, cache, groups, *REAL_TRANSITIONS[gate], args, device,
    )
    nulls = structural_nulls(gate)
    if len(nulls) > args.max_nulls:
        rng = np.random.RandomState(args.seed + (34000 if gate == 'M' else 35000))
        chosen = rng.choice(len(nulls), args.max_nulls, replace=False)
        nulls = [nulls[int(index)] for index in sorted(chosen)]
    null_rows = [transition_metric(
        predictor, head, cache, groups, base, candidate, args, device,
    ) for base, candidate in nulls]
    result = dict(real)
    for branch in ('normal', 'pseudo'):
        for source in ('predicted', 'actual'):
            key = f'{source}_{branch}_flip'
            median = float(torch.tensor([row[key] for row in null_rows]).median())
            result[f'{source}_{branch}_null_median'] = median
            result[f'{source}_{branch}_gain'] = result[key] - median
    result['null_count'] = len(null_rows)
    return result


@torch.no_grad()
def evaluate_policy(predictor, head, cache, groups, args, device):
    head.eval(); predictor.eval()
    middle = evaluate_gate(predictor, head, cache, groups, 'M', args, device)
    shallow = evaluate_gate(predictor, head, cache, groups, 'S', args, device)
    boundary, utility = cumulative_coarse_policy(
        torch.tensor(middle['predicted_normal_gain']),
        torch.tensor(shallow['predicted_normal_gain']),
    )
    return {'boundary': boundary, 'selected_taps': list(range(boundary, 12)),
            'middle': middle, 'shallow': shallow,
            'utilities': [float(value) for value in utility]}


def rolled_cache(cache, seed):
    return {'paths': cache['paths'],
            'normal': independent_layer_roll(cache['normal'].float(), seed),
            'pseudo': independent_layer_roll(cache['pseudo'].float(), seed + 1)}


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
    parser.add_argument('--predictor_epochs', type=int, default=10)
    parser.add_argument('--head_epochs', type=int, default=20)
    parser.add_argument('--steps_per_epoch', type=int, default=44)
    parser.add_argument('--predictor_batch', type=int, default=24)
    parser.add_argument('--frame_batch', type=int, default=48)
    parser.add_argument('--pair_samples', type=int, default=512)
    parser.add_argument('--eval_pairs', type=int, default=4096)
    parser.add_argument('--max_nulls', type=int, default=16)
    parser.add_argument('--predictor_lr', type=float, default=1e-3)
    parser.add_argument('--head_lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--joint_finetune', action='store_true')
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    teacher, ocbe, student, cfg = load_probe(args.scene, args.probe_dir, device)
    paths, videos = split_video_balanced(
        all_scene_paths(args.scene), args.max_frames_per_video, args.seed,
    )
    support = cache_relation_maps(paths['support'], teacher, ocbe, student, cfg, args, device)
    query = cache_relation_maps(paths['query'], teacher, ocbe, student, cfg, args, device)
    del teacher, ocbe, student
    support_groups, query_groups = video_indices(support['paths']), video_indices(query['paths'])
    predictor = CrossLayerPredictor(channels=args.channels).to(device)
    predictor_history = train_predictor(predictor, support, args, device)
    predictor_query = evaluate_predictor(
        predictor, query, device, args.predictor_batch, args.seed + 36000,
    )
    print(f'[{args.scene}] predictor query={predictor_query["normal_aligned_smooth_l1"]:.6g} '
          f'roll={predictor_query["normal_roll_loss_ratio"]:.3f}', flush=True)
    head = InnovationOpeningHead(
        predictor_channels=args.channels, topk=args.topk, model_dim=args.model_dim,
    ).to(device)
    head_history = train_head(predictor, head, support, support_groups, args, device)
    predictor.eval().requires_grad_(False)
    predictor_query_after_head = evaluate_predictor(
        predictor, query, device, args.predictor_batch, args.seed + 36000,
    )
    query_policy = evaluate_policy(predictor, head, query, query_groups, args, device)
    support_policy = evaluate_policy(predictor, head, support, support_groups, args, device)
    shifted_query = rolled_cache(query, args.seed + 37000)
    shifted_policy = evaluate_policy(
        predictor, head, shifted_query, query_groups, args, device,
    )
    result = {
        'scene': args.scene, 'method': 'frozen_predictive_innovation_opening_head',
        'probe_checkpoint': str(probe_path(args.probe_dir, args.scene)),
        'probe_mutated': False, 'args': vars(args),
        'split': {'support_frames': len(support['paths']), 'query_frames': len(query['paths']),
                  'videos': videos},
        'predictor_history': predictor_history, 'predictor_query': predictor_query,
        'predictor_query_after_head': predictor_query_after_head,
        'head_history': head_history, 'query_policy': query_policy,
        'support_policy': support_policy, 'shifted_query_policy': shifted_policy,
        'predictor_parameters': sum(p.numel() for p in predictor.parameters()),
        'head_parameters': sum(p.numel() for p in head.parameters()),
    }
    torch.save({'predictor': predictor.state_dict(), 'head': head.state_dict(),
                'result': result}, out_dir / 'model.pth')
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'[{args.scene}] query=L{query_policy["boundary"]} '
          f'support=L{support_policy["boundary"]} shifted=L{shifted_policy["boundary"]}',
          flush=True)


if __name__ == '__main__':
    main()
