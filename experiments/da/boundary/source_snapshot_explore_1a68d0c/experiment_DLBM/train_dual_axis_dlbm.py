"""Train v017 dual-axis DLBM on normal order novelty and pseudo nuisance."""
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
from experiment_DLBM.dual_axis_dlbm import DualAxisDLBM
from experiment_DLBM.probe_cross_layer_predictor import (
    evaluate as evaluate_predictor, independent_layer_roll,
)
from experiment_DLBM.run_boundary_v001 import all_scene_paths, load_probe, split_video_balanced
from experiment_DLBM.train_boundary_v006 import cache_relation_maps
from experiment_DLBM.train_boundary_v007 import video_indices
from experiment_DLBM.train_boundary_v008 import (
    REAL_TRANSITIONS, group_score, sampled_pairs, structural_nulls,
)
from experiment_DLBM.train_episode_innovation_boundary import file_sha256


def balanced_soft_mask_bce(logits, target):
    if logits.shape != target.shape:
        raise ValueError('pseudo mask/logit shape mismatch')
    target = target.to(dtype=logits.dtype).clamp(0, 1)
    positive = target.mean().clamp(1e-4, 1 - 1e-4)
    weights = 0.5 * target / positive + 0.5 * (1 - target) / (1 - positive)
    return F.binary_cross_entropy_with_logits(logits, target, weight=weights)


def candidate_labels(maps, candidate, topk, first, second):
    score = group_score(maps, candidate, topk)
    return (
        score.index_select(0, first) > score.index_select(0, second)
    ).to(maps.dtype)


def normal_order_loss(predictor, model, maps, base, candidate, first, second, topk):
    with torch.no_grad():
        opening = predictor.opening_features(maps, base, candidate)
        target = candidate_labels(maps, candidate, topk, first, second)
    base_logit, opened_logit = model.pair_logits(opening, first, second)
    base_nll = F.binary_cross_entropy_with_logits(base_logit, target)
    opened_nll = F.binary_cross_entropy_with_logits(opened_logit, target)
    return 0.5 * (base_nll + opened_nll), {
        'base_nll': base_nll, 'opened_nll': opened_nll,
        'improvement': base_nll.detach() - opened_nll.detach(),
    }


def pseudo_mask_loss(predictor, model, maps, mask, base, candidate):
    with torch.no_grad():
        opening = predictor.opening_features(maps, base, candidate)
    logits = model.nuisance_logits(opening)
    loss = balanced_soft_mask_bce(logits, mask)
    return loss, logits


def sample_episode(cache, groups, count, rng, device):
    video = sorted(groups)[int(rng.randint(len(groups)))]
    available = groups[video]
    size = min(int(count), len(available))
    chosen = rng.choice(available, size=size, replace=False)
    index = torch.as_tensor(chosen, dtype=torch.long)
    return {
        name: cache[name].index_select(0, index).to(
            device=device, dtype=torch.float32, non_blocking=True,
        )
        for name in ('normal', 'pseudo', 'pseudo_mask')
    }


def choose_transition(step, rng):
    gate = ('M', 'S')[step % 2]
    if int(rng.randint(2)) == 0:
        return gate, *REAL_TRANSITIONS[gate], 'real'
    nulls = structural_nulls(gate)
    base, candidate = nulls[int(rng.randint(len(nulls)))]
    return gate, base, candidate, 'null'


def train_model(predictor, model, support, groups, args, device,
                query=None, query_groups=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    rng = np.random.RandomState(args.seed + 111000)
    history, milestones = [], {}
    for epoch in range(1, args.epochs + 1):
        model.train(); totals = defaultdict(list)
        for step in range(args.steps_per_epoch):
            gate, base, candidate, kind = choose_transition(step, rng)
            episode = sample_episode(
                support, groups, args.episode_frames, rng, device,
            )
            first, second = sampled_pairs(
                len(episode['normal']), args.pair_samples, device,
            )
            order, order_detail = normal_order_loss(
                predictor, model, episode['normal'], base, candidate,
                first, second, args.topk,
            )
            nuisance, _ = pseudo_mask_loss(
                predictor, model, episode['pseudo'], episode['pseudo_mask'],
                base, candidate,
            )
            loss = model.multitask_loss(order, nuisance)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite dual-axis loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(gradient):
                raise FloatingPointError('non-finite dual-axis gradient')
            optimizer.step()
            totals['loss'].append(float(loss.detach()))
            totals['order_loss'].append(float(order.detach()))
            totals['nuisance_loss'].append(float(nuisance.detach()))
            totals['grad_norm'].append(float(gradient.detach()))
            totals[f'{gate}_{kind}_base_nll'].append(
                float(order_detail['base_nll'].detach()),
            )
        row = {'epoch': epoch, **{
            key: float(np.mean(values)) for key, values in totals.items()
        }, 'axis_precision': [float(value) for value in model.axis_precision().detach()]}
        if epoch in args.milestones and query is not None:
            policy = evaluate_policy(
                predictor, model, query, query_groups, args, device,
            )
            row['query'] = policy; milestones[str(epoch)] = policy
            print(
                f'  dual ep{epoch:02d} loss={row["loss"]:.4f} '
                f'query=L{policy["boundary"]} '
                f'axis={[[round(v,5) for v in policy[g]["axes"]] for g in ("M","S")]}',
                flush=True,
            )
        history.append(row)
    return history, milestones


@torch.no_grad()
def transition_metric(predictor, model, cache, groups, base, candidate, args, device):
    rows = []
    for video in sorted(groups):
        index = torch.as_tensor(groups[video], dtype=torch.long)
        normal = cache['normal'].index_select(0, index).to(device=device).float()
        pseudo = cache['pseudo'].index_select(0, index).to(device=device).float()
        mask = cache['pseudo_mask'].index_select(0, index).to(device=device).float()
        first, second = sampled_pairs(len(normal), args.eval_pairs, device)
        _, order = normal_order_loss(
            predictor, model, normal, base, candidate, first, second, args.topk,
        )
        nuisance, logits = pseudo_mask_loss(
            predictor, model, pseudo, mask, base, candidate,
        )
        rows.append({
            'video': video, 'normal_base_nll': float(order['base_nll']),
            'normal_opened_nll': float(order['opened_nll']),
            'normal_improvement': float(order['improvement']),
            'pseudo_mask_nll': float(nuisance),
            'pseudo_mask_logit_positive': float(torch.sigmoid(logits)[mask > 0.1].mean()),
            'pseudo_mask_logit_negative': float(torch.sigmoid(logits)[mask <= 0.1].mean()),
        })
    output = {'video_rows': rows}
    for key in rows[0]:
        if key != 'video':
            output[key] = float(np.mean([row[key] for row in rows]))
    return output


@torch.no_grad()
def evaluate_gate(predictor, model, cache, groups, gate, args, device):
    real = transition_metric(
        predictor, model, cache, groups, *REAL_TRANSITIONS[gate], args, device,
    )
    nulls = structural_nulls(gate)
    if len(nulls) > args.eval_nulls:
        rng = np.random.RandomState(args.seed + (112000 if gate == 'M' else 113000))
        selected = rng.choice(len(nulls), size=args.eval_nulls, replace=False)
        nulls = [nulls[int(index)] for index in sorted(selected)]
    null_rows = [
        transition_metric(predictor, model, cache, groups, base, candidate, args, device)
        for base, candidate in nulls
    ]
    null_base_nll = float(np.mean([row['normal_base_nll'] for row in null_rows]))
    null_mask_nll = float(np.mean([row['pseudo_mask_nll'] for row in null_rows]))
    normal_novelty = real['normal_base_nll'] - null_base_nll
    pseudo_nuisance = null_mask_nll - real['pseudo_mask_nll']
    return {
        'axes': [normal_novelty, pseudo_nuisance],
        'normal_novelty': normal_novelty,
        'pseudo_nuisance': pseudo_nuisance,
        'real': real, 'null_count': len(nulls),
        'null_normal_base_nll': null_base_nll,
        'null_pseudo_mask_nll': null_mask_nll,
    }


@torch.no_grad()
def evaluate_policy(predictor, model, cache, groups, args, device):
    model.eval(); predictor.eval()
    middle = evaluate_gate(predictor, model, cache, groups, 'M', args, device)
    shallow = evaluate_gate(predictor, model, cache, groups, 'S', args, device)
    boundary, probability, utility, evidence = model.dual_boundary_distribution(
        tuple(torch.tensor(value) for value in middle['axes']),
        tuple(torch.tensor(value) for value in shallow['axes']),
    )
    return {
        'boundary': boundary, 'selected_taps': list(range(boundary, 12)),
        'probabilities': {'D': float(probability[0]), 'MD': float(probability[1]),
                          'ALL': float(probability[2])},
        'utilities': {'D': float(utility[0]), 'MD': float(utility[1]),
                      'ALL': float(utility[2])},
        'gate_evidence': {'M': float(evidence[0]), 'S': float(evidence[1])},
        'axis_precision': [float(value) for value in model.axis_precision()],
        'M': middle, 'S': shallow,
    }


def rolled_cache(cache, seed):
    return {
        'paths': cache['paths'],
        'normal': independent_layer_roll(cache['normal'].float(), seed),
        'pseudo': independent_layer_roll(cache['pseudo'].float(), seed + 1),
        'pseudo_mask': cache['pseudo_mask'],
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
    parser.add_argument('--pseudo_mode', choices=['multi'], default='multi')
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--predictor_batch', type=int, default=24)
    parser.add_argument('--predictor_checkpoint', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--steps_per_epoch', type=int, default=30)
    parser.add_argument('--episode_frames', type=int, default=32)
    parser.add_argument('--pair_samples', type=int, default=256)
    parser.add_argument('--eval_pairs', type=int, default=2048)
    parser.add_argument('--eval_nulls', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--milestones', type=int, nargs='+', default=[5, 10, 20, 30])
    args = parser.parse_args()
    args.cache_pseudo_mask = True
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda': torch.cuda.set_device(device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = probe_path(args.probe_dir, args.scene); before = file_sha256(checkpoint)
    teacher, ocbe, student, cfg = load_probe(args.scene, args.probe_dir, device)
    paths, videos = split_video_balanced(
        all_scene_paths(args.scene), args.max_frames_per_video, args.seed,
    )
    support = cache_relation_maps(
        paths['support'], teacher, ocbe, student, cfg, args, device,
    )
    query = cache_relation_maps(
        paths['query'], teacher, ocbe, student, cfg, args, device,
    )
    del teacher, ocbe, student
    support_groups, query_groups = video_indices(support['paths']), video_indices(query['paths'])
    full_cache = {
        'paths': [*support['paths'], *query['paths']],
        **{name: torch.cat((support[name], query[name]), dim=0)
           for name in ('normal', 'pseudo', 'pseudo_mask')},
    }
    full_groups = video_indices(full_cache['paths'])
    saved = torch.load(args.predictor_checkpoint, map_location='cpu')
    predictor = CrossLayerPredictor(channels=args.channels).to(device)
    predictor.load_state_dict(saved['predictor']); predictor.eval().requires_grad_(False)
    predictor_query = evaluate_predictor(
        predictor, query, device, args.predictor_batch, args.seed + 114000,
    )
    print(
        f'[{args.scene}] predictor={predictor_query["normal_aligned_smooth_l1"]:.6g} '
        f'roll={predictor_query["normal_roll_loss_ratio"]:.3f}', flush=True,
    )
    set_seed(args.seed + 115000)
    model = DualAxisDLBM(
        predictor_channels=args.channels, topk=args.topk,
        model_dim=args.model_dim, heads=args.heads,
    ).to(device)
    history, milestones = train_model(
        predictor, model, support, support_groups, args, device, query, query_groups,
    )
    query_policy = evaluate_policy(predictor, model, query, query_groups, args, device)
    support_policy = evaluate_policy(predictor, model, support, support_groups, args, device)
    full_policy = evaluate_policy(predictor, model, full_cache, full_groups, args, device)
    shifted = evaluate_policy(
        predictor, model, rolled_cache(query, args.seed + 116000),
        query_groups, args, device,
    )
    if before != file_sha256(checkpoint): raise RuntimeError('Probe checkpoint changed')
    result = {
        'format_version': 17, 'scene': args.scene,
        'method': 'dual_axis_order_novelty_pseudo_mask_nuisance_dlbm',
        'probe_checkpoint': str(checkpoint), 'probe_sha256': before,
        'probe_mutated': False, 'args': vars(args), 'videos': videos,
        'predictor_query': predictor_query, 'history': history,
        'milestones': milestones, 'query_policy': query_policy,
        'support_policy': support_policy, 'full_scene_policy': full_policy,
        'shifted_query_policy': shifted,
    }
    torch.save({
        'predictor': predictor.state_dict(), 'dlbm': model.state_dict(),
        'result': result,
    }, out_dir / 'model.pth')
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(
        f'[{args.scene}] full=L{full_policy["boundary"]} '
        f'query=L{query_policy["boundary"]} support=L{support_policy["boundary"]} '
        f'shifted=L{shifted["boundary"]}', flush=True,
    )


if __name__ == '__main__':
    main()
