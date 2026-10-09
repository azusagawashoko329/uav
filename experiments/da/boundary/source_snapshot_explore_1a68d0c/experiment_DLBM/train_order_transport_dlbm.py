"""Train v012 DLBM directly on episode order-transport gain."""
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
from experiment_DLBM.order_transport_dlbm import OrderTransportDLBM
from experiment_DLBM.probe_cross_layer_predictor import (
    evaluate as evaluate_predictor, independent_layer_roll,
)
from experiment_DLBM.run_boundary_v001 import all_scene_paths, load_probe, split_video_balanced
from experiment_DLBM.train_boundary_v006 import cache_relation_maps
from experiment_DLBM.train_boundary_v007 import sample_video_batch, video_indices
from experiment_DLBM.train_boundary_v008 import (
    REAL_TRANSITIONS, group_score, sampled_pairs, structural_nulls,
)
from experiment_DLBM.train_episode_innovation_boundary import file_sha256, train_predictor


def transition_targets(maps, base, candidate, topk, first, second):
    base_score = group_score(maps, base, topk)
    candidate_score = group_score(maps, candidate, topk)
    base_delta = base_score.index_select(0, first) - base_score.index_select(0, second)
    candidate_delta = (
        candidate_score.index_select(0, first) - candidate_score.index_select(0, second)
    )
    return {
        'base_order': (base_delta > 0).to(maps.dtype),
        'candidate_order': (candidate_delta > 0).to(maps.dtype),
        'flip_rate': ((base_delta * candidate_delta) < 0).to(maps.dtype).mean(),
    }


def encode_transition(predictor, model, maps, base, candidate, first, second, topk):
    with torch.no_grad():
        opening = predictor.opening_features(maps, base, candidate)
    encoded = model.encode_branch(opening, first, second)
    target = transition_targets(maps, base, candidate, topk, first, second)
    base_logit = (
        encoded['base_score'].index_select(0, first)
        - encoded['base_score'].index_select(0, second)
    )
    candidate_logit = (
        encoded['candidate_score'].index_select(0, first)
        - encoded['candidate_score'].index_select(0, second)
    )
    order_loss = 0.5 * (
        F.binary_cross_entropy_with_logits(base_logit, target['base_order'])
        + F.binary_cross_entropy_with_logits(
            candidate_logit, target['candidate_order'],
        )
    )
    return encoded['episode'], target['flip_rate'], order_loss


def choose_nulls(gate, maximum, rng):
    nulls = structural_nulls(gate)
    if len(nulls) <= maximum:
        return nulls
    index = rng.choice(len(nulls), size=maximum, replace=False)
    return [nulls[int(value)] for value in sorted(index)]


def gate_step(predictor, model, normal, pseudo, gate, nulls, args):
    first, second = sampled_pairs(len(normal), args.pair_samples, normal.device)
    transitions = [REAL_TRANSITIONS[gate], *nulls]
    fused, flips = [], {'normal': [], 'pseudo': []}
    order_losses = []
    for base, candidate in transitions:
        branches = []
        for name, maps in (('normal', normal), ('pseudo', pseudo)):
            token, flip, order_loss = encode_transition(
                predictor, model, maps, base, candidate, first, second, args.topk,
            )
            branches.append(token); flips[name].append(flip)
            order_losses.append(order_loss)
        fused.append(model.fuse_branches(*branches))
    fused = torch.stack(fused)
    gain_logit = model.gain_logit(fused[0], fused[1:])
    branch_targets = []
    for name in ('normal', 'pseudo'):
        values = torch.stack(flips[name])
        branch_targets.append((values[0] > values[1:]).to(normal.dtype).mean())
    gain_target = torch.stack(branch_targets).mean()
    gain_loss = F.binary_cross_entropy_with_logits(gain_logit, gain_target)
    order_loss = torch.stack(order_losses).mean()
    loss = 0.5 * (gain_loss + order_loss)
    return loss, {'gain_loss': gain_loss, 'order_loss': order_loss,
                  'gain_logit': gain_logit, 'gain_target': gain_target,
                  'normal_real_flip': flips['normal'][0],
                  'pseudo_real_flip': flips['pseudo'][0]}


def train_head(predictor, model, support, groups, args, device,
               query=None, query_groups=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    rng = np.random.RandomState(args.seed + 82000)
    history, milestones = [], {}
    for epoch in range(1, args.epochs + 1):
        model.train(); totals = defaultdict(list)
        for step in range(args.steps_per_epoch):
            gate = ('M', 'S')[step % 2]
            nulls = choose_nulls(gate, args.train_nulls, rng)
            (normal, pseudo), _ = sample_video_batch(
                support, groups, args.episode_frames, rng, device,
            )
            loss, detail = gate_step(
                predictor, model, normal, pseudo, gate, nulls, args,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite order-transport loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(gradient):
                raise FloatingPointError('non-finite order-transport gradient')
            optimizer.step()
            totals['loss'].append(float(loss.detach()))
            totals['grad_norm'].append(float(gradient.detach()))
            for key, value in detail.items():
                totals[f'{gate}_{key}'].append(float(value.detach()))
        row = {'epoch': epoch, **{key: float(np.mean(value))
                                  for key, value in totals.items()}}
        if epoch in args.milestones and query is not None:
            policy = evaluate_policy(predictor, model, query, query_groups, args, device)
            row['query'] = policy; milestones[str(epoch)] = policy
            print(f'  head ep{epoch:02d} loss={row["loss"]:.4f} '
                  f'query=L{policy["boundary"]} '
                  f'gain={[round(policy[x]["gain_probability"],4) for x in ("M","S")]}',
                  flush=True)
        history.append(row)
    return history, milestones


@torch.no_grad()
def evaluate_gate(predictor, model, cache, groups, gate, args, device):
    rng = np.random.RandomState(args.seed + (83000 if gate == 'M' else 84000))
    nulls = choose_nulls(gate, args.eval_nulls, rng)
    rows = []
    for video in sorted(groups):
        index = torch.as_tensor(groups[video], dtype=torch.long)
        normal = cache['normal'].index_select(0, index).to(device=device).float()
        pseudo = cache['pseudo'].index_select(0, index).to(device=device).float()
        _, detail = gate_step(predictor, model, normal, pseudo, gate, nulls, args)
        rows.append({'video': video, **{key: float(value) for key, value in detail.items()}})
    logit = float(np.mean([row['gain_logit'] for row in rows]))
    return {'gain_logit': logit, 'gain_probability': float(torch.sigmoid(torch.tensor(logit))),
            'target': float(np.mean([row['gain_target'] for row in rows])),
            'normal_real_flip': float(np.mean([row['normal_real_flip'] for row in rows])),
            'pseudo_real_flip': float(np.mean([row['pseudo_real_flip'] for row in rows])),
            'video_rows': rows, 'null_count': len(nulls)}


@torch.no_grad()
def evaluate_policy(predictor, model, cache, groups, args, device):
    model.eval(); predictor.eval()
    middle = evaluate_gate(predictor, model, cache, groups, 'M', args, device)
    shallow = evaluate_gate(predictor, model, cache, groups, 'S', args, device)
    boundary, probability = model.boundary_distribution(
        torch.tensor(middle['gain_logit']), torch.tensor(shallow['gain_logit']),
    )
    return {'boundary': boundary, 'selected_taps': list(range(boundary, 12)),
            'probabilities': {'D': float(probability[0]), 'MD': float(probability[1]),
                              'ALL': float(probability[2])},
            'M': middle, 'S': shallow}


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
    parser.add_argument('--pseudo_mode', choices=['cutpaste', 'multi'], default='multi')
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--predictor_epochs', type=int, default=10)
    parser.add_argument('--predictor_steps', type=int, default=30)
    parser.add_argument('--predictor_batch', type=int, default=24)
    parser.add_argument('--predictor_lr', type=float, default=1e-3)
    parser.add_argument('--predictor_checkpoint', default='')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--steps_per_epoch', type=int, default=30)
    parser.add_argument('--episode_frames', type=int, default=32)
    parser.add_argument('--pair_samples', type=int, default=256)
    parser.add_argument('--train_nulls', type=int, default=4)
    parser.add_argument('--eval_nulls', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--milestones', type=int, nargs='+', default=[5, 10, 20, 30])
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda': torch.cuda.set_device(device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = probe_path(args.probe_dir, args.scene); before = file_sha256(checkpoint)
    teacher, ocbe, student, cfg = load_probe(args.scene, args.probe_dir, device)
    paths, videos = split_video_balanced(all_scene_paths(args.scene),
                                         args.max_frames_per_video, args.seed)
    support = cache_relation_maps(paths['support'], teacher, ocbe, student, cfg, args, device)
    query = cache_relation_maps(paths['query'], teacher, ocbe, student, cfg, args, device)
    del teacher, ocbe, student
    support_groups, query_groups = video_indices(support['paths']), video_indices(query['paths'])
    predictor = CrossLayerPredictor(channels=args.channels).to(device)
    if args.predictor_checkpoint:
        saved = torch.load(args.predictor_checkpoint, map_location='cpu')
        predictor.load_state_dict(saved['predictor']); predictor.eval().requires_grad_(False)
        predictor_history = saved.get('result', {}).get('history', [])
    else:
        predictor_history = train_predictor(predictor, support, args, device)
    predictor_query = evaluate_predictor(predictor, query, device, args.predictor_batch,
                                         args.seed + 85000)
    print(f'[{args.scene}] predictor={predictor_query["normal_aligned_smooth_l1"]:.6g} '
          f'roll={predictor_query["normal_roll_loss_ratio"]:.3f}', flush=True)
    set_seed(args.seed + 86000)
    model = OrderTransportDLBM(predictor_channels=args.channels,
                               model_dim=args.model_dim, heads=args.heads).to(device)
    history, milestones = train_head(predictor, model, support, support_groups, args,
                                     device, query, query_groups)
    query_policy = evaluate_policy(predictor, model, query, query_groups, args, device)
    support_policy = evaluate_policy(predictor, model, support, support_groups, args, device)
    shifted = evaluate_policy(predictor, model, rolled_cache(query, args.seed + 87000),
                              query_groups, args, device)
    if before != file_sha256(checkpoint): raise RuntimeError('Probe checkpoint changed')
    result = {'format_version': 12, 'scene': args.scene,
              'method': 'episode_order_transport_structural_null_dlbm',
              'probe_checkpoint': str(checkpoint), 'probe_sha256': before,
              'probe_mutated': False, 'args': vars(args), 'videos': videos,
              'predictor_history': predictor_history, 'predictor_query': predictor_query,
              'history': history, 'milestones': milestones,
              'query_policy': query_policy, 'support_policy': support_policy,
              'shifted_query_policy': shifted}
    torch.save({'predictor': predictor.state_dict(), 'dlbm': model.state_dict(),
                'result': result}, out_dir / 'model.pth')
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'[{args.scene}] query=L{query_policy["boundary"]} '
          f'support=L{support_policy["boundary"]} shifted=L{shifted["boundary"]}', flush=True)


if __name__ == '__main__': main()
