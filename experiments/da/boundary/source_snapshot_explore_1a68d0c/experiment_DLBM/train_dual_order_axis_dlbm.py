"""Train v018 with independent normal and pseudo order-transport axes."""
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
from experiment_DLBM.dual_order_axis_dlbm import DualOrderAxisDLBM
from experiment_DLBM.probe_cross_layer_predictor import (
    evaluate as evaluate_predictor, independent_layer_roll,
)
from experiment_DLBM.run_boundary_v001 import all_scene_paths, load_probe, split_video_balanced
from experiment_DLBM.train_boundary_v006 import cache_relation_maps
from experiment_DLBM.train_boundary_v007 import sample_video_batch, video_indices
from experiment_DLBM.train_boundary_v008 import REAL_TRANSITIONS, sampled_pairs, structural_nulls
from experiment_DLBM.train_episode_innovation_boundary import file_sha256
from experiment_DLBM.train_predictive_order_dlbm import transition_loss


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
    rng = np.random.RandomState(args.seed + 121000)
    history, milestones = [], {}
    for epoch in range(1, args.epochs + 1):
        model.train(); totals = defaultdict(list)
        for step in range(args.steps_per_epoch):
            gate, base, candidate, kind = choose_transition(step, rng)
            (normal, pseudo), _ = sample_video_batch(
                support, groups, args.episode_frames, rng, device,
            )
            first, second = sampled_pairs(len(normal), args.pair_samples, device)
            normal_loss, normal_detail = transition_loss(
                predictor, model.normal_axis, normal, base, candidate,
                first, second, args.topk,
            )
            pseudo_loss, pseudo_detail = transition_loss(
                predictor, model.pseudo_axis, pseudo, base, candidate,
                first, second, args.topk,
            )
            loss = model.multitask_loss(normal_loss, pseudo_loss)
            if not torch.isfinite(loss): raise FloatingPointError('non-finite dual-order loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(gradient): raise FloatingPointError('non-finite dual-order gradient')
            optimizer.step()
            totals['loss'].append(float(loss.detach()))
            totals['normal_loss'].append(float(normal_loss.detach()))
            totals['pseudo_loss'].append(float(pseudo_loss.detach()))
            totals['grad_norm'].append(float(gradient.detach()))
            totals[f'{gate}_{kind}_normal_base_nll'].append(
                float(normal_detail['base_nll'].detach()),
            )
            totals[f'{gate}_{kind}_pseudo_base_nll'].append(
                float(pseudo_detail['base_nll'].detach()),
            )
        row = {'epoch': epoch, **{
            key: float(np.mean(values)) for key, values in totals.items()
        }, 'axis_precision': [float(value) for value in model.axis_precision().detach()]}
        if epoch in args.milestones and query is not None:
            policy = evaluate_policy(predictor, model, query, query_groups, args, device)
            row['query'] = policy; milestones[str(epoch)] = policy
            print(
                f'  dual-order ep{epoch:02d} loss={row["loss"]:.4f} '
                f'query=L{policy["boundary"]} '
                f'axis={[[round(v,5) for v in policy[g]["axes"]] for g in ("M","S")]}',
                flush=True,
            )
        history.append(row)
    return history, milestones


@torch.no_grad()
def axis_metric(predictor, axis, maps, groups, base, candidate, args, device):
    rows = []
    for video in sorted(groups):
        index = torch.as_tensor(groups[video], dtype=torch.long)
        values = maps.index_select(0, index).to(device=device).float()
        first, second = sampled_pairs(len(values), args.eval_pairs, device)
        _, detail = transition_loss(
            predictor, axis, values, base, candidate, first, second, args.topk,
        )
        rows.append({
            'video': video, 'base_nll': float(detail['base_nll']),
            'opened_nll': float(detail['opened_nll']),
            'improvement': float(detail['improvement']),
        })
    return {
        'video_rows': rows,
        **{key: float(np.mean([row[key] for row in rows]))
           for key in ('base_nll', 'opened_nll', 'improvement')},
    }


@torch.no_grad()
def transition_metric(predictor, model, cache, groups, base, candidate, args, device):
    return {
        'normal': axis_metric(
            predictor, model.normal_axis, cache['normal'], groups,
            base, candidate, args, device,
        ),
        'pseudo': axis_metric(
            predictor, model.pseudo_axis, cache['pseudo'], groups,
            base, candidate, args, device,
        ),
    }


@torch.no_grad()
def evaluate_gate(predictor, model, cache, groups, gate, args, device):
    real = transition_metric(
        predictor, model, cache, groups, *REAL_TRANSITIONS[gate], args, device,
    )
    nulls = structural_nulls(gate)
    if len(nulls) > args.eval_nulls:
        rng = np.random.RandomState(args.seed + (122000 if gate == 'M' else 123000))
        selected = rng.choice(len(nulls), size=args.eval_nulls, replace=False)
        nulls = [nulls[int(index)] for index in sorted(selected)]
    null_rows = [
        transition_metric(predictor, model, cache, groups, base, candidate, args, device)
        for base, candidate in nulls
    ]
    axes, null_base = [], {}
    for branch in ('normal', 'pseudo'):
        null_base[branch] = float(np.mean([
            row[branch]['base_nll'] for row in null_rows
        ]))
        axes.append(real[branch]['base_nll'] - null_base[branch])
    return {
        'axes': axes, 'normal_novelty': axes[0], 'pseudo_novelty': axes[1],
        'real': real, 'null_base_nll': null_base, 'null_count': len(nulls),
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
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda': torch.cuda.set_device(device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = probe_path(args.probe_dir, args.scene); before = file_sha256(checkpoint)
    teacher, ocbe, student, cfg = load_probe(args.scene, args.probe_dir, device)
    paths, videos = split_video_balanced(
        all_scene_paths(args.scene), args.max_frames_per_video, args.seed,
    )
    support = cache_relation_maps(paths['support'], teacher, ocbe, student, cfg, args, device)
    query = cache_relation_maps(paths['query'], teacher, ocbe, student, cfg, args, device)
    del teacher, ocbe, student
    support_groups, query_groups = video_indices(support['paths']), video_indices(query['paths'])
    full_cache = {'paths': [*support['paths'], *query['paths']],
                  'normal': torch.cat((support['normal'], query['normal'])),
                  'pseudo': torch.cat((support['pseudo'], query['pseudo']))}
    full_groups = video_indices(full_cache['paths'])
    saved = torch.load(args.predictor_checkpoint, map_location='cpu')
    predictor = CrossLayerPredictor(channels=args.channels).to(device)
    predictor.load_state_dict(saved['predictor']); predictor.eval().requires_grad_(False)
    predictor_query = evaluate_predictor(
        predictor, query, device, args.predictor_batch, args.seed + 124000,
    )
    print(
        f'[{args.scene}] predictor={predictor_query["normal_aligned_smooth_l1"]:.6g} '
        f'roll={predictor_query["normal_roll_loss_ratio"]:.3f}', flush=True,
    )
    set_seed(args.seed + 125000)
    model = DualOrderAxisDLBM(
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
        predictor, model, rolled_cache(query, args.seed + 126000),
        query_groups, args, device,
    )
    if before != file_sha256(checkpoint): raise RuntimeError('Probe checkpoint changed')
    result = {
        'format_version': 18, 'scene': args.scene,
        'method': 'independent_normal_pseudo_order_transport_axes_dlbm',
        'probe_checkpoint': str(checkpoint), 'probe_sha256': before,
        'probe_mutated': False, 'args': vars(args), 'videos': videos,
        'predictor_query': predictor_query, 'history': history,
        'milestones': milestones, 'query_policy': query_policy,
        'support_policy': support_policy, 'full_scene_policy': full_policy,
        'shifted_query_policy': shifted,
    }
    torch.save({'predictor': predictor.state_dict(), 'dlbm': model.state_dict(),
                'result': result}, out_dir / 'model.pth')
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(
        f'[{args.scene}] full=L{full_policy["boundary"]} '
        f'query=L{query_policy["boundary"]} support=L{support_policy["boundary"]} '
        f'shifted=L{shifted["boundary"]}', flush=True,
    )


if __name__ == '__main__':
    main()
