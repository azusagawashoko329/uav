"""Train v021 raw-patch normal-order / pseudo-transport dual-axis DLBM."""
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
from experiment_DLBM.raw_transport_dual_axis_dlbm import RawTransportDualAxisDLBM
from experiment_DLBM.run_boundary_v001 import all_scene_paths, load_probe, split_video_balanced
from experiment_DLBM.train_boundary_v006 import cache_relation_maps
from experiment_DLBM.train_boundary_v007 import sample_video_batch, video_indices
from experiment_DLBM.train_boundary_v008 import (
    REAL_TRANSITIONS, group_score, sampled_pairs, structural_nulls,
)
from experiment_DLBM.train_episode_innovation_boundary import file_sha256


def choose_transition(step, rng):
    gate = ('M', 'S')[step % 2]
    if int(rng.randint(2)) == 0:
        return gate, *REAL_TRANSITIONS[gate], 'real'
    nulls = structural_nulls(gate)
    base, candidate = nulls[int(rng.randint(len(nulls)))]
    return gate, base, candidate, 'null'


def candidate_labels(maps, candidate, topk, first, second):
    score = group_score(maps, candidate, topk)
    return (score.index_select(0, first) > score.index_select(0, second)).to(maps.dtype)


def order_loss(model, maps, base, candidate, first, second, topk):
    target = candidate_labels(maps, candidate, topk, first, second)
    base_logit, opened_logit = model.order_logits(
        maps, base, candidate, first, second,
    )
    base_nll = F.binary_cross_entropy_with_logits(base_logit, target)
    opened_nll = F.binary_cross_entropy_with_logits(opened_logit, target)
    return 0.5 * (base_nll + opened_nll), {
        'base_nll': base_nll, 'opened_nll': opened_nll,
        'improvement': base_nll.detach() - opened_nll.detach(),
    }


def critic_loss(model, normal, pseudo, base, candidate):
    normal_logit = model.critic_logits(normal, base, candidate)
    pseudo_logit = model.critic_logits(pseudo, base, candidate)
    loss = 0.5 * (
        F.binary_cross_entropy_with_logits(normal_logit, torch.zeros_like(normal_logit))
        + F.binary_cross_entropy_with_logits(pseudo_logit, torch.ones_like(pseudo_logit))
    )
    return loss, normal_logit, pseudo_logit


def train_model(model, support, groups, args, device,
                query=None, query_groups=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    rng = np.random.RandomState(args.seed + 151000)
    history, milestones = [], {}
    for epoch in range(1, args.epochs + 1):
        model.train(); totals = defaultdict(list)
        for step in range(args.steps_per_epoch):
            if args.video_consistency:
                loss, detail = consistency_step(model, support, groups, args, device, rng, step)
                if not torch.isfinite(loss):
                    raise FloatingPointError('non-finite video-consistent raw-transport loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(gradient):
                    raise FloatingPointError('non-finite raw-transport gradient')
                optimizer.step()
                totals['loss'].append(float(loss.detach()))
                totals['order_loss'].append(float(detail['order'].detach()))
                totals['critic_loss'].append(float(detail['critic'].detach()))
                totals['video_consistency_loss'].append(float(detail['consistency'].detach()))
                totals['utility_spread'].append(float(detail['utility_spread'].detach()))
                totals['grad_norm'].append(float(gradient.detach()))
                continue
            gate, base, candidate, kind = choose_transition(step, rng)
            (normal, pseudo), _ = sample_video_batch(
                support, groups, args.episode_frames, rng, device,
            )
            first, second = sampled_pairs(len(normal), args.pair_samples, device)
            order, order_detail = order_loss(
                model, normal, base, candidate, first, second, args.topk,
            )
            critic, normal_logit, pseudo_logit = critic_loss(
                model, normal, pseudo, base, candidate,
            )
            loss = model.multitask_loss(order, critic)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite raw-transport loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(gradient):
                raise FloatingPointError('non-finite raw-transport gradient')
            optimizer.step()
            totals['loss'].append(float(loss.detach()))
            totals['order_loss'].append(float(order.detach()))
            totals['critic_loss'].append(float(critic.detach()))
            totals['grad_norm'].append(float(gradient.detach()))
            totals[f'{gate}_{kind}_base_nll'].append(float(order_detail['base_nll']))
            totals[f'{gate}_{kind}_critic_gap'].append(float(
                (pseudo_logit.mean() - normal_logit.mean()).detach(),
            ))
        row = {
            'epoch': epoch,
            **{key: float(np.mean(values)) for key, values in totals.items()},
            'axis_precision': [float(value) for value in model.axis_precision().detach()],
        }
        if epoch in args.milestones and query is not None:
            policy = evaluate_policy(model, query, query_groups, args, device)
            row['query'] = policy; milestones[str(epoch)] = policy
            print(
                f'  raw-dual ep{epoch:02d} loss={row["loss"]:.4f} '
                f'query=L{policy["boundary"]} '
                f'axes={[[round(v,5) for v in policy[g]["axes"]] for g in ("M", "S")]}',
                flush=True,
            )
        history.append(row)
    return history, milestones


def _video_batch(cache, groups, video, count, rng, device):
    available = groups[video]
    size = min(int(count), len(available))
    chosen = rng.choice(available, size=size, replace=False)
    index = torch.as_tensor(chosen, dtype=torch.long)
    return tuple(
        cache[name].index_select(0, index).to(device=device, dtype=torch.float32)
        for name in ('normal', 'pseudo')
    )


def consistency_step(model, support, groups, args, device, rng, step):
    """Same gate and null on two videos; only their learned utility must agree."""
    gate = ('M', 'S')[step % 2]
    real_base, real_candidate = REAL_TRANSITIONS[gate]
    nulls = structural_nulls(gate)
    null_base, null_candidate = nulls[int(rng.randint(len(nulls)))]
    videos = sorted(groups)
    chosen = rng.choice(len(videos), size=2, replace=False)
    order_losses, critic_losses, utilities = [], [], []
    precision = model.axis_precision()
    for video_index in chosen:
        normal, pseudo = _video_batch(
            support, groups, videos[int(video_index)],
            args.episode_frames, rng, device,
        )
        first, second = sampled_pairs(len(normal), args.pair_samples, device)
        real_order, real_detail = order_loss(
            model, normal, real_base, real_candidate, first, second, args.topk,
        )
        real_critic, _, _ = critic_loss(
            model, normal, pseudo, real_base, real_candidate,
        )
        null_order, null_detail = order_loss(
            model, normal, null_base, null_candidate, first, second, args.topk,
        )
        null_critic, _, _ = critic_loss(
            model, normal, pseudo, null_base, null_candidate,
        )
        order_losses.extend((real_order, null_order))
        critic_losses.extend((real_critic, null_critic))
        normal_novelty = real_detail['base_nll'] - null_detail['base_nll']
        pseudo_nuisance = null_critic - real_critic
        utilities.append(
            precision[0] * normal_novelty - precision[1] * pseudo_nuisance,
        )
    order = torch.stack(order_losses).mean()
    critic = torch.stack(critic_losses).mean()
    utility = torch.stack(utilities)
    consistency = (utility - utility.mean()).square().mean()
    loss = model.multitask_loss(order, critic, consistency)
    return loss, {
        'order': order, 'critic': critic, 'consistency': consistency,
        'utility_spread': utility.max() - utility.min(),
    }


@torch.no_grad()
def transition_metric(model, cache, groups, base, candidate, args, device):
    rows = []
    for video in sorted(groups):
        index = torch.as_tensor(groups[video], dtype=torch.long)
        normal = cache['normal'].index_select(0, index).to(device=device).float()
        pseudo = cache['pseudo'].index_select(0, index).to(device=device).float()
        first, second = sampled_pairs(len(normal), args.eval_pairs, device)
        _, order = order_loss(
            model, normal, base, candidate, first, second, args.topk,
        )
        critic, normal_logit, pseudo_logit = critic_loss(
            model, normal, pseudo, base, candidate,
        )
        rows.append({
            'video': video, 'normal_base_nll': float(order['base_nll']),
            'normal_opened_nll': float(order['opened_nll']),
            'critic_nll': float(critic),
            'critic_gap': float(pseudo_logit.mean() - normal_logit.mean()),
        })
    output = {'video_rows': rows}
    for key in rows[0]:
        if key != 'video':
            output[key] = float(np.mean([row[key] for row in rows]))
    return output


@torch.no_grad()
def evaluate_gate(model, cache, groups, gate, args, device):
    real = transition_metric(
        model, cache, groups, *REAL_TRANSITIONS[gate], args, device,
    )
    nulls = structural_nulls(gate)
    if len(nulls) > args.eval_nulls:
        rng = np.random.RandomState(args.seed + (152000 if gate == 'M' else 153000))
        chosen = rng.choice(len(nulls), size=args.eval_nulls, replace=False)
        nulls = [nulls[int(index)] for index in sorted(chosen)]
    null_rows = [
        transition_metric(model, cache, groups, base, candidate, args, device)
        for base, candidate in nulls
    ]
    null_base = float(np.mean([row['normal_base_nll'] for row in null_rows]))
    null_critic = float(np.mean([row['critic_nll'] for row in null_rows]))
    normal_novelty = real['normal_base_nll'] - null_base
    pseudo_nuisance = null_critic - real['critic_nll']
    return {
        'axes': [normal_novelty, pseudo_nuisance],
        'normal_novelty': normal_novelty, 'pseudo_nuisance': pseudo_nuisance,
        'real': real, 'null_normal_base_nll': null_base,
        'null_critic_nll': null_critic, 'null_count': len(nulls),
    }


@torch.no_grad()
def evaluate_policy(model, cache, groups, args, device):
    model.eval()
    middle = evaluate_gate(model, cache, groups, 'M', args, device)
    shallow = evaluate_gate(model, cache, groups, 'S', args, device)
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
    parser.add_argument('--predictor_checkpoint', default='')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps_per_epoch', type=int, default=20)
    parser.add_argument('--episode_frames', type=int, default=32)
    parser.add_argument('--pair_samples', type=int, default=256)
    parser.add_argument('--eval_pairs', type=int, default=2048)
    parser.add_argument('--eval_nulls', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--video_consistency', action='store_true')
    parser.add_argument('--milestones', type=int, nargs='+', default=[1, 3, 5, 10])
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
    full_cache = {
        'paths': [*support['paths'], *query['paths']],
        'normal': torch.cat((support['normal'], query['normal'])),
        'pseudo': torch.cat((support['pseudo'], query['pseudo'])),
    }
    full_groups = video_indices(full_cache['paths'])

    set_seed(args.seed + 154000)
    model = RawTransportDualAxisDLBM(
        topk=args.topk, channels=args.channels,
        model_dim=args.model_dim, heads=args.heads,
        video_consistency=args.video_consistency,
    ).to(device)
    history, milestones = train_model(
        model, support, support_groups, args, device, query, query_groups,
    )
    query_policy = evaluate_policy(model, query, query_groups, args, device)
    support_policy = evaluate_policy(model, support, support_groups, args, device)
    full_policy = evaluate_policy(model, full_cache, full_groups, args, device)
    if before != file_sha256(checkpoint):
        raise RuntimeError('Probe checkpoint changed')
    result = {
        'format_version': 22 if args.video_consistency else 21, 'scene': args.scene,
        'method': ('video_consistent_raw_transport_dual_axis_dlbm'
                   if args.video_consistency else
                   'raw_topk_patch_normal_order_pseudo_transport_dual_axis_dlbm'),
        'probe_checkpoint': str(checkpoint), 'probe_sha256': before,
        'probe_mutated': False, 'args': vars(args), 'videos': videos,
        'history': history, 'milestones': milestones,
        'query_policy': query_policy, 'support_policy': support_policy,
        'full_scene_policy': full_policy,
    }
    torch.save({'dlbm': model.state_dict(), 'result': result}, out_dir / 'model.pth')
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(
        f'[{args.scene}] full=L{full_policy["boundary"]} '
        f'query=L{query_policy["boundary"]} support=L{support_policy["boundary"]}',
        flush=True,
    )


if __name__ == '__main__':
    main()
