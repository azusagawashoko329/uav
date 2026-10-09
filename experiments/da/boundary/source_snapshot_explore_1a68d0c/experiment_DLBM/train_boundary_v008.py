"""Train DLBM v008 to estimate null-calibrated coarse opening-order gain."""
import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_DLBM.common import SCENES, probe_path, set_seed, write_json
from experiment_DLBM.dlbm_v008 import NeuralOpeningOrder, cumulative_coarse_policy
from experiment_DLBM.run_boundary_v001 import all_scene_paths, load_probe, split_video_balanced
from experiment_DLBM.train_boundary_v006 import cache_relation_maps
from experiment_DLBM.train_boundary_v007 import sample_video_batch, video_indices


D = tuple(range(8, 12))
MD = tuple(range(4, 12))
ALL = tuple(range(12))
REAL_TRANSITIONS = {'M': (D, MD), 'S': (MD, ALL)}


def structural_nulls(gate):
    """Same expansion ratio as a real opening, using only the current deep set."""
    if gate == 'M':
        return [(base, D) for base in itertools.combinations(D, 2)]
    if gate == 'S':
        return [(base, MD) for base in itertools.combinations(MD, 4)]
    raise ValueError(f'unknown gate {gate}')


def group_score(maps, layers, topk):
    index = torch.as_tensor(layers, device=maps.device, dtype=torch.long)
    selected = maps.index_select(1, index)
    return selected.topk(topk, dim=-1).values.mean(dim=-1).mean(dim=-1)


def sampled_pairs(frames, maximum, device):
    first, second = torch.triu_indices(frames, frames, offset=1, device=device)
    if len(first) > maximum:
        chosen = torch.randperm(len(first), device=device)[:maximum]
        first, second = first[chosen], second[chosen]
    return first, second


def transition_targets(maps, base_layers, candidate_layers, topk, first, second):
    base = group_score(maps, base_layers, topk)
    candidate = group_score(maps, candidate_layers, topk)
    base_delta = base.index_select(0, first) - base.index_select(0, second)
    candidate_delta = candidate.index_select(0, first) - candidate.index_select(0, second)
    return base, candidate, ((base_delta * candidate_delta) < 0).to(maps.dtype)


def balanced_flip_loss(logits, labels, maximum):
    positive = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
    negative = torch.nonzero(labels <= 0.5, as_tuple=False).flatten()
    if not len(positive) or not len(negative):
        count = min(len(labels), maximum)
        selected = torch.randperm(len(labels), device=labels.device)[:count]
    else:
        count = min(len(positive), len(negative), maximum // 2)
        positive = positive[torch.randperm(len(positive), device=labels.device)[:count]]
        negative = negative[torch.randperm(len(negative), device=labels.device)[:count]]
        selected = torch.cat((positive, negative))
    return F.binary_cross_entropy_with_logits(logits.index_select(0, selected),
                                               labels.index_select(0, selected))


def branch_loss(model, maps, base_layers, candidate_layers, args):
    base_embedding, candidate_embedding = model.encode_transition(
        maps, base_layers, candidate_layers,
    )
    first, second = sampled_pairs(len(maps), args.pair_samples, maps.device)
    base, candidate, flip = transition_targets(
        maps, base_layers, candidate_layers, args.topk, first, second,
    )
    predicted_base, predicted_candidate = model.order_scores(
        base_embedding, candidate_embedding,
    )
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
    flip_logits = model.flip_logits(base_embedding, candidate_embedding, first, second)
    flip_loss = balanced_flip_loss(flip_logits, flip, args.pair_samples)
    return rank, flip_loss, float(flip.mean())


def choose_training_transition(step, rng):
    gate = ('M', 'S')[step % 2]
    if int(rng.randint(2)) == 0:
        return gate, *REAL_TRANSITIONS[gate], False
    nulls = structural_nulls(gate)
    base, candidate = nulls[int(rng.randint(len(nulls)))]
    return gate, base, candidate, True


def train_step(model, normal, pseudo, base_layers, candidate_layers, args):
    normal_rank, normal_flip, normal_rate = branch_loss(
        model, normal, base_layers, candidate_layers, args,
    )
    pseudo_rank, pseudo_flip, pseudo_rate = branch_loss(
        model, pseudo, base_layers, candidate_layers, args,
    )
    loss = 0.25 * (normal_rank + normal_flip + pseudo_rank + pseudo_flip)
    return loss, {
        'normal_rank': float(normal_rank.detach()),
        'normal_flip': float(normal_flip.detach()),
        'pseudo_rank': float(pseudo_rank.detach()),
        'pseudo_flip': float(pseudo_flip.detach()),
        'normal_flip_rate': normal_rate,
        'pseudo_flip_rate': pseudo_rate,
    }


@torch.no_grad()
def transition_metric(model, cache, groups, base_layers, candidate_layers, args, device):
    predicted_normal, actual_normal = [], []
    predicted_pseudo, actual_pseudo = [], []
    for video in sorted(groups):
        index = torch.as_tensor(groups[video], dtype=torch.long)
        normal = cache['normal'].index_select(0, index).to(device=device, dtype=torch.float32)
        pseudo = cache['pseudo'].index_select(0, index).to(device=device, dtype=torch.float32)
        for maps, predicted, actual in (
                (normal, predicted_normal, actual_normal),
                (pseudo, predicted_pseudo, actual_pseudo)):
            base_embedding, candidate_embedding = model.encode_transition(
                maps, base_layers, candidate_layers,
            )
            first, second = sampled_pairs(len(maps), args.eval_pairs, device)
            _, _, target = transition_targets(
                maps, base_layers, candidate_layers, args.topk, first, second,
            )
            logits = model.flip_logits(base_embedding, candidate_embedding, first, second)
            predicted.append(torch.sigmoid(logits).mean())
            actual.append(target.mean())
    return {
        'predicted_normal_flip': float(torch.stack(predicted_normal).mean()),
        'actual_normal_flip': float(torch.stack(actual_normal).mean()),
        'predicted_pseudo_flip': float(torch.stack(predicted_pseudo).mean()),
        'actual_pseudo_flip': float(torch.stack(actual_pseudo).mean()),
    }


@torch.no_grad()
def evaluate_gate(model, cache, groups, gate, args, device):
    real = transition_metric(model, cache, groups, *REAL_TRANSITIONS[gate], args, device)
    nulls = structural_nulls(gate)
    if len(nulls) > args.max_nulls:
        rng = np.random.RandomState(args.seed + (800 if gate == 'M' else 900))
        chosen = rng.choice(len(nulls), size=args.max_nulls, replace=False)
        nulls = [nulls[int(index)] for index in sorted(chosen)]
    rows = [transition_metric(model, cache, groups, base, candidate, args, device)
            for base, candidate in nulls]
    result = dict(real)
    for branch in ('normal', 'pseudo'):
        key = f'predicted_{branch}_flip'
        actual_key = f'actual_{branch}_flip'
        predicted_null = torch.tensor([row[key] for row in rows])
        actual_null = torch.tensor([row[actual_key] for row in rows])
        result[f'predicted_{branch}_null_median'] = float(predicted_null.median())
        result[f'actual_{branch}_null_median'] = float(actual_null.median())
        result[f'{branch}_gain'] = result[key] - float(predicted_null.median())
        result[f'actual_{branch}_gain'] = result[actual_key] - float(actual_null.median())
    result['null_count'] = len(rows)
    return result


@torch.no_grad()
def evaluate(model, cache, groups, args, device):
    model.eval()
    middle = evaluate_gate(model, cache, groups, 'M', args, device)
    shallow = evaluate_gate(model, cache, groups, 'S', args, device)
    boundary, utility = cumulative_coarse_policy(
        torch.tensor(middle['normal_gain']), torch.tensor(shallow['normal_gain']),
    )
    return {
        'boundary': boundary,
        'selected_taps': list(range(boundary, 12)),
        'middle': middle,
        'shallow': shallow,
        'utilities': {'L8': float(utility[0]), 'L4': float(utility[1]),
                      'L0': float(utility[2])},
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
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--steps_per_epoch', type=int, default=44)
    parser.add_argument('--frame_batch', type=int, default=48)
    parser.add_argument('--pair_samples', type=int, default=512)
    parser.add_argument('--eval_pairs', type=int, default=4096)
    parser.add_argument('--max_nulls', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--milestones', type=int, nargs='+', default=[5, 10, 20, 30])
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
        raise ValueError('DLBM and Probe top-k must match')
    paths, videos = split_video_balanced(
        all_scene_paths(args.scene), args.max_frames_per_video, args.seed,
    )
    torch.manual_seed(args.seed + 21000)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed + 21000)
        torch.cuda.reset_peak_memory_stats(device)
    support = cache_relation_maps(paths['support'], teacher, ocbe, student, cfg, args, device)
    query = cache_relation_maps(paths['query'], teacher, ocbe, student, cfg, args, device)
    del teacher, ocbe, student
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    support_groups = video_indices(support['paths'])
    query_groups = video_indices(query['paths'])
    model = NeuralOpeningOrder(
        num_layers=12, topk=args.topk, model_dim=args.model_dim, heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    rng = np.random.RandomState(args.seed + 22000)
    history, milestones = [], {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = defaultdict(list)
        for step in range(args.steps_per_epoch):
            gate, base_layers, candidate_layers, is_null = choose_training_transition(step, rng)
            (normal, pseudo), _ = sample_video_batch(
                support, support_groups, args.frame_batch, rng, device,
            )
            loss, detail = train_step(
                model, normal, pseudo, base_layers, candidate_layers, args,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss epoch={epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(gradient):
                raise FloatingPointError(f'non-finite gradient epoch={epoch}')
            optimizer.step()
            totals['loss'].append(float(loss.detach()))
            totals['grad_norm'].append(float(gradient.detach()))
            totals['null_fraction'].append(float(is_null))
            for key, value in detail.items():
                totals[key].append(value)
        row = {'epoch': epoch, **{key: float(np.mean(value))
                                  for key, value in totals.items()}}
        if epoch in args.milestones:
            metrics = evaluate(model, query, query_groups, args, device)
            row['query'] = metrics
            milestones[str(epoch)] = metrics
            torch.save({'dlbm': model.state_dict(), 'args': vars(args), 'epoch': epoch,
                        'query': metrics}, checkpoint_dir / f'ep{epoch:03d}.pth')
            print(
                f'[{args.scene}] ep{epoch:02d}/{args.epochs} loss={row["loss"]:.4f} '
                f'rankN={row["normal_rank"]:.4f} flipN={row["normal_flip"]:.4f} '
                f'boundary=L{metrics["boundary"]}', flush=True,
            )
        history.append(row)
    final = evaluate(model, query, query_groups, args, device)
    support_final = evaluate(model, support, support_groups, args, device)
    output = {
        'format_version': 8,
        'scene': args.scene,
        'method': 'neural_opening_order_with_structural_layer_null',
        'probe_checkpoint': str(probe_path(args.probe_dir, args.scene)),
        'probe_mutated': False,
        'args': vars(args),
        'split': {'support_frames': len(support['paths']), 'query_frames': len(query['paths']),
                  'videos': videos},
        'history': history,
        'milestones': milestones,
        'final_policy': final,
        'support_policy': support_final,
        'model_parameters': sum(parameter.numel() for parameter in model.parameters()),
        'peak_gpu_gib': (torch.cuda.max_memory_allocated(device) / 2**30
                         if device.type == 'cuda' else 0.0),
    }
    torch.save({'dlbm': model.state_dict(), 'result': output}, out_dir / 'dlbm_final.pth')
    write_json(out_dir / 'result.json', output)
    print(
        f'[{args.scene}] final boundary=L{final["boundary"]} '
        f'support=L{support_final["boundary"]} peak={output["peak_gpu_gib"]:.2f} GiB',
        flush=True,
    )


if __name__ == '__main__':
    main()
