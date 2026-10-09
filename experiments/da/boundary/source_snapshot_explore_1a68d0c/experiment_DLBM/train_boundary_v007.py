"""Train DLBM v007 on learned real-opening versus shuffled-null evidence."""
import argparse
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
from experiment_DLBM.dlbm_v007 import LearnedOpeningTransition, boundary_from_evidence
from experiment_DLBM.run_boundary_v001 import all_scene_paths, load_probe, split_video_balanced
from experiment_DLBM.train_boundary_v006 import cache_relation_maps


def video_indices(paths):
    result = defaultdict(list)
    for index, path in enumerate(paths):
        result[str(Path(path).parent)].append(index)
    return dict(result)


def sample_video_batch(cache, groups, count, rng, device):
    video = sorted(groups)[int(rng.randint(len(groups)))]
    available = groups[video]
    size = min(int(count), len(available))
    chosen = rng.choice(available, size=size, replace=False)
    index = torch.as_tensor(chosen, dtype=torch.long)
    maps = tuple(
        cache[name].index_select(0, index).to(device=device, dtype=torch.float32,
                                             non_blocking=True)
        for name in ('normal', 'pseudo')
    )
    return maps, video


def derangement(size, device):
    if size < 2:
        raise ValueError('derangement needs at least two items')
    original = torch.arange(size, device=device)
    for _ in range(16):
        permutation = torch.randperm(size, device=device)
        if not torch.eq(permutation, original).any():
            return permutation
    return original.roll(1)


def layer_topk_scores(maps, topk):
    return maps.topk(topk, dim=-1).values.mean(dim=-1)


def opening_scores(maps, boundary, topk):
    scores = layer_topk_scores(maps, topk)
    base = scores[:, -1]
    opened = scores[:, int(boundary):].mean(dim=1)
    return base, opened


def order_loss(model, embedding, maps, boundary, topk, pair_samples):
    first, second = torch.triu_indices(
        maps.shape[0], maps.shape[0], offset=1, device=maps.device,
    )
    if len(first) > pair_samples:
        selected = torch.randperm(len(first), device=maps.device)[:pair_samples]
        first, second = first[selected], second[selected]
    target_base, target_open = opening_scores(maps, boundary, topk)
    predicted_base, predicted_open = model.order_scores(embedding)
    base_label = (target_base.index_select(0, first)
                  > target_base.index_select(0, second)).to(maps.dtype)
    open_label = (target_open.index_select(0, first)
                  > target_open.index_select(0, second)).to(maps.dtype)
    base_logit = predicted_base.index_select(0, first) - predicted_base.index_select(0, second)
    open_logit = predicted_open.index_select(0, first) - predicted_open.index_select(0, second)
    loss = 0.5 * (
        F.binary_cross_entropy_with_logits(base_logit, base_label)
        + F.binary_cross_entropy_with_logits(open_logit, open_label)
    )
    return loss


def train_step(model, normal, pseudo, boundary, args):
    permutation = derangement(len(normal), normal.device)
    real_normal, real_pseudo = model.encode(normal, pseudo, boundary)
    null_normal, null_pseudo = model.encode(
        normal, pseudo, boundary, opened_permutation=permutation,
    )
    real_logits = model.realness_logits(real_normal, real_pseudo)
    null_logits = model.realness_logits(null_normal, null_pseudo)
    relation_loss = 0.5 * (
        F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        + F.binary_cross_entropy_with_logits(null_logits, torch.zeros_like(null_logits))
    )
    normal_order = order_loss(
        model, real_normal, normal, boundary, args.topk, args.pair_samples,
    )
    pseudo_order = order_loss(
        model, real_pseudo, pseudo, boundary, args.topk, args.pair_samples,
    )
    # One transition objective: coherent real/null evidence plus dense ordering guidance.
    loss = relation_loss + 0.5 * (normal_order + pseudo_order)
    return loss, {
        'relation': float(relation_loss.detach()),
        'normal_order': float(normal_order.detach()),
        'pseudo_order': float(pseudo_order.detach()),
    }


def pair_flip_rate(base_score, open_score, maximum):
    first, second = torch.triu_indices(
        len(base_score), len(base_score), offset=1, device=base_score.device,
    )
    if len(first) > maximum:
        selected = torch.randperm(len(first), device=base_score.device)[:maximum]
        first, second = first[selected], second[selected]
    base_delta = base_score.index_select(0, first) - base_score.index_select(0, second)
    open_delta = open_score.index_select(0, first) - open_score.index_select(0, second)
    return ((base_delta * open_delta) < 0).to(base_score.dtype).mean()


@torch.no_grad()
def evaluate_boundary(model, cache, groups, boundary, args, device):
    relation_evidence, real_values, null_values = [], [], []
    normal_order_evidence, pseudo_order_evidence = [], []
    predicted_normal_real, predicted_pseudo_real = [], []
    actual_normal_real, actual_pseudo_real = [], []
    for video in sorted(groups):
        index = torch.as_tensor(groups[video], dtype=torch.long)
        normal = cache['normal'].index_select(0, index).to(device=device, dtype=torch.float32)
        pseudo = cache['pseudo'].index_select(0, index).to(device=device, dtype=torch.float32)
        real_normal, real_pseudo = model.encode(normal, pseudo, boundary)
        real = model.realness_logits(real_normal, real_pseudo).mean()
        real_normal_base, real_normal_open = model.order_scores(real_normal)
        real_pseudo_base, real_pseudo_open = model.order_scores(real_pseudo)
        normal_real_flip = pair_flip_rate(
            real_normal_base, real_normal_open, args.eval_pairs,
        )
        pseudo_real_flip = pair_flip_rate(
            real_pseudo_base, real_pseudo_open, args.eval_pairs,
        )
        target_normal_base, target_normal_open = opening_scores(normal, boundary, args.topk)
        target_pseudo_base, target_pseudo_open = opening_scores(pseudo, boundary, args.topk)
        actual_normal_real.append(pair_flip_rate(
            target_normal_base, target_normal_open, args.eval_pairs,
        ))
        actual_pseudo_real.append(pair_flip_rate(
            target_pseudo_base, target_pseudo_open, args.eval_pairs,
        ))
        null, normal_null_flip, pseudo_null_flip = [], [], []
        for _ in range(args.null_repeats):
            permutation = derangement(len(normal), device)
            null_normal, null_pseudo = model.encode(
                normal, pseudo, boundary, opened_permutation=permutation,
            )
            null.append(model.realness_logits(null_normal, null_pseudo).mean())
            null_normal_base, null_normal_open = model.order_scores(null_normal)
            null_pseudo_base, null_pseudo_open = model.order_scores(null_pseudo)
            normal_null_flip.append(pair_flip_rate(
                null_normal_base, null_normal_open, args.eval_pairs,
            ))
            pseudo_null_flip.append(pair_flip_rate(
                null_pseudo_base, null_pseudo_open, args.eval_pairs,
            ))
        null = torch.stack(null).mean()
        normal_null_flip = torch.stack(normal_null_flip).mean()
        pseudo_null_flip = torch.stack(pseudo_null_flip).mean()
        relation_evidence.append(real - null)
        normal_order_evidence.append(normal_real_flip - normal_null_flip)
        pseudo_order_evidence.append(pseudo_real_flip - pseudo_null_flip)
        predicted_normal_real.append(normal_real_flip)
        predicted_pseudo_real.append(pseudo_real_flip)
        real_values.append(real)
        null_values.append(null)
    return {
        # Boundary selection is driven only by learned Normal order change versus null.
        'evidence': float(torch.stack(normal_order_evidence).mean()),
        'normal_order_evidence': float(torch.stack(normal_order_evidence).mean()),
        'pseudo_order_evidence': float(torch.stack(pseudo_order_evidence).mean()),
        'relation_evidence': float(torch.stack(relation_evidence).mean()),
        'real_logit': float(torch.stack(real_values).mean()),
        'null_logit': float(torch.stack(null_values).mean()),
        'predicted_normal_flip_rate': float(torch.stack(predicted_normal_real).mean()),
        'predicted_pseudo_flip_rate': float(torch.stack(predicted_pseudo_real).mean()),
        'actual_normal_flip_rate': float(torch.stack(actual_normal_real).mean()),
        'actual_pseudo_flip_rate': float(torch.stack(actual_pseudo_real).mean()),
    }


@torch.no_grad()
def evaluate(model, cache, groups, args, device):
    model.eval()
    rows = [evaluate_boundary(model, cache, groups, boundary, args, device)
            for boundary in range(11)]
    evidence = torch.tensor([row['evidence'] for row in rows])
    boundary, complete = boundary_from_evidence(evidence)
    return {
        'boundary': boundary,
        'selected_taps': list(range(boundary, 12)),
        'boundary_rows': {str(index): row for index, row in enumerate(rows)},
        'complete_evidence': [float(value) for value in complete],
        'max_evidence': float(complete.max()),
        'evidence_spread': float(complete.max() - complete.min()),
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
    parser.add_argument('--pair_samples', type=int, default=256)
    parser.add_argument('--eval_pairs', type=int, default=4096)
    parser.add_argument('--null_repeats', type=int, default=8)
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
    torch.manual_seed(args.seed + 19000)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed + 19000)
        torch.cuda.reset_peak_memory_stats(device)
    support = cache_relation_maps(paths['support'], teacher, ocbe, student, cfg, args, device)
    query = cache_relation_maps(paths['query'], teacher, ocbe, student, cfg, args, device)
    del teacher, ocbe, student
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    support_groups = video_indices(support['paths'])
    query_groups = video_indices(query['paths'])
    model = LearnedOpeningTransition(
        num_layers=12, topk=args.topk, model_dim=args.model_dim, heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    rng = np.random.RandomState(args.seed + 20000)
    history, milestones = [], {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, relation_losses, normal_orders, pseudo_orders, gradients = [], [], [], [], []
        order = rng.permutation(11)
        for step in range(args.steps_per_epoch):
            boundary = int(order[step % 11])
            (normal, pseudo), _ = sample_video_batch(
                support, support_groups, args.frame_batch, rng, device,
            )
            loss, detail = train_step(model, normal, pseudo, boundary, args)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite transition loss epoch={epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(gradient):
                raise FloatingPointError(f'non-finite gradient epoch={epoch}')
            optimizer.step()
            losses.append(float(loss.detach()))
            relation_losses.append(detail['relation'])
            normal_orders.append(detail['normal_order'])
            pseudo_orders.append(detail['pseudo_order'])
            gradients.append(float(gradient.detach()))
        row = {
            'epoch': epoch, 'loss': float(np.mean(losses)),
            'relation_loss': float(np.mean(relation_losses)),
            'normal_order_loss': float(np.mean(normal_orders)),
            'pseudo_order_loss': float(np.mean(pseudo_orders)),
            'grad_norm': float(np.mean(gradients)),
        }
        if epoch in args.milestones:
            metrics = evaluate(model, query, query_groups, args, device)
            row['query'] = metrics
            milestones[str(epoch)] = metrics
            torch.save({'dlbm': model.state_dict(), 'args': vars(args), 'epoch': epoch,
                        'query': metrics}, checkpoint_dir / f'ep{epoch:03d}.pth')
            print(
                f'[{args.scene}] ep{epoch:02d}/{args.epochs} loss={row["loss"]:.4f} '
                f'relation={row["relation_loss"]:.4f} orderN={row["normal_order_loss"]:.4f} '
                f'boundary=L{metrics["boundary"]} evidence={metrics["max_evidence"]:.4f}',
                flush=True,
            )
        history.append(row)
    final = evaluate(model, query, query_groups, args, device)
    support_final = evaluate(model, support, support_groups, args, device)
    output = {
        'format_version': 7,
        'scene': args.scene,
        'method': 'learned_matched_vs_shuffled_dense_dual_order_transition',
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
        f'support=L{support_final["boundary"]} spread={final["evidence_spread"]:.4f} '
        f'peak={output["peak_gpu_gib"]:.2f} GiB', flush=True,
    )


if __name__ == '__main__':
    main()
