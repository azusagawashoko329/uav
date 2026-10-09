"""Validate a learned, spatial opening target before fitting another boundary predictor.

Each coarse suffix is normal-only micro-adapted from the reusable Probe10 checkpoint.  A single
cross-layer evaluator is then trained on support CutPaste masks for all suffixes and evaluated on
held-out sub-videos.  External artificial-Gate boundaries are post-hoc sanity labels only.
"""
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_DLBM.common import SCENES, probe_path, set_seed, write_json
from experiment_DLBM.dlbm_v004 import SharedLayerPatchLocalizer
from experiment_DLBM.run_boundary_v001 import (PathFrameDataset, all_scene_paths, load_probe,
                                                split_video_balanced)
from model import build_fixed_suffix_ocbe, build_fixed_suffix_student, topk_loss


def cutpaste_with_mask(batch, min_area, max_area):
    """Within-frame CutPaste plus its exact pixel support; never mixes frames or scenes."""
    if batch.ndim != 4:
        raise ValueError(f'expected BCHW batch, got {tuple(batch.shape)}')
    if not 0 < min_area <= max_area < 1:
        raise ValueError('expected 0 < min_area <= max_area < 1')
    output = batch.clone()
    mask = batch.new_zeros((len(batch), 1, batch.shape[-2], batch.shape[-1]))
    _, _, height, width = batch.shape
    for index in range(len(batch)):
        area = float(torch.empty((), device=batch.device).uniform_(min_area, max_area)) * height * width
        aspect = math.exp(float(torch.empty((), device=batch.device).uniform_(math.log(0.5), math.log(2))))
        patch_h = max(1, min(height - 1, round(math.sqrt(area / aspect))))
        patch_w = max(1, min(width - 1, round(math.sqrt(area * aspect))))
        source_y = int(torch.randint(0, height - patch_h + 1, (), device=batch.device))
        source_x = int(torch.randint(0, width - patch_w + 1, (), device=batch.device))
        target_y = int(torch.randint(0, height - patch_h + 1, (), device=batch.device))
        target_x = int(torch.randint(0, width - patch_w + 1, (), device=batch.device))
        source = batch[index, :, source_y:source_y + patch_h, source_x:source_x + patch_w].clone()
        if bool(torch.rand((), device=batch.device) < 0.5):
            source = source.flip(-1)
        output[index, :, target_y:target_y + patch_h, target_x:target_x + patch_w] = source
        mask[index, :, target_y:target_y + patch_h, target_x:target_x + patch_w] = 1
    return output, mask


def patch_discrepancy_maps(teacher_taps, student_recons, boundary, total_layers=12):
    """Raw patch distances padded into absolute layer positions, returned as (B,L,P)."""
    if len(teacher_taps) != len(student_recons) or len(teacher_taps) != total_layers - boundary:
        raise ValueError('suffix feature count mismatch')
    maps = torch.stack([
        1.0 - F.cosine_similarity(teacher_map, student_map, dim=-1)
        for teacher_map, student_map in zip(teacher_taps, student_recons)
    ], dim=1)
    padded = maps.new_zeros((len(maps), total_layers, maps.shape[-1]))
    padded[:, boundary:] = maps
    valid = torch.zeros((len(maps), total_layers), dtype=torch.bool, device=maps.device)
    valid[:, boundary:] = True
    return padded, valid


def pixel_to_patch_mask(pixel_mask, patches):
    grid = math.isqrt(int(patches))
    if grid * grid != patches:
        raise ValueError(f'patch count must be a square grid, got {patches}')
    # Fractional overlap is a parameter-free soft target and preserves boundary patches.
    return F.adaptive_avg_pool2d(pixel_mask.float(), (grid, grid)).flatten(1)


def make_loader(paths, cfg, teacher, batch_size, workers, shuffle=False, seed=42):
    return DataLoader(
        PathFrameDataset(paths, cfg['img'], teacher.img_mean, teacher.img_std),
        batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=torch.cuda.is_available(), drop_last=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def adapt_suffix(boundary, support_paths, teacher, probe_ocbe, probe_student, cfg, args, device):
    ocbe = build_fixed_suffix_ocbe(probe_ocbe, boundary).to(device).train()
    student = build_fixed_suffix_student(probe_student, boundary).to(device).train()
    loader = make_loader(
        support_paths, cfg, teacher, args.train_bs, args.workers, shuffle=True,
        seed=args.seed + boundary,
    )
    optimizer = torch.optim.AdamW(
        list(ocbe.parameters()) + list(student.parameters()), lr=args.lr, weight_decay=args.wd,
    )
    history = []
    for epoch in range(1, args.micro_epochs + 1):
        losses = []
        for images in loader:
            images = images.to(device, non_blocking=True)
            taps = teacher(images)['patches'][boundary:]
            loss = topk_loss(taps, student(ocbe(taps)), k=args.topk)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite suffix loss L{boundary} epoch={epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({'epoch': epoch, 'loss': float(np.mean(losses))})
    return ocbe.eval(), student.eval(), history


@torch.no_grad()
def cache_localization_split(paths, boundary, teacher, ocbe, student, cfg, args, device, seed):
    loader = make_loader(paths, cfg, teacher, args.eval_bs, args.workers)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    chunks = defaultdict(list)
    for images in loader:
        images = images.to(device, non_blocking=True)
        pseudo, pixel_mask = cutpaste_with_mask(
            images, args.pseudo_min_area, args.pseudo_max_area,
        )
        taps = teacher(pseudo)['patches'][boundary:]
        maps, valid = patch_discrepancy_maps(taps, student(ocbe(taps)), boundary)
        chunks['maps'].append(maps.cpu())
        chunks['valid'].append(valid.cpu())
        chunks['mask'].append(pixel_to_patch_mask(pixel_mask, maps.shape[-1]).cpu())
    return {key: torch.cat(value) for key, value in chunks.items()}


def binary_auc(labels, scores):
    labels = labels.detach().cpu().numpy().astype(np.bool_).reshape(-1)
    scores = scores.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if labels.shape != scores.shape:
        raise ValueError(f'label/score shape mismatch: {labels.shape} vs {scores.shape}')
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return float('nan')
    # Rank-sum AUC with tie handling; avoids a potentially huge positive x negative matrix.
    order = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    begin = 0
    while begin < len(scores):
        end = begin + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[begin]:
            end += 1
        ranks[order[begin:end]] = 0.5 * (begin + 1 + end)
        begin = end
    rank_sum = ranks[labels].sum()
    return float((rank_sum - len(positive) * (len(positive) + 1) / 2) /
                 (len(positive) * len(negative)))


@torch.no_grad()
def evaluate(localizer, caches, boundaries, device, batch_size):
    localizer.eval()
    metrics = {}
    for boundary in boundaries:
        cache = caches[str(boundary)]
        logits = []
        for start in range(0, len(cache['maps']), batch_size):
            logits.append(localizer(
                cache['maps'][start:start + batch_size].to(device),
                cache['valid'][start:start + batch_size].to(device),
            ).cpu())
        logits = torch.cat(logits)
        targets = cache['mask']
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        metrics[str(boundary)] = {
            'soft_bce': float(loss),
            'patch_auc': binary_auc(targets > 0, logits),
            'positive_patch_fraction': float((targets > 0).float().mean()),
            'mean_soft_overlap': float(targets.mean()),
            'positive_logit': float(logits[targets > 0].mean()),
            'negative_logit': float(logits[targets == 0].mean()),
        }
    return metrics


def train_localizer(support, query, boundaries, args, device):
    model = SharedLayerPatchLocalizer(
        num_layers=12, model_dim=args.model_dim, depth=args.localizer_depth,
        heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.localizer_lr,
                                  weight_decay=args.localizer_wd)
    rng = np.random.RandomState(args.seed + 9000)
    history, milestones = [], {}
    for epoch in range(1, args.localizer_epochs + 1):
        model.train(); losses = []; gradients = []
        for _ in range(args.steps_per_epoch):
            boundary = int(boundaries[int(rng.randint(len(boundaries)))])
            cache = support[str(boundary)]
            indices = torch.as_tensor(
                rng.randint(0, len(cache['maps']), size=args.localizer_bs), dtype=torch.long,
            )
            maps = cache['maps'].index_select(0, indices).to(device)
            valid = cache['valid'].index_select(0, indices).to(device)
            targets = cache['mask'].index_select(0, indices).to(device)
            logits = model(maps, valid)
            # The soft spatial mask is the complete target; no Normal-Pseudo score delta, margin,
            # layer threshold, or hand-written opening score enters this loss.
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite localizer loss epoch={epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(grad):
                raise FloatingPointError(f'non-finite localizer gradient epoch={epoch}')
            optimizer.step()
            losses.append(float(loss.detach())); gradients.append(float(grad.detach()))
        row = {'epoch': epoch, 'train_bce': float(np.mean(losses)),
               'grad_norm': float(np.mean(gradients))}
        if epoch in args.milestones:
            metrics = evaluate(model, query, boundaries, device, args.eval_bs)
            row['query'] = metrics; milestones[str(epoch)] = metrics
            choice = max(boundaries, key=lambda value: metrics[str(value)]['patch_auc'])
            print(f'  localizer ep{epoch:02d} train={row["train_bce"]:.4f} '
                  f'choice=L{choice} aucs=' + ','.join(
                      f'L{b}:{metrics[str(b)]["patch_auc"]:.4f}' for b in boundaries), flush=True)
        history.append(row)
    return model, history, milestones


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, choices=sorted(SCENES))
    parser.add_argument('--probe_dir', default='experiment_DLBM/run/probe/ckpt')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--boundaries', type=int, nargs='+', default=[0, 4, 8, 11])
    parser.add_argument('--micro_epochs', type=int, default=5)
    parser.add_argument('--localizer_epochs', type=int, default=20)
    parser.add_argument('--milestones', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--steps_per_epoch', type=int, default=64)
    parser.add_argument('--max_frames_per_video', type=int, default=48)
    parser.add_argument('--train_bs', type=int, default=8)
    parser.add_argument('--eval_bs', type=int, default=8)
    parser.add_argument('--localizer_bs', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--localizer_lr', type=float, default=1e-3)
    parser.add_argument('--localizer_wd', type=float, default=1e-2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--model_dim', type=int, default=16)
    parser.add_argument('--localizer_depth', type=int, default=1)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--pseudo_min_area', type=float, default=0.02)
    parser.add_argument('--pseudo_max_area', type=float, default=0.15)
    args = parser.parse_args()
    if max(args.milestones) > args.localizer_epochs:
        raise ValueError('milestones cannot exceed localizer epochs')
    if any(value < 0 or value > 11 for value in args.boundaries):
        raise ValueError('boundaries must lie in [0,11]')

    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    teacher, probe_ocbe, probe_student, cfg = load_probe(args.scene, args.probe_dir, device)
    if args.topk != int(cfg.get('k', args.topk)):
        raise ValueError(f'top-k mismatch: boundary={args.topk}, Probe={cfg.get("k")}')
    paths, videos = split_video_balanced(
        all_scene_paths(args.scene), args.max_frames_per_video, args.seed,
    )
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    print(f'[{args.scene}] support={len(paths["support"])} query={len(paths["query"])} '
          f'boundaries={args.boundaries} micro={args.micro_epochs} localizer={args.localizer_epochs}',
          flush=True)

    support, query, adaptation = {}, {}, {}
    for boundary in args.boundaries:
        ocbe, student, history = adapt_suffix(
            boundary, paths['support'], teacher, probe_ocbe, probe_student, cfg, args, device,
        )
        # Resetting each split seed makes the synthetic images/masks identical across boundaries.
        support[str(boundary)] = cache_localization_split(
            paths['support'], boundary, teacher, ocbe, student, cfg, args, device,
            args.seed + 10000,
        )
        query[str(boundary)] = cache_localization_split(
            paths['query'], boundary, teacher, ocbe, student, cfg, args, device,
            args.seed + 11000,
        )
        adaptation[str(boundary)] = history
        del ocbe, student
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'  cached L{boundary}', flush=True)

    model, history, milestones = train_localizer(
        support, query, args.boundaries, args, device,
    )
    final = evaluate(model, query, args.boundaries, device, args.eval_bs)
    choice = max(args.boundaries, key=lambda value: final[str(value)]['patch_auc'])
    output = {
        'format_version': 4, 'scene': args.scene,
        'method': 'coarse_micro_adapted_shared_patch_localization_target',
        'probe_checkpoint': str(probe_path(args.probe_dir, args.scene)), 'args': vars(args),
        'split': {'support_frames': len(paths['support']), 'query_frames': len(paths['query']),
                  'videos': videos},
        'adaptation': adaptation, 'localizer_history': history,
        'milestone_metrics': milestones, 'final_metrics': final,
        'best_boundary': choice, 'selected_taps': list(range(choice, 12)),
        'peak_gpu_gib': (torch.cuda.max_memory_allocated(device) / 2**30
                         if device.type == 'cuda' else 0),
    }
    torch.save({'localizer': model.state_dict(), 'args': vars(args), 'best_boundary': choice},
               out_dir / 'localizer.pth')
    write_json(out_dir / 'result.json', output)
    print(f'  final choice=L{choice} peak={output["peak_gpu_gib"]:.2f} GiB', flush=True)


if __name__ == '__main__':
    main()
