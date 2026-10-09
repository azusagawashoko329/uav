"""Train one scene-level DLBM with an exact one-step bilevel GDAS objective."""
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_DLBM.common import SCENES, probe_path, set_seed, write_json
from experiment_DLBM.dlbm_v005 import (ExactBoundarySearchSupernet, SceneBoundaryDistribution,
                                       area_rank_loss, boundary_policy)
from experiment_DLBM.run_boundary_v001 import (PathFrameDataset, all_scene_paths, load_probe,
                                                split_video_balanced)
from experiment_DLBM.train_boundary_v002 import balanced_indices


def cutpaste_with_area(batch, min_area, max_area):
    """Within-frame CutPaste and the realized normalized rectangle area for ranking supervision."""
    if batch.ndim != 4 or not 0 < min_area <= max_area < 1:
        raise ValueError('invalid CutPaste input or area range')
    output = batch.clone()
    areas = batch.new_empty(len(batch))
    _, _, height, width = batch.shape
    for index in range(len(batch)):
        requested = float(torch.empty((), device=batch.device).uniform_(min_area, max_area))
        aspect = math.exp(float(torch.empty((), device=batch.device).uniform_(math.log(0.5), math.log(2))))
        patch_h = max(1, min(height - 1, round(math.sqrt(requested * height * width / aspect))))
        patch_w = max(1, min(width - 1, round(math.sqrt(requested * height * width * aspect))))
        source_y = int(torch.randint(0, height - patch_h + 1, (), device=batch.device))
        source_x = int(torch.randint(0, width - patch_w + 1, (), device=batch.device))
        target_y = int(torch.randint(0, height - patch_h + 1, (), device=batch.device))
        target_x = int(torch.randint(0, width - patch_w + 1, (), device=batch.device))
        source = batch[index, :, source_y:source_y + patch_h, source_x:source_x + patch_w].clone()
        if bool(torch.rand((), device=batch.device) < 0.5):
            source = source.flip(-1)
        output[index, :, target_y:target_y + patch_h, target_x:target_x + patch_w] = source
        areas[index] = patch_h * patch_w / float(height * width)
    return output, areas


def _loader(paths, teacher, cfg, batch_size, workers):
    return DataLoader(
        PathFrameDataset(paths, cfg['img'], teacher.img_mean, teacher.img_std),
        batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )


@torch.no_grad()
def cache_support(paths, teacher, cfg, args, device):
    layers = defaultdict(list)
    for images in _loader(paths, teacher, cfg, args.cache_bs, args.workers):
        taps = teacher(images.to(device, non_blocking=True))['patches']
        for layer, value in enumerate(taps):
            layers[layer].append(value.to(dtype=torch.float16, device='cpu'))
    return {'taps': [torch.cat(layers[layer]) for layer in range(12)], 'paths': list(paths)}


@torch.no_grad()
def cache_query(paths, teacher, cfg, args, device):
    values = defaultdict(list)
    torch.manual_seed(args.seed + 12000)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed + 12000)
    for images in _loader(paths, teacher, cfg, args.cache_bs, args.workers):
        images = images.to(device, non_blocking=True)
        first, first_area = cutpaste_with_area(images, args.pseudo_min_area, args.pseudo_max_area)
        second, second_area = cutpaste_with_area(images, args.pseudo_min_area, args.pseudo_max_area)
        first_taps = teacher(first)['patches']; second_taps = teacher(second)['patches']
        for layer in range(12):
            values[f'first_{layer}'].append(first_taps[layer].to(dtype=torch.float16, device='cpu'))
            values[f'second_{layer}'].append(second_taps[layer].to(dtype=torch.float16, device='cpu'))
        values['first_area'].append(first_area.cpu()); values['second_area'].append(second_area.cpu())
    return {
        'first': [torch.cat(values[f'first_{layer}']) for layer in range(12)],
        'second': [torch.cat(values[f'second_{layer}']) for layer in range(12)],
        'first_area': torch.cat(values['first_area']),
        'second_area': torch.cat(values['second_area']),
        'paths': list(paths),
    }


def indexed_taps(taps, indices, device):
    index = torch.as_tensor(indices, dtype=torch.long)
    return [value.index_select(0, index).to(device=device, dtype=torch.float32,
                                           non_blocking=True) for value in taps]


def named_trainable_parameters(model):
    return {name: value for name, value in model.named_parameters() if value.requires_grad}


def one_step_parameters(model, support_taps, gate, inner_lr, create_graph=True):
    """One exact differentiable normal-RD update from the unchanged Probe parameters."""
    parameters = named_trainable_parameters(model)
    boundary_scores = model(support_taps)
    inner_loss = (boundary_scores.mean(dim=0) * gate).sum()
    gradients = torch.autograd.grad(
        inner_loss, tuple(parameters.values()), create_graph=create_graph, allow_unused=True,
    )
    updated = {
        name: value if gradient is None else value - inner_lr * gradient
        for (name, value), gradient in zip(parameters.items(), gradients)
    }
    return updated, inner_loss


def bilevel_step(model, dlbm, support_taps, first_taps, second_taps, first_area, second_area,
                 args):
    gate = dlbm.sample(args.temperature)
    updated, inner_loss = one_step_parameters(model, support_taps, gate, args.inner_lr, True)
    first_all = functional_call(model, updated, (first_taps,))
    second_all = functional_call(model, updated, (second_taps,))
    first_score = (first_all * gate).sum(dim=1)
    second_score = (second_all * gate).sum(dim=1)
    outer_loss = area_rank_loss(first_score, second_score, first_area, second_area)
    return inner_loss, outer_loss, gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, choices=sorted(SCENES))
    parser.add_argument('--probe_dir', default='experiment_DLBM/run/probe/ckpt')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--boundaries', type=int, nargs='+', default=[0, 4, 8, 11])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps_per_epoch', type=int, default=16)
    parser.add_argument('--support_bs', type=int, default=2)
    parser.add_argument('--query_bs', type=int, default=4)
    parser.add_argument('--cache_bs', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max_frames_per_video', type=int, default=48)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=5e-4)
    parser.add_argument('--dlbm_lr', type=float, default=5e-2)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--milestones', type=int, nargs='+', default=[5, 10, 20, 30, 50])
    parser.add_argument('--pseudo_min_area', type=float, default=0.02)
    parser.add_argument('--pseudo_max_area', type=float, default=0.15)
    args = parser.parse_args()
    if max(args.milestones) > args.epochs:
        raise ValueError('milestones cannot exceed epochs')
    if tuple(sorted(set(args.boundaries))) != tuple(args.boundaries):
        raise ValueError('boundaries must be sorted and unique')

    set_seed(args.seed)
    if torch.cuda.is_available():
        # Exact hyper-gradients require a scaled-dot-product attention backend with double backward.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    teacher, ocbe, student, cfg = load_probe(args.scene, args.probe_dir, device)
    if args.topk != int(cfg.get('k', args.topk)):
        raise ValueError(f'top-k mismatch: boundary={args.topk}, Probe={cfg.get("k")}')
    ocbe.requires_grad_(True); student.requires_grad_(True)
    model = ExactBoundarySearchSupernet(ocbe, student, args.boundaries, args.topk).to(device)
    model.eval()  # no BatchNorm/dropout noise; gradients and virtual updates remain enabled
    paths, videos = split_video_balanced(
        all_scene_paths(args.scene), args.max_frames_per_video, args.seed,
    )
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    print(f'[{args.scene}] caching support={len(paths["support"])} query={len(paths["query"])} '
          f'boundaries={args.boundaries}', flush=True)
    support = cache_support(paths['support'], teacher, cfg, args, device)
    query = cache_query(paths['query'], teacher, cfg, args, device)
    del teacher
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    dlbm = SceneBoundaryDistribution(args.boundaries).to(device)
    optimizer = torch.optim.Adam([dlbm.logits], lr=args.dlbm_lr)
    rng = np.random.RandomState(args.seed + 13000)
    history, milestones = [], {}
    sample_counts = {str(value): 0 for value in args.boundaries}
    for epoch in range(1, args.epochs + 1):
        inner_losses, outer_losses, grad_norms = [], [], []
        for _ in range(args.steps_per_epoch):
            support_indices = balanced_indices(
                support['paths'], args.support_bs, rng,
            )
            query_indices = balanced_indices(query['paths'], args.query_bs, rng)
            support_taps = indexed_taps(support['taps'], support_indices, device)
            first_taps = indexed_taps(query['first'], query_indices, device)
            second_taps = indexed_taps(query['second'], query_indices, device)
            index = torch.as_tensor(query_indices, dtype=torch.long)
            first_area = query['first_area'].index_select(0, index).to(device)
            second_area = query['second_area'].index_select(0, index).to(device)

            optimizer.zero_grad(set_to_none=True)
            for parameter in model.parameters():
                parameter.grad = None
            inner_loss, outer_loss, gate = bilevel_step(
                model, dlbm, support_taps, first_taps, second_taps,
                first_area, second_area, args,
            )
            if not torch.isfinite(inner_loss) or not torch.isfinite(outer_loss):
                raise FloatingPointError(f'non-finite bilevel loss epoch={epoch}')
            outer_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([dlbm.logits], args.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f'non-finite DLBM gradient epoch={epoch}')
            optimizer.step()
            sampled = args.boundaries[int(gate.argmax().item())]
            sample_counts[str(sampled)] += 1
            inner_losses.append(float(inner_loss.detach()))
            outer_losses.append(float(outer_loss.detach()))
            grad_norms.append(float(grad_norm.detach()))
            del support_taps, first_taps, second_taps, inner_loss, outer_loss, gate
            for parameter in model.parameters():
                parameter.grad = None

        policy = boundary_policy(dlbm.logits, args.boundaries)
        row = {
            'epoch': epoch, 'inner_rd': float(np.mean(inner_losses)),
            'outer_rank': float(np.mean(outer_losses)), 'grad_norm': float(np.mean(grad_norms)),
            'policy': policy,
        }
        history.append(row)
        if epoch in args.milestones:
            milestones[str(epoch)] = row
            torch.save({'dlbm': dlbm.state_dict(), 'epoch': epoch, 'args': vars(args),
                        'policy': policy}, out_dir / f'dlbm_ep{epoch:03d}.pth')
            print(f'  ep{epoch:02d}/{args.epochs} inner={row["inner_rd"]:.4f} '
                  f'outer={row["outer_rank"]:.4f} grad={row["grad_norm"]:.3e} '
                  f'boundary=L{policy["boundary"]} conf={policy["confidence"]:.3f} '
                  f'probs={policy["probabilities"]}', flush=True)

    final = boundary_policy(dlbm.logits, args.boundaries)
    output = {
        'format_version': 5, 'scene': args.scene,
        'method': 'scene_wise_gdas_one_step_bilevel_pseudo_area_rank',
        'probe_checkpoint': str(probe_path(args.probe_dir, args.scene)), 'args': vars(args),
        'split': {'support_frames': len(paths['support']), 'query_frames': len(paths['query']),
                  'videos': videos},
        'history': history, 'milestones': milestones, 'sample_counts': sample_counts,
        'final_policy': final,
        'peak_gpu_gib': (torch.cuda.max_memory_allocated(device) / 2**30
                         if device.type == 'cuda' else 0),
    }
    write_json(out_dir / 'result.json', output)
    print(f'  final boundary=L{final["boundary"]} confidence={final["confidence"]:.3f} '
          f'peak={output["peak_gpu_gib"]:.2f} GiB', flush=True)


if __name__ == '__main__':
    main()
