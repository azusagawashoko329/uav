"""Coarse exact-suffix micro-adaptation probe before committing to a v003 DLBM target."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_DLBM.common import SCENES, probe_path, set_seed, write_json
from experiment_DLBM.run_boundary_v001 import (PathFrameDataset, all_scene_paths, load_probe,
                                                split_video_balanced, within_frame_cutpaste)
from model import (build_fixed_suffix_ocbe, build_fixed_suffix_student, per_layer_topk_dist,
                   topk_loss)


def fixed_scores(teacher, ocbe, student, images, boundary, topk):
    taps = teacher(images)['patches'][boundary:]
    reconstructions = student(ocbe(taps))
    layers = per_layer_topk_dist(taps, reconstructions, k=topk)
    return layers.mean(dim=1), layers


@torch.no_grad()
def evaluate(paths, teacher, ocbe, student, cfg, boundary, args, device):
    loader = DataLoader(
        PathFrameDataset(paths, cfg['img'], teacher.img_mean, teacher.img_std),
        batch_size=args.eval_bs, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == 'cuda', drop_last=False,
    )
    normal_scores, pseudo_scores, normal_layers = [], [], []
    torch.manual_seed(args.seed + 7000)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed + 7000)
    for images in loader:
        images = images.to(device, non_blocking=True)
        pseudo = within_frame_cutpaste(images, args.pseudo_min_area, args.pseudo_max_area)
        score_n, layers_n = fixed_scores(teacher, ocbe, student, images, boundary, args.topk)
        score_p, _ = fixed_scores(teacher, ocbe, student, pseudo, boundary, args.topk)
        normal_scores.append(score_n.cpu())
        pseudo_scores.append(score_p.cpu())
        normal_layers.append(layers_n.cpu())
    normal_scores = torch.cat(normal_scores)
    pseudo_scores = torch.cat(pseudo_scores)
    return {
        'pairwise_risk': float(F.softplus(normal_scores - pseudo_scores).mean()),
        'normal_score': float(normal_scores.mean()),
        'pseudo_score': float(pseudo_scores.mean()),
        'normal_layer_scores': [float(value) for value in torch.cat(normal_layers).mean(0)],
    }


def train_boundary(boundary, support_paths, query_paths, teacher, probe_ocbe, probe_student,
                   cfg, args, device):
    ocbe = build_fixed_suffix_ocbe(probe_ocbe, boundary).to(device).train()
    student = build_fixed_suffix_student(probe_student, boundary).to(device).train()
    generator = torch.Generator().manual_seed(args.seed + boundary)
    loader = DataLoader(
        PathFrameDataset(support_paths, cfg['img'], teacher.img_mean, teacher.img_std),
        batch_size=args.train_bs, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == 'cuda', drop_last=True, generator=generator,
    )
    optimizer = torch.optim.AdamW(
        list(ocbe.parameters()) + list(student.parameters()), lr=args.lr, weight_decay=args.wd,
    )
    milestones = {0, *args.milestones}
    records = {'0': evaluate(query_paths, teacher, ocbe.eval(), student.eval(), cfg,
                             boundary, args, device)}
    train_history = []
    for epoch in range(1, args.epochs + 1):
        ocbe.train(); student.train()
        losses = []
        for images in loader:
            images = images.to(device, non_blocking=True)
            taps = teacher(images)['patches'][boundary:]
            loss = topk_loss(taps, student(ocbe(taps)), k=args.topk)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite micro loss boundary={boundary} epoch={epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        train_history.append({'epoch': epoch, 'loss': float(np.mean(losses))})
        if epoch in milestones:
            records[str(epoch)] = evaluate(
                query_paths, teacher, ocbe.eval(), student.eval(), cfg, boundary, args, device,
            )
    del ocbe, student, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {'boundary': boundary, 'selected_taps': list(range(boundary, 12)),
            'train_history': train_history, 'milestones': records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, choices=sorted(SCENES))
    parser.add_argument('--probe_dir', default='experiment_DLBM/run/probe/ckpt')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--boundaries', type=int, nargs='+', default=[0, 4, 8, 11])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--milestones', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--max_frames_per_video', type=int, default=48)
    parser.add_argument('--train_bs', type=int, default=8)
    parser.add_argument('--eval_bs', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--pseudo_min_area', type=float, default=0.02)
    parser.add_argument('--pseudo_max_area', type=float, default=0.15)
    args = parser.parse_args()
    if max(args.milestones) > args.epochs:
        raise ValueError('milestones cannot exceed epochs')
    if any(value < 0 or value > 11 for value in args.boundaries):
        raise ValueError('boundaries must lie in [0,11]')

    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    teacher, probe_ocbe, probe_student, cfg = load_probe(args.scene, args.probe_dir, device)
    paths, videos = split_video_balanced(
        all_scene_paths(args.scene), args.max_frames_per_video, args.seed,
    )
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    print(f"[{args.scene}] support={len(paths['support'])} query={len(paths['query'])} "
          f"boundaries={args.boundaries} micro_epochs={args.epochs}", flush=True)
    results = {}
    for boundary in args.boundaries:
        result = train_boundary(
            boundary, paths['support'], paths['query'], teacher, probe_ocbe, probe_student,
            cfg, args, device,
        )
        results[str(boundary)] = result
        risks = {epoch: value['pairwise_risk'] for epoch, value in result['milestones'].items()}
        print(f'  L{boundary} risks={risks}', flush=True)
    choices = {}
    for milestone in [0, *args.milestones]:
        key = str(milestone)
        choices[key] = min(args.boundaries,
                           key=lambda boundary: results[str(boundary)]['milestones'][key]['pairwise_risk'])
    output = {
        'format_version': 3, 'scene': args.scene, 'method': 'coarse_exact_suffix_micro_adaptation',
        'probe_checkpoint': str(probe_path(args.probe_dir, args.scene)), 'args': vars(args),
        'split': {'support_frames': len(paths['support']), 'query_frames': len(paths['query']),
                  'videos': videos},
        'boundaries': results, 'best_boundary_by_milestone': choices,
        'peak_gpu_gib': (torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0),
    }
    write_json(out_dir / 'result.json', output)
    print(f'  choices={choices} peak={output["peak_gpu_gib"]:.2f} GiB', flush=True)


if __name__ == '__main__':
    main()
