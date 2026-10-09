"""Stage 2: learn a shared DLBM from frozen all-layer normal probes.

The sole v1 objective is pairwise separation between normal reconstruction discrepancy and a
CutPaste counterfactual made exclusively from normal frames.  The policy is predicted from the
unaltered normal T/S maps; the counterfactual supplies a downstream ranking signal, not a DLBM input.
"""
import argparse
import math
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_DLBM.common import (ALL_SCENES, DATA_ROOT, SCENES, cross_frame_cutpaste,
                                    ensure_all_layer_probe, parse_scenes, policy_from_logits,
                                    probe_path, set_seed, write_json)
from model import (build_appearance_teacher, build_bottleneck, build_dlbm, build_student,
                   per_layer_topk_dist)
from train import FrameDataset


def load_probe(scene, probe_dir, teacher, device):
    checkpoint = torch.load(probe_path(probe_dir, scene), map_location='cpu')
    cfg = ensure_all_layer_probe(checkpoint, scene)
    ocbe = build_bottleneck(
        n_taps=12, code_dim=cfg['code_dim'], use_tap_norm=not cfg.get('no_tap_norm', False),
    ).to(device).eval()
    student = build_student(
        n_taps=12, code_dim=cfg['code_dim'], depth=cfg['depth'], tap_indices=list(range(12)),
        tap_align='absolute',
    ).to(device).eval()
    ocbe.load_state_dict(checkpoint['ocbe'])
    student.load_state_dict(checkpoint['student'])
    ocbe.requires_grad_(False)
    student.requires_grad_(False)
    scene_dir = DATA_ROOT / SCENES[scene]
    dataset = FrameDataset(str(scene_dir), cfg['img'], teacher.img_mean, teacher.img_std)
    return ocbe, student, dataset, cfg


@torch.no_grad()
def probe_features(teacher, ocbe, student, images):
    taps = teacher(images)['patches']
    reconstructions = student(ocbe(taps))
    return taps, reconstructions


def train_one_scene(model, optimizer, teacher, scene, args, device):
    ocbe, student, dataset, cfg = load_probe(scene, args.probe_dir, teacher, device)
    loader = DataLoader(
        dataset, batch_size=args.bs, shuffle=True, num_workers=args.workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )
    totals = {
        'loss': 0.0, 'gap': 0.0, 'expected': 0.0, 'entropy': 0.0,
        'confidence': 0.0, 'grad_norm': 0.0, 'steps': 0,
    }
    model.train()
    for step, images in enumerate(loader):
        if args.max_steps and step >= args.max_steps:
            break
        images = images.to(device, non_blocking=True)
        counterfactual = cross_frame_cutpaste(images, args.cutpaste_min_area, args.cutpaste_max_area)
        normal_taps, normal_recon = probe_features(teacher, ocbe, student, images)
        cf_taps, cf_recon = probe_features(teacher, ocbe, student, counterfactual)

        # DLBM sees the full normal feature maps. Scores are downstream reconstruction readouts,
        # reduced only after their per-layer values have been computed.
        output = model(normal_taps, normal_recon, reduce='scene')
        normal_layers = per_layer_topk_dist(normal_taps, normal_recon, k=args.k).mean(0, keepdim=True)
        cf_layers = per_layer_topk_dist(cf_taps, cf_recon, k=args.k).mean(0, keepdim=True)
        normal_score = model.soft_suffix_score(normal_layers, output.gates)
        cf_score = model.soft_suffix_score(cf_layers, output.gates)
        loss = F.softplus(normal_score - cf_score).mean()
        if not torch.isfinite(loss) or not torch.isfinite(output.gates).all():
            raise FloatingPointError(f'non-finite DLBM state for scene={scene} step={step}')

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_limit = args.grad_clip if args.grad_clip > 0 else float('inf')
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_limit)
        if not torch.isfinite(grad_norm) or float(grad_norm) == 0:
            raise FloatingPointError(f'invalid DLBM gradient norm for scene={scene} step={step}: {grad_norm}')
        optimizer.step()

        entropy = -(output.probabilities * output.probabilities.clamp_min(1e-12).log()).sum(-1)
        entropy = entropy / math.log(model.num_layers)
        totals['loss'] += float(loss.detach())
        totals['gap'] += float((cf_score - normal_score).mean().detach())
        totals['expected'] += float(output.expected_boundary.mean().detach())
        totals['entropy'] += float(entropy.mean().detach())
        totals['confidence'] += float(output.probabilities.max(dim=-1).values.mean().detach())
        totals['grad_norm'] += float(grad_norm.detach())
        totals['steps'] += 1
    del ocbe, student
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    n = max(totals.pop('steps'), 1)
    return {key: value / n for key, value in totals.items()}, len(dataset), cfg


@torch.no_grad()
def infer_scene_policy(model, teacher, scene, args, device):
    ocbe, student, dataset, _ = load_probe(scene, args.probe_dir, teacher, device)
    loader = DataLoader(
        dataset, batch_size=args.bs, shuffle=False, num_workers=args.workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )
    model.eval()
    logit_sum = torch.zeros(model.num_layers, device=device)
    sample_count = 0
    histogram = Counter()
    for step, images in enumerate(loader):
        if args.infer_steps and step >= args.infer_steps:
            break
        images = images.to(device, non_blocking=True)
        taps, recon = probe_features(teacher, ocbe, student, images)
        output = model(taps, recon)
        logit_sum += output.logits.sum(dim=0)
        sample_count += len(images)
        histogram.update(int(v) for v in output.hard_boundary.cpu().tolist())
    if sample_count == 0:
        raise RuntimeError(f'no frames used to infer {scene} policy')
    policy = policy_from_logits(logit_sum / sample_count, temperature=model.temperature)
    policy['num_probe_frames'] = sample_count
    policy['sample_boundary_histogram'] = {str(i): histogram.get(i, 0) for i in range(model.num_layers)}
    del ocbe, student
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe_dir', required=True, help='directory containing <scene>.pth probes')
    parser.add_argument('--out', required=True, help='DLBM checkpoint path')
    parser.add_argument('--policies', required=True, help='output JSON with one final boundary per scene')
    parser.add_argument('--scenes', default=','.join(ALL_SCENES))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=0, help='per scene and epoch; 0 uses all batches')
    parser.add_argument('--infer_steps', type=int, default=100, help='per scene; 0 uses all batches')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--model_dim', type=int, default=96)
    parser.add_argument('--model_depth', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--cutpaste_min_area', type=float, default=0.02)
    parser.add_argument('--cutpaste_max_area', type=float, default=0.15)
    args = parser.parse_args()

    set_seed(args.seed)
    scenes = parse_scenes(args.scenes)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    teacher = build_appearance_teacher(layers=list(range(12))).to(device).eval()
    model = build_dlbm(
        num_layers=12, feature_dim=teacher.embed_dim, model_dim=args.model_dim,
        depth=args.model_depth, heads=args.heads, dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f'DLBM | scenes={scenes} device={device} params='
          f'{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M objective=normal-vs-CutPaste-ranking',
          flush=True)

    history = []
    for epoch in range(args.epochs):
        order = scenes.copy()
        random.shuffle(order)
        for scene in order:
            stats, num_frames, _ = train_one_scene(model, optimizer, teacher, scene, args, device)
            row = {'epoch': epoch + 1, 'scene': scene, 'num_frames': num_frames, **stats}
            history.append(row)
            print(f"  ep{epoch + 1:02d}/{args.epochs} {scene:10s} loss={stats['loss']:.4f} "
                  f"gap={stats['gap']:.4f} E[k]={stats['expected']:.2f} "
                  f"H={stats['entropy']:.3f} conf={stats['confidence']:.3f} "
                  f"grad={stats['grad_norm']:.3e}",
                  flush=True)

    policies = {}
    for scene in scenes:
        policies[scene] = infer_scene_policy(model, teacher, scene, args, device)
        policy = policies[scene]
        print(f"  policy {scene:10s}: boundary=L{policy['boundary']} "
              f"confidence={policy['confidence']:.3f} entropy={policy['normalized_entropy']:.3f} "
              f"frames={policy['num_probe_frames']}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'dlbm': model.state_dict(),
        'model_cfg': {
            'num_layers': 12, 'feature_dim': teacher.embed_dim, 'model_dim': args.model_dim,
            'depth': args.model_depth, 'heads': args.heads, 'dropout': args.dropout,
            'temperature': args.temperature,
        },
        'train_cfg': vars(args),
        'objective': 'softplus(normal_soft_suffix_score - cutpaste_soft_suffix_score)',
        'policies': policies,
        'history': history,
    }, out_path)
    write_json(args.policies, {
        'format_version': 1,
        'semantics': 'boundary k selects the contiguous suffix [k, ..., 11]',
        'objective': 'normal-vs-normal-CutPaste pairwise reconstruction ranking',
        'scenes': policies,
    })
    print(f'saved {out_path} and {args.policies}', flush=True)
    if device.type == 'cuda':
        print(f'peak_gpu_memory={torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB', flush=True)


if __name__ == '__main__':
    main()
