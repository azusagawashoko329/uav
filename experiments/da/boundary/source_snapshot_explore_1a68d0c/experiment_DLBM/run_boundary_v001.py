"""Per-scene DLBM v001 counterfactual-gain probe.

The command is intentionally a Stage-2 diagnostic, not final training.  It reads one frozen
10-epoch all-layer Probe, trains a small top-k sequence evaluator on train-only normal/pseudo views,
and estimates signed adjacent opening gains from normal-only distillation gradients.
"""
import argparse
import copy
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_DLBM.common import (DATA_ROOT, SCENES, ensure_all_layer_probe, probe_path,
                                    set_seed, write_json)
from experiment_DLBM.dlbm_v001 import (TopKSequenceEvaluator, boundary_from_adjacent_gains,
                                       detached_gradients, first_order_adjacent_gains,
                                       suffix_gradient_alignments)
from model import build_appearance_teacher, build_bottleneck, build_student, per_layer_topk_dist


class PathFrameDataset(Dataset):
    def __init__(self, paths, image_size, mean, std):
        self.paths = list(paths)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(Image.open(self.paths[index]).convert('RGB'))


def within_frame_cutpaste(batch, min_area=0.02, max_area=0.15):
    """Copy a crop within each individual frame; never crosses batch items or sub-videos."""
    if batch.ndim != 4:
        raise ValueError(f'expected BCHW batch, got {tuple(batch.shape)}')
    if not 0 < min_area <= max_area < 1:
        raise ValueError('expected 0 < min_area <= max_area < 1')
    output = batch.clone()
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
    return output


def all_scene_paths(scene):
    pattern = DATA_ROOT / SCENES[scene] / 'sequence*' / 'train' / '*' / '*.jpg'
    paths = sorted(glob.glob(str(pattern)))
    if not paths:
        raise FileNotFoundError(pattern)
    return paths


def split_video_balanced(paths, max_frames_per_video, seed):
    by_video = defaultdict(list)
    for path in paths:
        by_video[str(Path(path).parent)].append(path)
    videos = sorted(by_video)
    if len(videos) < 2:
        raise ValueError('at least two train sub-videos are required for held-out gain probing')
    rng = np.random.RandomState(seed)
    rng.shuffle(videos)
    query_count = max(1, round(0.3 * len(videos)))
    query_videos = set(videos[:query_count])
    split = {'support': [], 'query': []}
    split_videos = {'support': [], 'query': []}
    for video in videos:
        frames = by_video[video]
        if max_frames_per_video and len(frames) > max_frames_per_video:
            indices = sorted(rng.choice(len(frames), max_frames_per_video, replace=False).tolist())
            frames = [frames[i] for i in indices]
        name = 'query' if video in query_videos else 'support'
        split[name].extend(frames)
        split_videos[name].append(video)
    return split, split_videos


def load_probe(scene, probe_dir, device):
    checkpoint = torch.load(probe_path(probe_dir, scene), map_location='cpu')
    cfg = ensure_all_layer_probe(checkpoint, scene)
    teacher = build_appearance_teacher(layers=list(range(12))).to(device).eval()
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
    return teacher, ocbe, student, cfg


def fast_parameters(ocbe, student):
    """Shared bottleneck/input path plus layer-specific reconstruction heads."""
    named = []
    named.extend((f'ocbe.{name}', value) for name, value in ocbe.named_parameters())
    named.extend((f'student.proj_in.{name}', value) for name, value in student.proj_in.named_parameters())
    named.extend((f'student.heads.{name}', value) for name, value in student.heads.named_parameters())
    for _, value in named:
        value.requires_grad_(True)
    return named


def probe_scores(teacher, ocbe, student, images, topk):
    taps = teacher(images)['patches']
    reconstructions = student(ocbe(taps))
    return per_layer_topk_dist(taps, reconstructions, k=topk)


@torch.no_grad()
def cache_scores(paths, teacher, ocbe, student, cfg, args, device):
    loader = DataLoader(
        PathFrameDataset(paths, cfg['img'], teacher.img_mean, teacher.img_std),
        batch_size=args.cache_bs, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == 'cuda',
    )
    normal, pseudo = [], []
    for images in loader:
        images = images.to(device, non_blocking=True)
        normal.append(probe_scores(teacher, ocbe, student, images, args.topk).cpu())
        pseudo_images = within_frame_cutpaste(images, args.pseudo_min_area, args.pseudo_max_area)
        pseudo.append(probe_scores(teacher, ocbe, student, pseudo_images, args.topk).cpu())
    return torch.cat(normal), torch.cat(pseudo)


def binary_auc(labels, scores):
    labels = labels.detach().cpu().numpy().astype(np.int64)
    scores = scores.detach().cpu().numpy().astype(np.float64)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float('nan')
    # Exact pairwise AUC; the caches are deliberately small.
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean()))


def evaluator_metrics(evaluator, normal, pseudo, device):
    evaluator.eval()
    with torch.no_grad():
        normal_logits = evaluator(normal.to(device))
        pseudo_logits = evaluator(pseudo.to(device))
        logits = torch.cat((normal_logits, pseudo_logits))
        labels = torch.cat((torch.zeros_like(normal_logits), torch.ones_like(pseudo_logits)))
        loss = F.binary_cross_entropy_with_logits(logits, labels)
    return {'loss': float(loss), 'auc': binary_auc(labels, logits),
            'normal_logit': float(normal_logits.mean()), 'pseudo_logit': float(pseudo_logits.mean())}


def train_evaluator(support_normal, support_pseudo, query_normal, query_pseudo, args, device):
    evaluator = TopKSequenceEvaluator(
        num_layers=12, model_dim=args.evaluator_dim, depth=args.evaluator_depth,
        heads=args.evaluator_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(evaluator.parameters(), lr=args.evaluator_lr,
                                  weight_decay=args.evaluator_wd)
    features = torch.cat((support_normal, support_pseudo))
    labels = torch.cat((torch.zeros(len(support_normal)), torch.ones(len(support_pseudo))))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(torch.utils.data.TensorDataset(features, labels), batch_size=args.evaluator_bs,
                        shuffle=True, generator=generator, drop_last=False)
    history = []
    for epoch in range(args.evaluator_epochs):
        evaluator.train()
        total, count = 0.0, 0
        for values, targets in loader:
            values = values.to(device); targets = targets.to(device)
            loss = F.binary_cross_entropy_with_logits(evaluator(values), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite evaluator loss at epoch {epoch + 1}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(values); count += len(values)
        if epoch in {0, args.evaluator_epochs - 1}:
            history.append({'epoch': epoch + 1, 'support_loss': total / count,
                            'query': evaluator_metrics(evaluator, query_normal, query_pseudo, device)})
    evaluator.eval().requires_grad_(False)
    return evaluator, history


def load_images(paths, teacher, cfg, device):
    dataset = PathFrameDataset(paths, cfg['img'], teacher.img_mean, teacher.img_std)
    return torch.stack([dataset[i] for i in range(len(dataset))]).to(device)


def evaluator_risk(teacher, ocbe, student, evaluator, normal_images, pseudo_images, args):
    normal_scores = probe_scores(teacher, ocbe, student, normal_images, args.topk)
    pseudo_scores = probe_scores(teacher, ocbe, student, pseudo_images, args.topk)
    normal_logits = evaluator(normal_scores)
    pseudo_logits = evaluator(pseudo_scores)
    return (F.binary_cross_entropy_with_logits(normal_logits, torch.zeros_like(normal_logits))
            + F.binary_cross_entropy_with_logits(pseudo_logits, torch.ones_like(pseudo_logits)))


def query_risk_and_gradient(teacher, ocbe, student, evaluator, normal_images, pseudo_images,
                            args, parameters):
    risk = evaluator_risk(
        teacher, ocbe, student, evaluator, normal_images, pseudo_images, args,
    )
    gradients = detached_gradients(risk, parameters)
    return float(risk.detach()), gradients


def support_alignments(teacher, ocbe, student, normal_images, args, parameters, risk_gradients):
    scores = probe_scores(teacher, ocbe, student, normal_images, args.topk)
    layer_losses = scores.mean(dim=0)
    alignments = suffix_gradient_alignments(layer_losses, parameters, risk_gradients)
    return layer_losses.detach(), alignments


def sample_paths(paths, count, rng):
    replace = len(paths) < count
    indices = rng.choice(len(paths), count, replace=replace)
    return [paths[int(i)] for i in indices]


def exact_one_step_risks(teacher, ocbe, student, evaluator, support_images, query_images,
                         query_pseudo_images, args, parameters, boundaries):
    """Apply one real SGD step per requested suffix, evaluate, and restore the Probe exactly."""
    boundaries = sorted(set(int(value) for value in boundaries))
    if not boundaries or boundaries[0] < 0 or boundaries[-1] > 11:
        raise ValueError(f'invalid exact boundaries: {boundaries}')
    originals = [parameter.detach().clone() for parameter in parameters]
    risks = {}
    try:
        for boundary in boundaries:
            # Every boundary starts from precisely the same frozen Probe weights.
            with torch.no_grad():
                for parameter, original in zip(parameters, originals):
                    parameter.copy_(original)
            support_scores = probe_scores(teacher, ocbe, student, support_images, args.topk)
            suffix_loss = support_scores[:, boundary:].mean()
            gradients = torch.autograd.grad(
                suffix_loss, parameters, allow_unused=True, create_graph=False,
            )
            with torch.no_grad():
                for parameter, gradient in zip(parameters, gradients):
                    if gradient is not None:
                        parameter.add_(gradient, alpha=-args.inner_lr)
                risk = evaluator_risk(
                    teacher, ocbe, student, evaluator, query_images, query_pseudo_images, args,
                )
            risks[boundary] = float(risk)
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals):
                parameter.copy_(original)
    return risks


def pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def optional_metric_text(value):
    return 'n/a' if value is None else f'{value:.3f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, choices=sorted(SCENES))
    parser.add_argument('--probe_dir', default='experiment_DLBM/run/probe/ckpt')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--max_frames_per_video', type=int, default=48)
    parser.add_argument('--cache_bs', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pseudo_min_area', type=float, default=0.02)
    parser.add_argument('--pseudo_max_area', type=float, default=0.15)
    parser.add_argument('--evaluator_dim', type=int, default=32)
    parser.add_argument('--evaluator_depth', type=int, default=1)
    parser.add_argument('--evaluator_heads', type=int, default=4)
    parser.add_argument('--evaluator_epochs', type=int, default=40)
    parser.add_argument('--evaluator_bs', type=int, default=64)
    parser.add_argument('--evaluator_lr', type=float, default=1e-3)
    parser.add_argument('--evaluator_wd', type=float, default=1e-2)
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--support_bs', type=int, default=3)
    parser.add_argument('--query_bs', type=int, default=3)
    parser.add_argument('--inner_lr', type=float, default=5e-4)
    parser.add_argument('--exact_candidates', type=int, nargs='*', default=[0, 4, 8, 10],
                        help='adjacent openings k to validate with a real one-step update; empty disables')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher, ocbe, student, cfg = load_probe(args.scene, args.probe_dir, device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    if args.topk != int(cfg.get('k', args.topk)):
        raise ValueError(f"top-k must match Probe: requested {args.topk}, Probe used {cfg.get('k')}")
    paths, videos = split_video_balanced(
        all_scene_paths(args.scene), args.max_frames_per_video, args.seed,
    )
    print(f"[{args.scene}] support={len(paths['support'])}/{len(videos['support'])} videos "
          f"query={len(paths['query'])}/{len(videos['query'])} videos device={device}", flush=True)

    support_normal, support_pseudo = cache_scores(
        paths['support'], teacher, ocbe, student, cfg, args, device)
    query_normal, query_pseudo = cache_scores(
        paths['query'], teacher, ocbe, student, cfg, args, device)
    evaluator, evaluator_history = train_evaluator(
        support_normal, support_pseudo, query_normal, query_pseudo, args, device)
    final_metrics = evaluator_metrics(evaluator, query_normal, query_pseudo, device)
    print(f"  evaluator query loss={final_metrics['loss']:.4f} auc={final_metrics['auc']:.4f} "
          f"normal={final_metrics['normal_logit']:.3f} pseudo={final_metrics['pseudo_logit']:.3f}",
          flush=True)

    named_fast = fast_parameters(ocbe, student)
    parameters = tuple(value for _, value in named_fast)
    fast_count = sum(value.numel() for value in parameters)
    rng = np.random.RandomState(args.seed + 1000)
    episode_rows = []
    for episode in range(args.episodes):
        support_images = load_images(
            sample_paths(paths['support'], args.support_bs, rng), teacher, cfg, device)
        query_images = load_images(
            sample_paths(paths['query'], args.query_bs, rng), teacher, cfg, device)
        # Re-seed pseudo creation per episode while keeping it reproducible.
        torch.manual_seed(args.seed + 2000 + episode)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed + 2000 + episode)
        query_pseudo_images = within_frame_cutpaste(
            query_images, args.pseudo_min_area, args.pseudo_max_area)
        risk, risk_gradients = query_risk_and_gradient(
            teacher, ocbe, student, evaluator, query_images, query_pseudo_images, args, parameters)
        layer_losses, alignments = support_alignments(
            teacher, ocbe, student, support_images, args, parameters, risk_gradients)
        gains = first_order_adjacent_gains(alignments, args.inner_lr)
        if not torch.isfinite(gains).all():
            raise FloatingPointError(f'non-finite gain at episode {episode}')
        decision = boundary_from_adjacent_gains(gains)
        episode_rows.append({
            'episode': episode,
            'query_risk': risk,
            'layer_losses': [float(value) for value in layer_losses.cpu()],
            'alignments': [float(value) for value in alignments.cpu()],
            'adjacent_gains': [float(value) for value in gains.cpu()],
            'boundary': decision.boundary,
            'cumulative_benefits': [float(value) for value in decision.cumulative_benefits.cpu()],
        })
        if episode == 0 and args.exact_candidates:
            invalid = [value for value in args.exact_candidates if value < 0 or value > 10]
            if invalid:
                raise ValueError(f'exact candidates must lie in [0,10], got {invalid}')
            exact_boundaries = set(args.exact_candidates)
            exact_boundaries.update(value + 1 for value in args.exact_candidates)
            exact_risks = exact_one_step_risks(
                teacher, ocbe, student, evaluator, support_images, query_images,
                query_pseudo_images, args, parameters, exact_boundaries,
            )
            exact_gains = [exact_risks[k + 1] - exact_risks[k] for k in args.exact_candidates]
            approximate_gains = [float(gains[k]) for k in args.exact_candidates]
            nonzero = [(a, b) for a, b in zip(approximate_gains, exact_gains)
                       if a != 0.0 and b != 0.0]
            sign_agreement = (sum(np.sign(a) == np.sign(b) for a, b in nonzero) / len(nonzero)
                              if nonzero else None)
            episode_rows[-1]['exact_validation'] = {
                'candidates': list(args.exact_candidates),
                'risks': {str(key): value for key, value in exact_risks.items()},
                'first_order_gains': approximate_gains,
                'exact_gains': exact_gains,
                'pearson': pearson(approximate_gains, exact_gains),
                'sign_agreement': sign_agreement,
            }
            print(f'  exact sign={optional_metric_text(sign_agreement)} '
                  f"pearson={optional_metric_text(episode_rows[-1]['exact_validation']['pearson'])}",
                  flush=True)
        print(f'  episode={episode} boundary=L{decision.boundary} '
              f'gain_range=[{float(gains.min()):.3e},{float(gains.max()):.3e}]', flush=True)
        del support_images, query_images, query_pseudo_images, risk_gradients
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    gain_matrix = torch.tensor([row['adjacent_gains'] for row in episode_rows])
    mean_gains = gain_matrix.mean(dim=0)
    std_gains = gain_matrix.std(dim=0, unbiased=False)
    final_decision = boundary_from_adjacent_gains(mean_gains)
    result = {
        'format_version': 1,
        'scene': args.scene,
        'method': 'DLBM_v001_topk_counterfactual_first_order',
        'semantics': 'gain[k] = held-out risk([k+1..11]) - held-out risk([k..11])',
        'probe_checkpoint': str(probe_path(args.probe_dir, args.scene)),
        'probe_cfg': cfg,
        'args': vars(args),
        'split': {key: {'frames': len(paths[key]), 'videos': videos[key]} for key in paths},
        'fast_parameters': {'scope': 'ocbe+student.proj_in+student.heads', 'count': fast_count,
                            'names': [name for name, _ in named_fast]},
        'evaluator_history': evaluator_history,
        'evaluator_query': final_metrics,
        'episodes': episode_rows,
        'mean_adjacent_gains': [float(value) for value in mean_gains],
        'std_adjacent_gains': [float(value) for value in std_gains],
        'boundary': final_decision.boundary,
        'selected_taps': list(range(final_decision.boundary, 12)),
        'cumulative_benefits': [float(value) for value in final_decision.cumulative_benefits],
    }
    torch.save({'evaluator': evaluator.state_dict(), 'evaluator_cfg': {
        'num_layers': 12, 'model_dim': args.evaluator_dim, 'depth': args.evaluator_depth,
        'heads': args.evaluator_heads,
    }}, out_dir / 'evaluator.pth')
    write_json(out_dir / 'result.json', result)
    print(f"  final boundary=L{final_decision.boundary} taps={result['selected_taps']}", flush=True)
    if device.type == 'cuda':
        print(f'  peak_gpu_memory={torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB', flush=True)
    print(f'saved {out_dir}', flush=True)


if __name__ == '__main__':
    main()
