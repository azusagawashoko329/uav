"""Train one scene-wise DLBM v006 from paired patch relations and rollout gain.

The permanent Probe10 checkpoint is read-only.  Short-lived fixed-suffix copies
produce difference-in-differences opening targets and are discarded before the
small relation network is trained.
"""
import argparse
import gc
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

from experiment_DLBM.common import (SCENES, multi_mode_pseudo_stress, probe_path,
                                    set_seed, write_json)
from experiment_DLBM.dlbm_v001 import boundary_from_adjacent_gains
from experiment_DLBM.dlbm_v006 import (PairedIncrementalRelationNetwork,
                                       policy_from_adjacent_logits)
from experiment_DLBM.run_boundary_v001 import (PathFrameDataset, all_scene_paths, load_probe,
                                                split_video_balanced, within_frame_cutpaste)
from model.boundary_supernet import build_fixed_suffix_ocbe, build_fixed_suffix_student
from model.loss import per_layer_topk_dist, topk_loss


def _loader(paths, teacher, cfg, batch_size, workers):
    return DataLoader(
        PathFrameDataset(paths, cfg['img'], teacher.img_mean, teacher.img_std),
        batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )


def discrepancy_maps(teacher_taps, reconstructions):
    if len(teacher_taps) != len(reconstructions):
        raise ValueError('teacher/reconstruction layer mismatch')
    return torch.stack([
        1.0 - F.cosine_similarity(teacher, reconstruction, dim=-1)
        for teacher, reconstruction in zip(teacher_taps, reconstructions)
    ], dim=1)


@torch.no_grad()
def cache_relation_maps(paths, teacher, ocbe, student, cfg, args, device):
    values = defaultdict(list)
    torch.manual_seed(args.seed + 16000)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed + 16000)
    for images in _loader(paths, teacher, cfg, args.cache_bs, args.workers):
        images = images.to(device, non_blocking=True)
        pseudo_mode = getattr(args, 'pseudo_mode', 'cutpaste')
        if pseudo_mode == 'cutpaste':
            if getattr(args, 'cache_pseudo_mask', False):
                raise ValueError('pseudo-mask caching requires pseudo_mode=multi')
            pseudo = within_frame_cutpaste(
                images, args.pseudo_min_area, args.pseudo_max_area,
            )
        elif pseudo_mode == 'multi':
            if getattr(args, 'cache_pseudo_mask', False):
                pseudo, pseudo_mask = multi_mode_pseudo_stress(
                    images, return_mask=True,
                )
            else:
                pseudo = multi_mode_pseudo_stress(images)
        else:
            raise ValueError(f'unknown pseudo_mode={pseudo_mode}')
        for name, view in (('normal', images), ('pseudo', pseudo)):
            taps = teacher(view)['patches']
            reconstruction = student(ocbe(taps))
            values[name].append(
                discrepancy_maps(taps, reconstruction).to(dtype=torch.float16, device='cpu'),
            )
            if name == 'pseudo' and getattr(args, 'cache_pseudo_mask', False):
                side = int(values[name][-1].shape[-1] ** 0.5)
                if side * side != values[name][-1].shape[-1]:
                    raise ValueError('patch count must form a square grid')
                patch_mask = F.interpolate(
                    pseudo_mask.unsqueeze(1), size=(side, side),
                    mode='bilinear', align_corners=False,
                ).squeeze(1)
                values['pseudo_mask'].append(
                    patch_mask.to(dtype=torch.float16, device='cpu'),
                )
    result = {
        'normal': torch.cat(values['normal']),
        'pseudo': torch.cat(values['pseudo']),
        'paths': list(paths),
    }
    if getattr(args, 'cache_pseudo_mask', False):
        result['pseudo_mask'] = torch.cat(values['pseudo_mask'])
    return result


def sample_paths(paths, count, rng):
    indices = rng.choice(len(paths), count, replace=len(paths) < count)
    return [paths[int(index)] for index in indices]


@torch.no_grad()
def teacher_taps_for_paths(paths, teacher, cfg, device):
    dataset = PathFrameDataset(paths, cfg['img'], teacher.img_mean, teacher.img_std)
    images = torch.stack([dataset[index] for index in range(len(dataset))]).to(device)
    return teacher(images)['patches']


@torch.no_grad()
def query_layer_losses(ocbe, student, taps, boundary, topk):
    reconstructions = student(ocbe(taps[boundary:]))
    return per_layer_topk_dist(taps[boundary:], reconstructions, k=topk).mean(dim=0)


def one_boundary_learning_progress(all_ocbe, all_student, support_taps, query_taps,
                                   boundary, schedule, args, device):
    """Return per-layer hold-out improvement for one ephemeral suffix learner."""
    ocbe = build_fixed_suffix_ocbe(all_ocbe, boundary).to(device).eval()
    student = build_fixed_suffix_student(all_student, boundary).to(device).eval()
    ocbe.requires_grad_(True)
    student.requires_grad_(True)
    parameters = list(ocbe.parameters()) + list(student.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.inner_lr, weight_decay=args.inner_wd)
    before = query_layer_losses(ocbe, student, query_taps, boundary, args.topk)
    for indices in schedule:
        index = torch.as_tensor(indices, dtype=torch.long, device=device)
        selected = [value.index_select(0, index) for value in support_taps[boundary:]]
        reconstruction = student(ocbe(selected))
        loss = topk_loss(selected, reconstruction, k=args.topk)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite rollout loss boundary={boundary}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    after = query_layer_losses(ocbe, student, query_taps, boundary, args.topk)
    improvement = (before - after).detach().cpu()
    del optimizer, parameters, ocbe, student, before, after
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return improvement


def rollout_episode(all_ocbe, all_student, support_taps, query_taps,
                    candidate_layers, args, rng, device):
    needed = sorted(set(candidate_layers) | {value + 1 for value in candidate_layers})
    schedule = [
        rng.choice(len(support_taps[0]), args.inner_bs,
                   replace=len(support_taps[0]) < args.inner_bs)
        for _ in range(args.inner_steps)
    ]
    improvements = {}
    for boundary in needed:
        improvements[boundary] = one_boundary_learning_progress(
            all_ocbe, all_student, support_taps, query_taps,
            boundary, schedule, args, device,
        )
    gains = adjacent_rollout_gains(improvements, candidate_layers)
    if not torch.isfinite(gains).all():
        raise FloatingPointError('non-finite rollout gain')
    return gains


def adjacent_rollout_gains(improvements, candidate_layers):
    """Difference-in-differences gain on the common already-selected anchor.

    ``improvements[b]`` stores pre-minus-post query loss for layers ``b..11``
    after training suffix ``b..11``. Opening layer ``b`` is credited only for
    additional progress on the common layers ``b+1..11``.
    """
    gains = []
    for boundary in candidate_layers:
        if boundary not in improvements or boundary + 1 not in improvements:
            raise ValueError(f'missing rollout pair for candidate layer {boundary}')
        opened = improvements[boundary]
        stopped = improvements[boundary + 1]
        if opened.ndim != 1 or stopped.ndim != 1 or len(opened) != len(stopped) + 1:
            raise ValueError(f'invalid adjacent improvement shapes at layer {boundary}')
        gains.append(opened[1:].mean() - stopped.mean())
    return torch.stack(gains)


def generate_rollout_targets(scene, paths, teacher, ocbe, student, cfg, candidate_layers,
                             args, device):
    rng = np.random.RandomState(args.seed + 17000)
    path_to_index = {path: index for index, path in enumerate(paths['support'])}
    episodes = []
    for episode in range(args.target_episodes):
        support_paths = sample_paths(paths['support'], args.episode_frames, rng)
        query_paths = sample_paths(paths['query'], args.query_frames, rng)
        support_taps = teacher_taps_for_paths(support_paths, teacher, cfg, device)
        query_taps = teacher_taps_for_paths(query_paths, teacher, cfg, device)
        gains = rollout_episode(
            ocbe, student, support_taps, query_taps, candidate_layers, args, rng, device,
        )
        input_indices = [path_to_index[path] for path in support_paths]
        full_gains = torch.zeros(11)
        valid = torch.zeros(11, dtype=torch.bool)
        for index, boundary in enumerate(candidate_layers):
            full_gains[boundary] = gains[index]
            valid[boundary] = True
        episodes.append({
            'episode': episode,
            'input_indices': input_indices,
            'gains': full_gains,
            'valid': valid,
        })
        partial = boundary_from_adjacent_gains(gains)
        print(
            f'  target episode={episode + 1}/{args.target_episodes} '
            f'candidate_gain={[float(value) for value in gains]} '
            f'local_boundary_index={partial.boundary}', flush=True,
        )
        del support_taps, query_taps
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return episodes


def episode_maps(cache, episode, device):
    index = torch.as_tensor(episode['input_indices'], dtype=torch.long)
    return tuple(
        cache[name].index_select(0, index).to(device=device, dtype=torch.float32,
                                             non_blocking=True)
        for name in ('normal', 'pseudo')
    )


def complete_vector(values, candidate_layers, fill=0.0):
    output = torch.full((11,), float(fill), dtype=values.dtype, device=values.device)
    output[torch.as_tensor(candidate_layers, device=values.device)] = values
    return output


def metrics_from_episodes(model, cache, episodes, candidate_layers, device,
                          destroy_pairing=False):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for episode in episodes:
            normal, pseudo = episode_maps(cache, episode, device)
            permutation = None
            if destroy_pairing:
                permutation = torch.arange(len(normal) - 1, -1, -1, device=device)
            logits = model(
                normal, pseudo, candidate_layers=candidate_layers,
                candidate_permutation=permutation,
            )
            predictions.append(logits.cpu())
            targets.append(episode['gains'][list(candidate_layers)])
    prediction = torch.stack(predictions)
    target = torch.stack(targets)
    loss = F.mse_loss(prediction, target)
    accuracy = ((prediction > 0) == (target > 0)).to(torch.float32).mean()
    mean_prediction = prediction.mean(dim=0)
    mean_target = target.mean(dim=0)
    full_prediction = complete_vector(mean_prediction, candidate_layers)
    full_target = complete_vector(mean_target, candidate_layers)
    predicted_policy = policy_from_adjacent_logits(full_prediction)
    oracle_policy = boundary_from_adjacent_gains(full_target)
    return {
        'mse': float(loss),
        'mae': float(F.l1_loss(prediction, target)),
        'sign_accuracy': float(accuracy),
        'positive_fraction': float((target > 0).to(torch.float32).mean()),
        'mean_predicted_gains': {str(b): float(mean_prediction[i])
                                 for i, b in enumerate(candidate_layers)},
        'mean_target_gains': {str(b): float(mean_target[i])
                              for i, b in enumerate(candidate_layers)},
        'boundary': predicted_policy.boundary,
        'oracle_boundary': oracle_policy.boundary,
        'cumulative_utilities': [float(value) for value in predicted_policy.cumulative_utilities],
        'target_cumulative_gains': [float(value) for value in oracle_policy.cumulative_benefits],
        '_predictions': prediction,
    }


def public_metrics(metrics):
    return {key: value for key, value in metrics.items() if not key.startswith('_')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, choices=sorted(SCENES))
    parser.add_argument('--probe_dir', default='experiment_DLBM/run/probe/ckpt')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--candidate_layers', type=int, nargs='+', default=list(range(11)))
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--max_frames_per_video', type=int, default=48)
    parser.add_argument('--cache_bs', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pseudo_min_area', type=float, default=0.02)
    parser.add_argument('--pseudo_max_area', type=float, default=0.15)
    parser.add_argument('--target_episodes', type=int, default=8)
    parser.add_argument('--validation_episodes', type=int, default=2)
    parser.add_argument('--episode_frames', type=int, default=8)
    parser.add_argument('--query_frames', type=int, default=8)
    parser.add_argument('--inner_steps', type=int, default=20)
    parser.add_argument('--inner_bs', type=int, default=2)
    parser.add_argument('--inner_lr', type=float, default=5e-4)
    parser.add_argument('--inner_wd', type=float, default=5e-2)
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--frame_depth', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--milestones', type=int, nargs='+', default=[5, 10, 20, 30, 50])
    args = parser.parse_args()

    candidate_layers = tuple(sorted(set(args.candidate_layers)))
    if not candidate_layers or candidate_layers[0] < 0 or candidate_layers[-1] > 10:
        raise ValueError('candidate layers must lie in [0,10]')
    if args.validation_episodes < 1 or args.validation_episodes >= args.target_episodes:
        raise ValueError('validation episodes must be in [1,target_episodes-1]')
    if max(args.milestones) > args.epochs:
        raise ValueError('milestones cannot exceed epochs')
    set_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    teacher, ocbe, student, cfg = load_probe(args.scene, args.probe_dir, device)
    if args.topk != int(cfg.get('k', args.topk)):
        raise ValueError(f'top-k mismatch: DLBM={args.topk}, Probe={cfg.get("k")}')
    if args.inner_lr != float(cfg.get('lr', args.inner_lr)):
        raise ValueError('inner rollout lr must match Probe lr')
    if args.inner_wd != float(cfg.get('wd', args.inner_wd)):
        raise ValueError('inner rollout weight decay must match Probe weight decay')
    paths, videos = split_video_balanced(
        all_scene_paths(args.scene), args.max_frames_per_video, args.seed,
    )
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    print(
        f'[{args.scene}] relation cache support={len(paths["support"])} '
        f'query={len(paths["query"])} candidates={candidate_layers}', flush=True,
    )
    relation_cache = cache_relation_maps(
        paths['support'], teacher, ocbe, student, cfg, args, device,
    )
    targets = generate_rollout_targets(
        args.scene, paths, teacher, ocbe, student, cfg, candidate_layers, args, device,
    )
    rollout_peak = (torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == 'cuda' else 0.0)
    del teacher, ocbe, student
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    validation = targets[-args.validation_episodes:]
    training = targets[:-args.validation_episodes]
    model = PairedIncrementalRelationNetwork(
        num_layers=12, topk=args.topk, model_dim=args.model_dim,
        heads=args.heads, frame_depth=args.frame_depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    rng = np.random.RandomState(args.seed + 18000)
    history, milestone_metrics = [], {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, grad_norms = [], []
        for episode_index in rng.permutation(len(training)):
            episode = training[int(episode_index)]
            normal, pseudo = episode_maps(relation_cache, episode, device)
            prediction = model(normal, pseudo, candidate_layers=candidate_layers)
            gains = episode['gains'][list(candidate_layers)].to(device)
            loss = F.mse_loss(prediction, gains)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite DLBM loss epoch={epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f'non-finite DLBM gradient epoch={epoch}')
            optimizer.step()
            losses.append(float(loss.detach()))
            grad_norms.append(float(grad_norm.detach()))
        row = {
            'epoch': epoch,
            'train_mse': float(np.mean(losses)),
            'grad_norm': float(np.mean(grad_norms)),
        }
        if epoch in args.milestones:
            metrics = metrics_from_episodes(
                model, relation_cache, validation, candidate_layers, device,
            )
            row['validation'] = public_metrics(metrics)
            milestone_metrics[str(epoch)] = public_metrics(metrics)
            torch.save({
                'dlbm': model.state_dict(), 'epoch': epoch, 'args': vars(args),
                'validation': public_metrics(metrics),
            }, checkpoint_dir / f'ep{epoch:03d}.pth')
            print(
                f'  ep{epoch:02d}/{args.epochs} train={row["train_mse"]:.4e} '
                f'val={metrics["mse"]:.4e} sign={metrics["sign_accuracy"]:.3f} '
                f'boundary=L{metrics["boundary"]} oracle=L{metrics["oracle_boundary"]}',
                flush=True,
            )
        history.append(row)

    final_validation = metrics_from_episodes(
        model, relation_cache, validation, candidate_layers, device,
    )
    destroyed = metrics_from_episodes(
        model, relation_cache, validation, candidate_layers, device, destroy_pairing=True,
    )
    full_episode = {
        'input_indices': list(range(len(relation_cache['paths']))),
        'gains': torch.zeros(11), 'valid': torch.ones(11, dtype=torch.bool),
    }
    model.eval()
    with torch.no_grad():
        normal, pseudo = episode_maps(relation_cache, full_episode, device)
        candidate_logits = model(normal, pseudo, candidate_layers=candidate_layers).cpu()
    full_logits = complete_vector(candidate_logits, candidate_layers)
    final_policy = policy_from_adjacent_logits(full_logits)
    sensitivity = float(
        (final_validation['_predictions'] - destroyed['_predictions']).abs().mean(),
    )
    train_target = torch.stack([
        episode['gains'][list(candidate_layers)] for episode in training
    ])
    validation_target = torch.stack([
        episode['gains'][list(candidate_layers)] for episode in validation
    ])
    constant_prediction = train_target.mean(dim=0).expand_as(validation_target)
    constant_baseline = {
        'mse': float(F.mse_loss(constant_prediction, validation_target)),
        'mae': float(F.l1_loss(constant_prediction, validation_target)),
        'sign_accuracy': float(
            ((constant_prediction > 0) == (validation_target > 0)).float().mean(),
        ),
    }
    target_rows = [{
        'episode': episode['episode'],
        'input_indices': episode['input_indices'],
        'gains': [float(value) for value in episode['gains']],
        'valid': [bool(value) for value in episode['valid']],
    } for episode in targets]
    result = {
        'format_version': 6,
        'scene': args.scene,
        'method': 'paired_incremental_relation_difference_in_differences',
        'probe_checkpoint': str(probe_path(args.probe_dir, args.scene)),
        'probe_mutated': False,
        'args': vars(args),
        'candidate_layers': list(candidate_layers),
        'split': {'support_frames': len(paths['support']), 'query_frames': len(paths['query']),
                  'videos': videos},
        'rollout_targets': target_rows,
        'history': history,
        'milestones': milestone_metrics,
        'validation': public_metrics(final_validation),
        'candidate_constant_baseline': constant_baseline,
        'destroyed_pairing_validation': public_metrics(destroyed),
        'paired_input_sensitivity': sensitivity,
        'final_policy': {
            'boundary': final_policy.boundary,
            'selected_taps': list(range(final_policy.boundary, 12)),
            'adjacent_predicted_gains': [float(value) for value in full_logits],
            'cumulative_utilities': [float(value) for value in final_policy.cumulative_utilities],
        },
        'rollout_peak_gpu_gib': rollout_peak,
        'model_parameters': sum(parameter.numel() for parameter in model.parameters()),
    }
    torch.save({'dlbm': model.state_dict(), 'result': result}, out_dir / 'dlbm_final.pth')
    write_json(out_dir / 'result.json', result)
    print(
        f'[{args.scene}] final boundary=L{final_policy.boundary} '
        f'val_sign={final_validation["sign_accuracy"]:.3f} '
        f'pair_sensitivity={sensitivity:.3e} peak={rollout_peak:.2f} GiB', flush=True,
    )


if __name__ == '__main__':
    main()
