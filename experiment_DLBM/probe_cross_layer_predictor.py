"""Train/evaluate the aligned ConvGRU representation without selecting a boundary."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_DLBM.common import SCENES, set_seed
from experiment_DLBM.cross_layer_predictor import CrossLayerPredictor
from experiment_DLBM.run_boundary_v001 import all_scene_paths, load_probe, split_video_balanced
from experiment_DLBM.train_boundary_v006 import cache_relation_maps


def prediction_loss(model, maps):
    output = model(maps)
    target = maps[:, :-1].reshape_as(output['prediction'])
    return F.smooth_l1_loss(output['prediction'], target), output


def independent_layer_roll(maps, seed):
    """Destroy cross-layer correspondence while preserving every layer's local map."""
    generator = torch.Generator(device=maps.device).manual_seed(seed)
    side = int(round(maps.shape[-1] ** 0.5))
    grid = maps.reshape(len(maps), maps.shape[1], side, side).clone()
    for frame in range(len(grid)):
        for layer in range(grid.shape[1]):
            dy = int(torch.randint(0, side, (), generator=generator, device=maps.device))
            dx = int(torch.randint(0, side, (), generator=generator, device=maps.device))
            grid[frame, layer] = torch.roll(grid[frame, layer], (dy, dx), dims=(-2, -1))
    return grid.flatten(2)


@torch.no_grad()
def evaluate(model, cache, device, batch_size, seed):
    model.eval()
    result = {}
    for branch in ('normal', 'pseudo'):
        maps = cache[branch].to(device=device, dtype=torch.float32)
        variants = {'aligned': maps,
                    'independent_layer_roll': independent_layer_roll(maps, seed)}
        for name, values in variants.items():
            losses, absolute, count = [], [], 0
            for start in range(0, len(values), batch_size):
                batch = values[start:start + batch_size]
                loss, output = prediction_loss(model, batch)
                target = batch[:, :-1].reshape_as(output['prediction'])
                losses.append(float(loss) * len(batch))
                absolute.append(float((output['prediction'] - target).abs().mean()) * len(batch))
                count += len(batch)
            result[f'{branch}_{name}_smooth_l1'] = sum(losses) / count
            result[f'{branch}_{name}_mae'] = sum(absolute) / count
    result['normal_roll_loss_ratio'] = (
        result['normal_independent_layer_roll_smooth_l1']
        / max(result['normal_aligned_smooth_l1'], 1e-12)
    )
    result['pseudo_over_normal_loss_ratio'] = (
        result['pseudo_aligned_smooth_l1']
        / max(result['normal_aligned_smooth_l1'], 1e-12)
    )
    return result


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
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, choices=[1, 3], default=3)
    parser.add_argument('--cell', choices=['gru', 'residual'], default='gru')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps_per_epoch', type=int, default=44)
    parser.add_argument('--frame_batch', type=int, default=24)
    parser.add_argument('--eval_bs', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher, ocbe, student, cfg = load_probe(args.scene, args.probe_dir, device)
    paths, videos = split_video_balanced(
        all_scene_paths(args.scene), args.max_frames_per_video, args.seed,
    )
    support = cache_relation_maps(paths['support'], teacher, ocbe, student, cfg, args, device)
    query = cache_relation_maps(paths['query'], teacher, ocbe, student, cfg, args, device)
    del teacher, ocbe, student
    model = CrossLayerPredictor(
        channels=args.channels, kernel_size=args.kernel_size, cell=args.cell,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    rng = np.random.RandomState(args.seed + 30000)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, gradients = [], []
        for _ in range(args.steps_per_epoch):
            index = torch.as_tensor(
                rng.randint(0, len(support['normal']), size=args.frame_batch),
                dtype=torch.long,
            )
            maps = support['normal'].index_select(0, index).to(
                device=device, dtype=torch.float32,
            )
            loss, _ = prediction_loss(model, maps)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite predictor loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
            gradients.append(float(gradient.detach()))
        row = {'epoch': epoch, 'train_smooth_l1': float(np.mean(losses)),
               'grad_norm': float(np.mean(gradients))}
        if epoch in {1, 5, args.epochs}:
            row['query'] = evaluate(
                model, query, device, args.eval_bs, args.seed + 31000,
            )
            print(f'[{args.scene}] ep{epoch:02d} train={row["train_smooth_l1"]:.6f} '
                  f'query={row["query"]["normal_aligned_smooth_l1"]:.6f} '
                  f'roll_ratio={row["query"]["normal_roll_loss_ratio"]:.3f}', flush=True)
        history.append(row)
    result = {
        'scene': args.scene,
        'method': 'normal_only_aligned_cross_layer_convgru_probe',
        'args': vars(args),
        'split': {'support_frames': len(support['paths']), 'query_frames': len(query['paths']),
                  'videos': videos},
        'history': history,
        'final_query': evaluate(model, query, device, args.eval_bs, args.seed + 31000),
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
    }
    torch.save({'predictor': model.state_dict(), 'result': result}, out_dir / 'predictor.pth')
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
