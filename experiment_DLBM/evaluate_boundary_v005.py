"""Post-hoc fixed-action audit for v005; verifies GDAS policy against its true outer objective."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_DLBM.common import SCENES, set_seed, write_json
from experiment_DLBM.dlbm_v005 import ExactBoundarySearchSupernet, area_rank_loss
from experiment_DLBM.run_boundary_v001 import all_scene_paths, load_probe, split_video_balanced
from experiment_DLBM.train_boundary_v002 import balanced_indices
from experiment_DLBM.train_boundary_v005 import (cache_query, cache_support, indexed_taps,
                                                  one_step_parameters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, choices=sorted(SCENES))
    parser.add_argument('--probe_dir', default='experiment_DLBM/run/probe/ckpt')
    parser.add_argument('--out', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--boundaries', type=int, nargs='+', default=[0, 4, 8, 11])
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--support_bs', type=int, default=2)
    parser.add_argument('--query_bs', type=int, default=4)
    parser.add_argument('--cache_bs', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max_frames_per_video', type=int, default=48)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=5e-4)
    parser.add_argument('--pseudo_min_area', type=float, default=0.02)
    parser.add_argument('--pseudo_max_area', type=float, default=0.15)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    teacher, ocbe, student, cfg = load_probe(args.scene, args.probe_dir, device)
    ocbe.requires_grad_(True); student.requires_grad_(True)
    model = ExactBoundarySearchSupernet(ocbe, student, args.boundaries, args.topk).to(device).eval()
    paths, _ = split_video_balanced(
        all_scene_paths(args.scene), args.max_frames_per_video, args.seed,
    )
    support = cache_support(paths['support'], teacher, cfg, args, device)
    query = cache_query(paths['query'], teacher, cfg, args, device)
    del teacher
    if device.type == 'cuda':
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)

    rng = np.random.RandomState(args.seed + 15000)
    losses = {boundary: [] for boundary in args.boundaries}
    pre_update_losses = {boundary: [] for boundary in args.boundaries}
    for episode in range(args.episodes):
        support_indices = balanced_indices(support['paths'], args.support_bs, rng)
        query_indices = balanced_indices(query['paths'], args.query_bs, rng)
        support_taps = indexed_taps(support['taps'], support_indices, device)
        first_taps = indexed_taps(query['first'], query_indices, device)
        second_taps = indexed_taps(query['second'], query_indices, device)
        index = torch.as_tensor(query_indices, dtype=torch.long)
        first_area = query['first_area'].index_select(0, index).to(device)
        second_area = query['second_area'].index_select(0, index).to(device)
        with torch.no_grad():
            pre_first = model(first_taps)
            pre_second = model(second_taps)
            for candidate, boundary in enumerate(args.boundaries):
                pre_update_losses[boundary].append(float(area_rank_loss(
                    pre_first[:, candidate], pre_second[:, candidate], first_area, second_area,
                )))
        for candidate, boundary in enumerate(args.boundaries):
            gate = torch.zeros(len(args.boundaries), device=device); gate[candidate] = 1
            updated, _ = one_step_parameters(
                model, support_taps, gate, args.inner_lr, create_graph=False,
            )
            updated = {name: value.detach() for name, value in updated.items()}
            with torch.no_grad():
                first_all = functional_call(model, updated, (first_taps,))
                second_all = functional_call(model, updated, (second_taps,))
                loss = area_rank_loss(
                    first_all[:, candidate], second_all[:, candidate], first_area, second_area,
                )
            losses[boundary].append(float(loss))
        if (episode + 1) % 8 == 0:
            print(f'[{args.scene}] objective audit {episode + 1}/{args.episodes}', flush=True)

    means = {str(boundary): float(np.mean(losses[boundary])) for boundary in args.boundaries}
    stds = {str(boundary): float(np.std(losses[boundary])) for boundary in args.boundaries}
    pre_means = {str(boundary): float(np.mean(pre_update_losses[boundary]))
                 for boundary in args.boundaries}
    improvements = {str(boundary): pre_means[str(boundary)] - means[str(boundary)]
                    for boundary in args.boundaries}
    oracle = min(args.boundaries, key=lambda boundary: means[str(boundary)])
    pre_oracle = min(args.boundaries, key=lambda boundary: pre_means[str(boundary)])
    output = {
        'scene': args.scene, 'episodes': args.episodes, 'boundaries': args.boundaries,
        'mean_pre_update_rank_loss': pre_means,
        'mean_outer_rank_loss': means, 'std_outer_rank_loss': stds,
        'mean_one_step_improvement': improvements,
        'pre_update_oracle_boundary': pre_oracle,
        'fixed_action_oracle_boundary': oracle,
        'episode_outer_rank_loss': {str(boundary): losses[boundary]
                                    for boundary in args.boundaries},
        'peak_gpu_gib': (torch.cuda.max_memory_allocated(device) / 2**30
                         if device.type == 'cuda' else 0),
    }
    write_json(args.out, output)
    print(f'[{args.scene}] pre=L{pre_oracle} post=L{oracle} losses={means}', flush=True)


if __name__ == '__main__':
    main()
