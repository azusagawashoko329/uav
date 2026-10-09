"""Read-only audit of D/M/S episode ordering signals on one Probe10 scene."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_DLBM.common import SCENES, set_seed
from experiment_DLBM.episode_innovation_boundary import SCALE_TRANSITIONS
from experiment_DLBM.run_boundary_v001 import (
    all_scene_paths, load_probe, split_video_balanced,
)
from experiment_DLBM.train_boundary_v006 import cache_relation_maps
from experiment_DLBM.train_boundary_v007 import video_indices
from experiment_DLBM.train_boundary_v008 import group_score
from experiment_DLBM.train_boundary_v008 import structural_nulls


def pair_flip_rate(base, candidate):
    first, second = torch.triu_indices(len(base), len(base), offset=1)
    if not len(first):
        return 0.0
    return float(((
        (base.index_select(0, first) - base.index_select(0, second))
        * (candidate.index_select(0, first) - candidate.index_select(0, second))
    ) < 0).float().mean())


def audit(cache, topk):
    groups = video_indices(cache['paths'])
    output = {}
    for scale, (base_layers, candidate_layers) in SCALE_TRANSITIONS.items():
        row = {}
        for branch in ('normal', 'pseudo'):
            flips, medians = [], []
            for video in sorted(groups):
                index = torch.as_tensor(groups[video], dtype=torch.long)
                maps = cache[branch].index_select(0, index).float()
                base = group_score(maps, base_layers, topk)
                candidate = group_score(maps, candidate_layers, topk)
                flips.append(pair_flip_rate(base, candidate))
                medians.append(float((candidate - base).median()))
            row[f'{branch}_flip'] = float(np.mean(flips))
            row[f'{branch}_delta_median'] = float(np.mean(medians))
        output[scale] = row
    winner = max(output, key=lambda scale: output[scale]['normal_flip'])
    return {'scales': output, 'normal_flip_winner': winner,
            'boundary': {'D': 8, 'M': 4, 'S': 0}[winner]}


def transition_values(normal, pseudo, base_layers, candidate_layers, topk):
    base_n = group_score(normal, base_layers, topk)
    candidate_n = group_score(normal, candidate_layers, topk)
    base_p = group_score(pseudo, base_layers, topk)
    candidate_p = group_score(pseudo, candidate_layers, topk)
    return pair_flip_rate(base_n, candidate_n), float((candidate_p - base_p).median())


def pareto_audit(cache, topk):
    groups = video_indices(cache['paths'])
    output = {}
    real_transitions = {'M': SCALE_TRANSITIONS['M'], 'S': SCALE_TRANSITIONS['S']}
    for gate, (base, candidate) in real_transitions.items():
        evidence, comparable, details = [], [], []
        for video in sorted(groups):
            index = torch.as_tensor(groups[video], dtype=torch.long)
            normal = cache['normal'].index_select(0, index).float()
            pseudo = cache['pseudo'].index_select(0, index).float()
            real_flip, real_pseudo = transition_values(
                normal, pseudo, base, candidate, topk,
            )
            votes = []
            for null_base, null_candidate in structural_nulls(gate):
                null_flip, null_pseudo = transition_values(
                    normal, pseudo, null_base, null_candidate, topk,
                )
                real_dominates = real_flip > null_flip and real_pseudo <= null_pseudo
                null_dominates = null_flip > real_flip and null_pseudo <= real_pseudo
                votes.append(float(real_dominates) - float(null_dominates))
            evidence.append(float(np.mean(votes)))
            comparable.append(float(np.mean(np.abs(votes))))
            details.append({'video': video, 'real_normal_flip': real_flip,
                            'real_pseudo_delta': real_pseudo,
                            'pareto_evidence': evidence[-1]})
        output[gate] = {
            'evidence': float(np.mean(evidence)),
            'comparable_fraction': float(np.mean(comparable)),
            'video_rows': details,
        }
    m, s = output['M']['evidence'], output['S']['evidence']
    if s > 0 and s >= m:
        boundary = 0
    elif m > 0:
        boundary = 4
    else:
        boundary = 8
    return {'gates': output, 'boundary': boundary}


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
    parser.add_argument('--pseudo_mode', choices=['cutpaste', 'multi'], default='cutpaste')
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    teacher, ocbe, student, cfg = load_probe(args.scene, args.probe_dir, device)
    paths, videos = split_video_balanced(
        all_scene_paths(args.scene), args.max_frames_per_video, args.seed,
    )
    result = {'scene': args.scene, 'split': {'videos': videos}}
    for split in ('support', 'query'):
        cache = cache_relation_maps(
            paths[split], teacher, ocbe, student, cfg, args, device,
        )
        result[split] = audit(cache, args.topk)
        result[split]['pareto'] = pareto_audit(cache, args.topk)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(
        f'[{args.scene}] support=L{result["support"]["boundary"]} '
        f'query=L{result["query"]["boundary"]} '
        f'pareto=L{result["query"]["pareto"]["boundary"]} '
        f'query_flip={[round(result["query"]["scales"][x]["normal_flip"], 4) for x in SCALE_TRANSITIONS]}',
        flush=True,
    )


if __name__ == '__main__':
    main()
