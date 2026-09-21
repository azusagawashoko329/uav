"""Read-only full-normal-scene audit for a trained v013 DLBM."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_DLBM.cross_layer_predictor import CrossLayerPredictor
from experiment_DLBM.predictive_order_dlbm import PredictiveOrderDLBM
from experiment_DLBM.run_boundary_v001 import (
    all_scene_paths, load_probe, split_video_balanced,
)
from experiment_DLBM.train_boundary_v006 import cache_relation_maps
from experiment_DLBM.train_boundary_v007 import video_indices
from experiment_DLBM.train_predictive_order_dlbm import evaluate_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--probe_dir', default='experiment_DLBM/run/probe/ckpt')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--max_frames_per_video', type=int, default=48)
    parser.add_argument('--cache_bs', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pseudo_min_area', type=float, default=0.02)
    parser.add_argument('--pseudo_max_area', type=float, default=0.15)
    parser.add_argument('--pseudo_mode', choices=['cutpaste', 'multi'], default='multi')
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--eval_pairs', type=int, default=2048)
    parser.add_argument('--eval_nulls', type=int, default=16)
    parser.add_argument('--boundary_evidence', choices=['improvement', 'novelty'],
                        default='improvement')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    teacher, ocbe, student, cfg = load_probe(args.scene, args.probe_dir, device)
    split, _ = split_video_balanced(
        all_scene_paths(args.scene), args.max_frames_per_video, args.seed,
    )
    selected_paths = [*split['support'], *split['query']]
    cache = cache_relation_maps(
        selected_paths, teacher, ocbe, student, cfg, args, device,
    )
    del teacher, ocbe, student
    saved = torch.load(args.model, map_location='cpu')
    predictor = CrossLayerPredictor(channels=args.channels).to(device)
    predictor.load_state_dict(saved['predictor'])
    predictor.eval().requires_grad_(False)
    model = PredictiveOrderDLBM(
        predictor_channels=args.channels, topk=args.topk,
        model_dim=args.model_dim, heads=args.heads,
    ).to(device)
    model.load_state_dict(saved['dlbm']); model.eval()
    policy = evaluate_policy(
        predictor, model, cache, video_indices(cache['paths']), args, device,
    )
    output = {
        'scene': args.scene, 'n_frames': len(cache['paths']),
        'n_videos': len(video_indices(cache['paths'])), 'full_policy': policy,
    }
    target = Path(args.model).with_name('full_scene_audit.json')
    target.write_text(json.dumps(output, indent=2) + '\n')
    print(
        f'[{args.scene}] full=L{policy["boundary"]} '
        f'evidence={[round(policy["gate_evidence"][x], 6) for x in ("M", "S")]}',
        flush=True,
    )


if __name__ == '__main__':
    main()
