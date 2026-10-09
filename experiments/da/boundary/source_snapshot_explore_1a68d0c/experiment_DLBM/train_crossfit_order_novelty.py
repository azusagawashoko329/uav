"""Train v016 with two-fold video-level cross-fitted order novelty."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_DLBM.common import SCENES, probe_path, set_seed
from experiment_DLBM.cross_layer_predictor import CrossLayerPredictor
from experiment_DLBM.predictive_order_dlbm import PredictiveOrderDLBM
from experiment_DLBM.probe_cross_layer_predictor import evaluate as evaluate_predictor
from experiment_DLBM.run_boundary_v001 import all_scene_paths, load_probe, split_video_balanced
from experiment_DLBM.train_boundary_v006 import cache_relation_maps
from experiment_DLBM.train_boundary_v007 import video_indices
from experiment_DLBM.train_episode_innovation_boundary import file_sha256
from experiment_DLBM.train_predictive_order_dlbm import evaluate_policy, train_model


def fit_fold(name, train_cache, heldout_cache, predictor, args, device, seed_offset):
    set_seed(args.seed + seed_offset)
    model = PredictiveOrderDLBM(
        predictor_channels=args.channels, topk=args.topk,
        model_dim=args.model_dim, heads=args.heads,
    ).to(device)
    train_groups = video_indices(train_cache['paths'])
    heldout_groups = video_indices(heldout_cache['paths'])
    history, milestones = train_model(
        predictor, model, train_cache, train_groups, args, device,
        heldout_cache, heldout_groups,
    )
    heldout = evaluate_policy(
        predictor, model, heldout_cache, heldout_groups, args, device,
    )
    training = evaluate_policy(
        predictor, model, train_cache, train_groups, args, device,
    )
    return model, {
        'name': name, 'train_videos': len(train_groups),
        'heldout_videos': len(heldout_groups), 'history': history,
        'milestones': milestones, 'heldout_policy': heldout,
        'training_policy': training,
    }


def combine_crossfit(model, folds):
    total = sum(fold['heldout_videos'] for fold in folds)
    evidence = {}
    for gate in ('M', 'S'):
        evidence[gate] = sum(
            fold['heldout_policy']['gate_evidence'][gate] * fold['heldout_videos']
            for fold in folds
        ) / total
    boundary, probabilities, utilities = model.boundary_distribution(
        torch.tensor(evidence['M']), torch.tensor(evidence['S']),
    )
    return {
        'boundary': boundary, 'selected_taps': list(range(boundary, 12)),
        'boundary_evidence': 'crossfit_novelty', 'gate_evidence': evidence,
        'probabilities': {'D': float(probabilities[0]),
                          'MD': float(probabilities[1]),
                          'ALL': float(probabilities[2])},
        'utilities': {'D': float(utilities[0]), 'MD': float(utilities[1]),
                      'ALL': float(utilities[2])},
    }


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
    parser.add_argument('--pseudo_mode', choices=['cutpaste', 'multi'], default='multi')
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--predictor_batch', type=int, default=24)
    parser.add_argument('--predictor_checkpoint', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--steps_per_epoch', type=int, default=30)
    parser.add_argument('--episode_frames', type=int, default=32)
    parser.add_argument('--pair_samples', type=int, default=256)
    parser.add_argument('--eval_pairs', type=int, default=2048)
    parser.add_argument('--eval_nulls', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--milestones', type=int, nargs='+', default=[5, 10, 20, 30])
    args = parser.parse_args()
    # evaluate_policy reads this field; no alternative evidence is exposed here.
    args.boundary_evidence = 'novelty'
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = probe_path(args.probe_dir, args.scene); before = file_sha256(checkpoint)
    teacher, ocbe, student, cfg = load_probe(args.scene, args.probe_dir, device)
    paths, videos = split_video_balanced(
        all_scene_paths(args.scene), args.max_frames_per_video, args.seed,
    )
    support = cache_relation_maps(
        paths['support'], teacher, ocbe, student, cfg, args, device,
    )
    query = cache_relation_maps(
        paths['query'], teacher, ocbe, student, cfg, args, device,
    )
    del teacher, ocbe, student
    saved_predictor = torch.load(args.predictor_checkpoint, map_location='cpu')
    predictor = CrossLayerPredictor(channels=args.channels).to(device)
    predictor.load_state_dict(saved_predictor['predictor'])
    predictor.eval().requires_grad_(False)
    predictor_query = evaluate_predictor(
        predictor, query, device, args.predictor_batch, args.seed + 101000,
    )
    print(
        f'[{args.scene}] predictor={predictor_query["normal_aligned_smooth_l1"]:.6g} '
        f'roll={predictor_query["normal_roll_loss_ratio"]:.3f}', flush=True,
    )
    model_a, fold_a = fit_fold(
        'support_to_query', support, query, predictor, args, device, 102000,
    )
    print(
        f'[{args.scene}] foldA heldout=L{fold_a["heldout_policy"]["boundary"]}',
        flush=True,
    )
    model_b, fold_b = fit_fold(
        'query_to_support', query, support, predictor, args, device, 103000,
    )
    print(
        f'[{args.scene}] foldB heldout=L{fold_b["heldout_policy"]["boundary"]}',
        flush=True,
    )
    folds = [fold_a, fold_b]
    policy = combine_crossfit(model_a, folds)
    if before != file_sha256(checkpoint):
        raise RuntimeError('Probe checkpoint changed')
    result = {
        'format_version': 16, 'scene': args.scene,
        'method': 'twofold_video_crossfit_predictive_order_novelty_dlbm',
        'probe_checkpoint': str(checkpoint), 'probe_sha256': before,
        'probe_mutated': False, 'args': vars(args), 'videos': videos,
        'predictor_query': predictor_query, 'folds': folds,
        'crossfit_policy': policy,
    }
    torch.save({
        'predictor': predictor.state_dict(),
        'dlbm_support_to_query': model_a.state_dict(),
        'dlbm_query_to_support': model_b.state_dict(), 'result': result,
    }, out_dir / 'model.pth')
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(
        f'[{args.scene}] crossfit=L{policy["boundary"]} '
        f'evidence={[round(policy["gate_evidence"][x], 6) for x in ("M", "S")]}',
        flush=True,
    )


if __name__ == '__main__':
    main()
