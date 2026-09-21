#!/usr/bin/env python3
"""Audit all v001 checkpoints and frozen scene decisions after the launcher completes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


VERSION_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = VERSION_ROOT.parent
ROOT = EXPERIMENT_ROOT.parent
VERSION = 'v001_relation_changepoint'
SCENES = ('bike', 'crossroads', 'farmland', 'highway', 'railway', 'solar', 'vehicle')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root', type=Path,
        default=EXPERIMENT_ROOT / 'run' / 'dlbm' / VERSION,
    )
    args = parser.parse_args()
    root = args.root.resolve()
    rows = {}
    failures = []
    for scene in SCENES:
        checkpoint_path = root / 'ckpt' / f'{scene}.pth'
        decision_path = root / 'decision' / f'{scene}.json'
        if not checkpoint_path.is_file() or not decision_path.is_file():
            failures.append(f'{scene}: missing checkpoint or decision')
            continue
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        decision = json.loads(decision_path.read_text(encoding='utf-8'))
        if checkpoint.get('version') != VERSION or decision.get('version') != VERSION:
            failures.append(f'{scene}: version mismatch')
        if checkpoint.get('scene') != scene or decision.get('scene') != scene:
            failures.append(f'{scene}: scene mismatch')
        for component in ('model', 'density'):
            state = checkpoint.get(component, {})
            if not state or not all(torch.isfinite(value).all() for value in state.values()):
                failures.append(f'{scene}: non-finite or empty {component} state')
        history = checkpoint.get('representation_history', [])
        if not history:
            failures.append(f'{scene}: empty representation history')
            continue
        for row in history:
            values = (
                row['train']['total'], row['train']['grad_norm'],
                row['validation']['total'], row['validation']['prediction'],
                row['validation']['transition'],
                row['validation']['transition_token_variance'],
            )
            if not all(finite_number(value) for value in values):
                failures.append(f'{scene}: non-finite representation metric')
        best_epoch = checkpoint.get('representation_best_epoch')
        if not isinstance(best_epoch, int) or not 1 <= best_epoch <= len(history):
            failures.append(f'{scene}: invalid best epoch {best_epoch}')
        density_history = checkpoint.get('density_history', [])
        if not density_history or not all(
            finite_number(row['train_nll']) and finite_number(row['validation_nll'])
            for row in density_history
        ):
            failures.append(f'{scene}: invalid density history')
        policy = decision.get('policy', {})
        boundary = policy.get('boundary')
        if not isinstance(boundary, int) or not 0 <= boundary <= 11:
            failures.append(f'{scene}: invalid boundary {boundary}')
            continue
        if policy.get('selected_taps') != list(range(boundary, 12)):
            failures.append(f'{scene}: selected taps are not a contiguous suffix')
        probabilities = policy.get('probabilities', [])
        if len(probabilities) != 12 or not all(finite_number(value) for value in probabilities):
            failures.append(f'{scene}: invalid boundary probabilities')
        elif abs(sum(probabilities) - 1.0) > 1e-4:
            failures.append(f'{scene}: probabilities do not sum to one')
        if decision.get('checkpoint_sha256') != sha256_file(checkpoint_path):
            failures.append(f'{scene}: checkpoint hash mismatch in decision')
        manifest = checkpoint['manifests']['boundary_decision']
        if decision['manifest']['sha256'] != manifest['sha256']:
            failures.append(f'{scene}: decision manifest hash mismatch')
        if 'no train-decision gap' not in decision.get('decision_role', ''):
            failures.append(f'{scene}: decision contract missing same-distribution statement')
        intervention = checkpoint.get('representation_interventions', {})
        rows[scene] = {
            'representation_epochs_run': len(history),
            'representation_best_epoch': best_epoch,
            'best_validation': checkpoint['representation_best_validation'],
            'density_epochs_run': len(density_history),
            'density_best_epoch': checkpoint['density_best_epoch'],
            'roll_over_aligned': intervention.get('roll_over_aligned'),
            'without_relation_over_full': intervention.get('without_relation_over_full'),
            'transition_token_variance': history[best_epoch - 1]['validation'][
                'transition_token_variance'
            ],
            'relation_residual_ratio': history[best_epoch - 1]['validation'][
                'relation_residual_ratio'
            ],
            'boundary': boundary,
            'selected_taps': policy['selected_taps'],
            'confidence': policy['confidence'],
            'normalized_entropy': policy['normalized_entropy'],
            'decision_subvideos': len(decision['subvideos']),
            'peak_gpu_gib': decision.get('peak_gpu_gib'),
        }
    result = {
        'schema_version': 1,
        'version': VERSION,
        'root': str(root),
        'passed': not failures and len(rows) == len(SCENES),
        'failures': failures,
        'scenes': rows,
        'interpretation': (
            'The audit checks software/data contracts and numerical validity. '
            'Roll, relation-ablation, entropy, and external Gate agreement remain scientific '
            'qualification metrics rather than pass/fail software assertions.'
        ),
    }
    output = root / 'audit.json'
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
