#!/usr/bin/env python3
"""Validate that every reusable Probe10 checkpoint matches the frozen split and run contract."""
import gc
import hashlib
import json
import re
from pathlib import Path

import torch

from prepare_splits import SCENES


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / 'experiment_mask_DLBM'
PROBE_ROOT = EXPERIMENT_ROOT / 'run' / 'probe'
SPLIT_ROOT = EXPERIMENT_ROOT / 'splits' / 'seed42'
EPOCH_PATTERN = re.compile(r'^\s*ep(\d+)/10 loss=([0-9.eE+-]+)', re.MULTILINE)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_signature(state):
    return tuple((name, tuple(value.shape), str(value.dtype)) for name, value in state.items())


def state_is_finite(state):
    for name, value in state.items():
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f'non-finite tensor: {name}')


def main():
    split_summary = json.loads((SPLIT_ROOT / 'split_summary.json').read_text(encoding='utf-8'))
    run_config = json.loads((PROBE_ROOT / 'config.json').read_text(encoding='utf-8'))
    if not (PROBE_ROOT / 'COMPLETE').is_file():
        raise FileNotFoundError(f'missing completion marker: {PROBE_ROOT / "COMPLETE"}')
    if run_config['split_fingerprint_sha256'] != split_summary['split_fingerprint_sha256']:
        raise ValueError('run config split fingerprint differs from fixed split summary')

    expected_cfg = {
        'img': 256,
        'taps': list(range(12)),
        'code_dim': 64,
        'depth': 12,
        'tap_align': 'absolute',
        'k': 5,
        'loss': 'topk',
        'epochs': 10,
        'bs': 64,
        'lr': 5e-4,
        'wd': 5e-2,
        'workers': 8,
        'seed': 42,
    }
    report = {
        'status': 'passed',
        'split_fingerprint_sha256': split_summary['split_fingerprint_sha256'],
        'expected_probe_config': expected_cfg,
        'scenes': {},
    }
    expected_signatures = None
    for scene, folder in SCENES.items():
        checkpoint_path = PROBE_ROOT / 'ckpt' / f'{scene}.pth'
        log_path = PROBE_ROOT / 'logs' / f'{scene}.log'
        status_path = PROBE_ROOT / 'status' / f'{scene}.done.json'
        status = json.loads(status_path.read_text(encoding='utf-8'))
        if status['exit_code'] != 0:
            raise ValueError(f'{scene}: nonzero recorded exit code {status["exit_code"]}')
        log = log_path.read_text(encoding='utf-8')
        if 'Traceback' in log or 'non-finite' in log:
            raise RuntimeError(f'{scene}: error marker found in log')
        epochs = EPOCH_PATTERN.findall(log)
        if [int(epoch) for epoch, _ in epochs] != list(range(1, 11)):
            raise ValueError(f'{scene}: expected epoch records 1..10, found {epochs}')

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        cfg = checkpoint['cfg']
        for key, expected in expected_cfg.items():
            if cfg.get(key) != expected:
                raise ValueError(f'{scene}: cfg[{key!r}] expected {expected!r}, found {cfg.get(key)!r}')
        manifest = (SPLIT_ROOT / scene / 'probe_train.txt').resolve()
        manifest_info = split_summary['manifests'][scene]['probe_train']
        if Path(cfg['manifest']).resolve() != manifest:
            raise ValueError(f'{scene}: checkpoint references unexpected manifest {cfg["manifest"]}')
        if cfg['manifest_sha256'] != manifest_info['sha256']:
            raise ValueError(f'{scene}: checkpoint manifest hash mismatch')
        if cfg['dataset_frames'] != manifest_info['frames']:
            raise ValueError(f'{scene}: checkpoint frame count mismatch')
        if checkpoint['scene'] != folder:
            raise ValueError(f'{scene}: expected full scene name {folder!r}, found {checkpoint["scene"]!r}')

        signatures = {
            component: state_signature(checkpoint[component]) for component in ('ocbe', 'student')
        }
        if expected_signatures is None:
            expected_signatures = signatures
        elif signatures != expected_signatures:
            raise ValueError(f'{scene}: model state signature differs from the other probes')
        for component in ('ocbe', 'student'):
            state_is_finite(checkpoint[component])

        report['scenes'][scene] = {
            'checkpoint': str(checkpoint_path),
            'checkpoint_bytes': checkpoint_path.stat().st_size,
            'checkpoint_sha256': file_sha256(checkpoint_path),
            'manifest': str(manifest),
            'manifest_sha256': cfg['manifest_sha256'],
            'dataset_frames': cfg['dataset_frames'],
            'epochs_verified': len(epochs),
            'final_loss': float(epochs[-1][1]),
            'all_state_tensors_finite': True,
            'ocbe_state_tensors': len(checkpoint['ocbe']),
            'student_state_tensors': len(checkpoint['student']),
        }
        print(f"PASS {scene:10s} frames={cfg['dataset_frames']:5d} "
              f"final_loss={float(epochs[-1][1]):.4f} size={checkpoint_path.stat().st_size / 2**20:.1f} MiB")
        del checkpoint
        gc.collect()

    audit_path = PROBE_ROOT / 'audit.json'
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'PASS all seven Probe10 checkpoints; audit={audit_path}')


if __name__ == '__main__':
    main()
