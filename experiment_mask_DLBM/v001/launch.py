#!/usr/bin/env python3
"""Dynamically schedule all seven DLBM v001 scene jobs over the requested GPUs."""
import argparse
import json
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = VERSION_ROOT.parent
ROOT = EXPERIMENT_ROOT.parent
VERSION = 'v001_relation_changepoint'
DEFAULT_OUT = EXPERIMENT_ROOT / 'run' / 'dlbm' / VERSION
SPLIT_ROOT = EXPERIMENT_ROOT / 'splits' / 'seed42'
PROBE_DIR = EXPERIMENT_ROOT / 'run' / 'probe' / 'ckpt'
SCENES = ('bike', 'crossroads', 'farmland', 'highway', 'railway', 'solar', 'vehicle')


def timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )


def manifest_count(scene):
    return sum(
        1 for value in (SPLIT_ROOT / scene / 'boundary_train.txt').read_text().splitlines()
        if value.strip() and not value.startswith('#')
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--out_root', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--min_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=7)
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(',') if value.strip()]
    if not gpus:
        raise ValueError('at least one GPU is required')
    out_root = args.out_root.resolve()
    ckpt_dir = out_root / 'ckpt'
    decision_dir = out_root / 'decision'
    log_dir = out_root / 'logs'
    status_dir = out_root / 'status'
    for path in (ckpt_dir, decision_dir, log_dir, status_dir):
        path.mkdir(parents=True, exist_ok=True)

    config = {
        'schema_version': 1,
        'version': VERSION,
        'stage': 'boundary_learning_and_frozen_decision',
        'scenes': list(SCENES),
        'gpus': gpus,
        'scheduling': 'dynamic_largest_boundary_train_first',
        'probe_dir': str(PROBE_DIR),
        'split_root': str(SPLIT_ROOT),
        'output_layout': {
            'ckpt': str(ckpt_dir),
            'decision': str(decision_dir),
        },
        'training': {
            'seed': 42,
            'batch_size': args.batch_size,
            'workers': args.workers,
            'max_epochs': args.max_epochs,
            'min_epochs': args.min_epochs,
            'early_stopping_patience': args.patience,
            'epoch_selection': (
                'held-out 20% of boundary_train inside every subvideo; '
                'boundary_decision is never consulted'
            ),
            'optimizer': 'AdamW lr=5e-4 wd=1e-2; ReduceLROnPlateau',
            'amp': True,
        },
        'model': {
            'feature_dim': 768,
            'token_dim': 64,
            'state_dim': 32,
            'heads': 4,
            'patches': 256,
            'boundary_tokens_per_layer': '256 local transition + 1 global transition',
            'masked_target': 'complete next-layer cosine-distance map',
            'boundary': 'single ordered two-regime change point; L11 fixed anchor',
        },
        'decision_contract': (
            'same-distribution frozen inference only; no train-decision gap is a feature or loss'
        ),
    }
    config_path = out_root / 'config.json'
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding='utf-8'))
        if existing != config:
            raise RuntimeError(f'existing run config differs: {config_path}')
    else:
        write_json(config_path, config)

    ordered = sorted(SCENES, key=manifest_count, reverse=True)
    jobs = queue.Queue()
    for scene in ordered:
        jobs.put(scene)
    failures = []
    print(f'[{timestamp()}] launch {VERSION} gpus={gpus} queue={ordered}', flush=True)

    def worker(gpu):
        while True:
            try:
                scene = jobs.get_nowait()
            except queue.Empty:
                print(f'[{timestamp()}] gpu={gpu} queue empty', flush=True)
                return
            checkpoint = ckpt_dir / f'{scene}.pth'
            decision = decision_dir / f'{scene}.json'
            done = status_dir / f'{scene}.done.json'
            failed = status_dir / f'{scene}.failed.json'
            if done.is_file() and checkpoint.is_file() and decision.is_file():
                print(f'[{timestamp()}] gpu={gpu} skip completed scene={scene}', flush=True)
                jobs.task_done()
                continue
            done.unlink(missing_ok=True)
            failed.unlink(missing_ok=True)
            command = [
                sys.executable,
                str(VERSION_ROOT / 'train.py'),
                '--scene', scene,
                '--gpu', str(gpu),
                '--split_root', str(SPLIT_ROOT),
                '--probe_dir', str(PROBE_DIR),
                '--out_root', str(out_root),
                '--seed', '42',
                '--batch_size', str(args.batch_size),
                '--workers', str(args.workers),
                '--max_epochs', str(args.max_epochs),
                '--min_epochs', str(args.min_epochs),
                '--patience', str(args.patience),
            ]
            started = timestamp()
            log_path = log_dir / f'{scene}.log'
            print(f'[{started}] gpu={gpu} start scene={scene} frames={manifest_count(scene)}', flush=True)
            with log_path.open('w', encoding='utf-8') as log:
                log.write(f'[{started}] command={" ".join(command)}\n')
                log.flush()
                result = subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                )
            record = {
                'scene': scene,
                'gpu': gpu,
                'started_at': started,
                'ended_at': timestamp(),
                'exit_code': result.returncode,
                'checkpoint': str(checkpoint),
                'decision': str(decision),
                'log': str(log_path),
            }
            if (
                result.returncode == 0
                and checkpoint.is_file() and checkpoint.stat().st_size
                and decision.is_file() and decision.stat().st_size
            ):
                write_json(done, record)
                print(f'[{timestamp()}] gpu={gpu} completed scene={scene}', flush=True)
            else:
                write_json(failed, record)
                failures.append(scene)
                print(
                    f'[{timestamp()}] gpu={gpu} FAILED scene={scene} '
                    f'exit={result.returncode}',
                    flush=True,
                )
            jobs.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), name=f'gpu-{gpu}')
        for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    incomplete = [
        scene for scene in SCENES
        if not (status_dir / f'{scene}.done.json').is_file()
    ]
    if failures or incomplete:
        raise SystemExit(f'{VERSION} failed: failures={failures} incomplete={incomplete}')
    (out_root / 'COMPLETE').write_text(timestamp() + '\n', encoding='utf-8')
    print(f'[{timestamp()}] all seven {VERSION} jobs complete', flush=True)


if __name__ == '__main__':
    main()
