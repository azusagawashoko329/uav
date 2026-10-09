#!/usr/bin/env python3
"""Dynamically schedule the seven fixed-subset Probe10 jobs across two GPUs."""
import argparse
import json
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from prepare_splits import SCENES, SPLITS, build_plan, write_or_verify


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / 'experiment_mask_DLBM'
SPLIT_ROOT = EXPERIMENT_ROOT / 'splits' / 'seed42'
PROBE_ROOT = EXPERIMENT_ROOT / 'run' / 'probe'


def timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0,1')
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(',') if value.strip()]
    if not gpus:
        raise ValueError('at least one GPU is required')

    manifests, base_summary = build_plan(
        '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly', seed=42)
    split_summary = write_or_verify(SPLIT_ROOT, manifests, base_summary, write=False)
    ckpt_dir = PROBE_ROOT / 'ckpt'
    log_dir = PROBE_ROOT / 'logs'
    status_dir = PROBE_ROOT / 'status'
    for path in (ckpt_dir, log_dir, status_dir, EXPERIMENT_ROOT / 'run' / 'boundary'):
        path.mkdir(parents=True, exist_ok=True)

    hyperparameters = {
        'img': 256,
        'taps': list(range(12)),
        'code_dim': 64,
        'student_depth': 12,
        'tap_align': 'absolute',
        'topk': 5,
        'loss': 'topk',
        'epochs': 10,
        'batch_size': 64,
        'learning_rate': 5e-4,
        'weight_decay': 5e-2,
        'workers': 8,
        'seed': 42,
    }
    run_config = {
        'stage': 'probe',
        'experiment': 'mask_DLBM',
        'scenes': list(SCENES),
        'gpus': gpus,
        'scheduling': 'dynamic_largest_subset_first',
        'subset': 'probe_train',
        'split_root': str(SPLIT_ROOT),
        'split_fingerprint_sha256': split_summary['split_fingerprint_sha256'],
        'scene_manifest_sha256': {
            scene: split_summary['manifests'][scene]['probe_train']['sha256'] for scene in SCENES
        },
        'hyperparameters': hyperparameters,
    }
    config_path = PROBE_ROOT / 'config.json'
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding='utf-8'))
        if existing != run_config:
            raise RuntimeError(f'existing run config differs; refusing to mix products in {PROBE_ROOT}')
    else:
        write_json(config_path, run_config)

    scene_queue = queue.Queue()
    ordered_scenes = sorted(
        SCENES, key=lambda scene: split_summary['scenes'][scene]['split_counts']['probe_train'],
        reverse=True)
    for scene in ordered_scenes:
        scene_queue.put(scene)
    failures = []
    print(f'[{timestamp()}] launch Probe10 gpus={gpus} queue={ordered_scenes}', flush=True)
    print(f"[{timestamp()}] fixed split fingerprint={split_summary['split_fingerprint_sha256']}", flush=True)

    def worker(gpu):
        while True:
            try:
                scene = scene_queue.get_nowait()
            except queue.Empty:
                print(f'[{timestamp()}] gpu={gpu} queue empty', flush=True)
                return
            checkpoint = ckpt_dir / f'{scene}.pth'
            done = status_dir / f'{scene}.done.json'
            failed = status_dir / f'{scene}.failed.json'
            if done.is_file() and checkpoint.is_file() and checkpoint.stat().st_size > 0:
                print(f'[{timestamp()}] gpu={gpu} skip completed scene={scene}', flush=True)
                scene_queue.task_done()
                continue
            done.unlink(missing_ok=True)
            failed.unlink(missing_ok=True)
            manifest = SPLIT_ROOT / scene / 'probe_train.txt'
            command = [
                sys.executable, str(ROOT / 'train.py'),
                '--scene', scene, '--gpu', str(gpu), '--img', '256',
                '--taps', *[str(index) for index in range(12)],
                '--code_dim', '64', '--depth', '12', '--tap_align', 'absolute',
                '--k', '5', '--loss', 'topk', '--epochs', '10', '--bs', '64',
                '--lr', '5e-4', '--wd', '5e-2', '--workers', '8', '--seed', '42',
                '--manifest', str(manifest), '--out', str(ckpt_dir),
            ]
            print(f'[{timestamp()}] gpu={gpu} start scene={scene}', flush=True)
            started = timestamp()
            log_path = log_dir / f'{scene}.log'
            with log_path.open('w', encoding='utf-8') as log:
                log.write(f'[{started}] command={" ".join(command)}\n')
                log.flush()
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            record = {
                'scene': scene,
                'gpu': gpu,
                'started_at': started,
                'ended_at': timestamp(),
                'exit_code': result.returncode,
                'manifest': str(manifest),
                'manifest_sha256': split_summary['manifests'][scene]['probe_train']['sha256'],
                'checkpoint': str(checkpoint),
                'log': str(log_path),
            }
            if result.returncode == 0 and checkpoint.is_file() and checkpoint.stat().st_size > 0:
                write_json(done, record)
                print(f'[{timestamp()}] gpu={gpu} completed scene={scene}', flush=True)
            else:
                write_json(failed, record)
                failures.append(scene)
                print(f'[{timestamp()}] gpu={gpu} FAILED scene={scene} exit={result.returncode}', flush=True)
            scene_queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), name=f'gpu-{gpu}') for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    incomplete = [scene for scene in SCENES if not (status_dir / f'{scene}.done.json').is_file()
                  or not (ckpt_dir / f'{scene}.pth').is_file()]
    if failures or incomplete:
        raise SystemExit(f'Probe10 failed: failures={failures} incomplete={incomplete}')
    (PROBE_ROOT / 'COMPLETE').write_text(timestamp() + '\n', encoding='utf-8')
    print(f'[{timestamp()}] all seven fixed-subset Probe10 checkpoints complete', flush=True)


if __name__ == '__main__':
    main()
