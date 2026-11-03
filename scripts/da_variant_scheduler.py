#!/usr/bin/env python3
"""Dynamic two-GPU scheduler for the DA six-variant matrix.

Each GPU runs one scene process at a time. When a scene finishes, the next queued scene
starts immediately on the same GPU; there are no fixed-duration turns and no idle card
while the queue is non-empty. GPUs are selected by nvidia-smi index but the child is
pinned by UUID through CUDA_VISIBLE_DEVICES, avoiding CUDA enumeration-order mistakes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCENES = ['bike', 'crossroads', 'farmland', 'highway', 'railway', 'solar', 'vehicle']
DEFAULT_PYTHON = '/home/lyx/miniconda3/envs/DMAD/bin/python'


def now():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def inventory(indices):
    raw = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total',
         '--format=csv,noheader,nounits'], text=True)
    out = {}
    for line in raw.strip().splitlines():
        idx, uuid, name, used, total = [x.strip() for x in line.split(',')]
        idx = int(idx)
        if idx in indices:
            out[idx] = {'uuid': uuid, 'name': name, 'used_mb': int(used), 'total_mb': int(total)}
    return out


def scene_costs(data_root: Path, scenes):
    costs = {}
    for scene in scenes:
        scene_dir = data_root / {
            'bike': 'Bike Roundabout', 'crossroads': 'Crossroads',
            'farmland': 'Farmland Inspection', 'highway': 'Highway',
            'railway': 'Railway Inspection', 'solar': 'Solar Panel Inspection',
            'vehicle': 'Vehicle Roundabout',
        }[scene]
        paths = glob.glob(str(scene_dir / 'sequence*' / 'train' / '*' / '*.jpg'))
        if not paths:
            raise FileNotFoundError(f'no train frames for {scene}: {scene_dir}')
        costs[scene] = len(paths)
    return costs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--experiments', nargs='+', required=True,
                   help='experiment directories relative to repo root')
    p.add_argument('--run-id', default=None)
    p.add_argument('--python', default=DEFAULT_PYTHON)
    p.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--scenes', nargs='+', choices=SCENES, default=SCENES)
    p.add_argument('--max-steps', type=int, default=0,
                   help='>0 limits batches/epoch; for smoke tests only')
    p.add_argument('--poll-seconds', type=float, default=5.0)
    p.add_argument('--max-memory-mb', type=int, default=2000)
    p.add_argument('--resume', action='store_true',
                   help='skip scenes with both results/<scene>.json and ckpt/<scene>_final.pth')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    root = args.repo_root.resolve()
    experiments = [(root / e).resolve() if not Path(e).is_absolute() else Path(e).resolve()
                   for e in args.experiments]
    run_id = args.run_id or (now() + '_da_variant_matrix_s42')

    def display(exp: Path) -> str:
        try:
            return str(exp.relative_to(root))
        except ValueError:
            return str(exp)

    configs = {}
    for exp in experiments:
        cfg_path = exp / 'config.json'
        if not cfg_path.is_file():
            raise FileNotFoundError(cfg_path)
        cfg = json.loads(cfg_path.read_text())
        if cfg.get('dataset') != 'da' or cfg.get('family') != 'appearance':
            raise ValueError(f'wrong dataset/family: {exp}')
        if int(cfg['seed']) != args.seed:
            raise ValueError(f'seed mismatch: {exp} config={cfg["seed"]} scheduler={args.seed}')
        if int(cfg.get('training', {}).get('epochs', args.epochs)) != args.epochs:
            raise ValueError(f'epochs mismatch: {exp} config={cfg.get("training", {}).get("epochs")} scheduler={args.epochs}')
        configs[exp] = cfg

    data_root = Path(configs[experiments[0]]['training'].get(
        'data_root', os.environ.get('UAV_DA_ROOT', '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly')))
    costs = scene_costs(data_root, args.scenes)
    # Longest scenes first (LPT) so the final tail is short; variants keep V0..V5 order.
    scene_order = sorted(args.scenes, key=lambda s: (-costs[s], SCENES.index(s)))
    jobs = []
    for scene in scene_order:
        for exp in experiments:
            jobs.append({'experiment': exp, 'variant_id': configs[exp]['variant_id'], 'scene': scene})

    logs = {exp: exp / 'logs' / f'scheduler_{run_id}.jsonl' for exp in experiments}
    for log in logs.values():
        log.parent.mkdir(parents=True, exist_ok=True)

    def emit(kind, **kw):
        event = {'time': now(), 'run_id': run_id, 'kind': kind, **kw}
        line = json.dumps(event, ensure_ascii=False) + '\n'
        for log in logs.values():
            with log.open('a') as f:
                f.write(line)
        print(line, end='', flush=True)

    inv = inventory(args.gpus)
    missing = [g for g in args.gpus if g not in inv]
    if missing:
        raise RuntimeError(f'requested GPUs missing from nvidia-smi: {missing}')
    busy = {g: inv[g]['used_mb'] for g in args.gpus if inv[g]['used_mb'] > args.max_memory_mb}
    if busy:
        raise RuntimeError(f'requested GPUs are busy above threshold {args.max_memory_mb} MB: {busy}')
    if args.dry_run:
        print(json.dumps({
            'kind': 'dry_run_ok', 'run_id': run_id, 'jobs': len(jobs),
            'gpus': [{'slot': g, **inv[g]} for g in args.gpus],
            'scene_order': scene_order, 'scene_train_frames': costs, 'max_steps': args.max_steps,
            'job_list': [{'variant_id': j['variant_id'], 'scene': j['scene'],
                          'experiment': display(j['experiment'])} for j in jobs],
        }, ensure_ascii=False), flush=True)
        return 0
    emit('start', experiments=[display(e) for e in experiments],
         gpus=[{'slot': g, **inv[g]} for g in args.gpus], jobs=len(jobs),
         epochs=args.epochs, batch=args.batch, workers=args.workers, seed=args.seed,
         scenes=args.scenes, max_steps=args.max_steps,
         scene_order=scene_order, scene_train_frames=costs,
         data_root=str(data_root), policy='dynamic per-GPU queue; next scene starts on free GPU')

    queue = []
    skipped = []
    for job in jobs:
        exp, scene = job['experiment'], job['scene']
        done = (exp / 'results' / f'{scene}.json').is_file() and (exp / 'ckpt' / f'{scene}_final.pth').is_file()
        if args.resume and done:
            skipped.append(job)
        else:
            queue.append(job)
    if skipped:
        emit('resume_skip', jobs=[{'variant_id': j['variant_id'], 'scene': j['scene'],
                                   'experiment': display(j['experiment'])} for j in skipped])

    running = {}
    failures = []
    completed = []

    def launch(slot, job):
        exp, scene, variant_id = job['experiment'], job['scene'], job['variant_id']
        stamp = now()
        log_path = exp / 'logs' / f'{stamp}_{scene}_gpu{slot}.log'
        cmd = [args.python, str(root / 'scripts' / 'da_variant_runner.py'),
               '--experiment', str(exp), '--scene', scene, '--gpu', '0', '--seed', str(args.seed),
               '--epochs', str(args.epochs), '--batch', str(args.batch),
               '--workers', str(args.workers)]
        if args.max_steps:
            cmd += ['--max-steps', str(args.max_steps)]
        if args.resume:
            cmd.append('--overwrite')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': inv[slot]['uuid'],
               'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'PYTHONUNBUFFERED': '1'}
        f = log_path.open('w')
        proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=f,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        running[slot] = {'job': job, 'proc': proc, 'log_file': f,
                         'log_path': log_path, 'started_at': stamp,
                         't0': time.monotonic(), 'command': cmd}
        emit('launch', variant_id=variant_id, experiment=display(exp),
             scene=scene, gpu_slot=slot, uuid=inv[slot]['uuid'], name=inv[slot]['name'],
             pid=proc.pid, log=str(log_path), command=cmd,
             remaining_after_launch=len(queue))

    for slot in args.gpus:
        if queue:
            launch(slot, queue.pop(0))
    while queue or running:
        for slot, state in list(running.items()):
            rc = state['proc'].poll()
            if rc is None:
                continue
            state['log_file'].close()
            job = state['job']
            emit('finish', variant_id=job['variant_id'],
                 experiment=display(job['experiment']), scene=job['scene'],
                 gpu_slot=slot, pid=state['proc'].pid, returncode=rc,
                 log=str(state['log_path']), elapsed_s=time.monotonic() - state['t0'])
            if rc == 0:
                completed.append(job)
            else:
                failures.append({'variant_id': job['variant_id'], 'scene': job['scene'],
                                 'experiment': display(job['experiment']),
                                 'gpu_slot': slot, 'returncode': rc})
            del running[slot]
        for slot in args.gpus:
            if queue and slot not in running:
                launch(slot, queue.pop(0))
        if queue or running:
            time.sleep(args.poll_seconds)

    emit('all_finished', jobs=len(jobs), completed=len(completed),
         skipped=len(skipped), failures=failures)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
