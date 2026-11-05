#!/usr/bin/env python3
"""Dynamic two-GPU scheduler for DA matched-score re-evaluation.

Runs scripts/da_variant_reeval.py for each (experiment, scene) checkpoint.
Only test inference is run; checkpoints are loaded read-only.  Jobs are
queued per GPU and pinned by UUID through CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.da_variant_runner import SCENES  # noqa: E402

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiments', nargs='+', required=True)
    p.add_argument('--run-id', default=None)
    p.add_argument('--python', default=DEFAULT_PYTHON)
    p.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    p.add_argument('--scenes', nargs='+', choices=SCENES, default=SCENES)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--poll-seconds', type=float, default=5.0)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    experiments = [(ROOT / e).resolve() if not Path(e).is_absolute() else Path(e).resolve()
                   for e in args.experiments]
    run_id = args.run_id or (now() + '_da_matched_reeval')
    for exp in experiments:
        cfg = json.loads((exp / 'config.json').read_text())
        if cfg.get('dataset') != 'da' or cfg.get('family') != 'appearance':
            raise ValueError(f'wrong dataset/family: {exp}')
        if not (exp / 'ckpt').is_dir():
            raise FileNotFoundError(f'missing ckpt dir: {exp}')
    jobs = [{'experiment': exp, 'scene': scene} for scene in SCENES for exp in experiments]
    if args.dry_run:
        print(json.dumps({'kind': 'dry_run_ok', 'jobs': len(jobs), 'gpus': args.gpus,
                          'run_id': run_id}, ensure_ascii=False))
        return 0
    inv = inventory(args.gpus)
    missing = [g for g in args.gpus if g not in inv]
    if missing:
        raise RuntimeError(f'requested GPUs missing: {missing}')
    logs = {e: e / 'logs' / f'reeval_{run_id}.jsonl' for e in experiments}
    for log in logs.values():
        log.parent.mkdir(parents=True, exist_ok=True)

    def emit(kind, **kw):
        ev = {'time': now(), 'run_id': run_id, 'kind': kind, **kw}
        line = json.dumps(ev, ensure_ascii=False) + '\n'
        for log in logs.values():
            with log.open('a') as f:
                f.write(line)
        print(line, end='', flush=True)

    queue = [j for j in jobs
             if not (j['experiment'] / 'results_matched' / f'{j["scene"]}.json').is_file()]
    running = {}
    failures = []
    emit('start', jobs=len(jobs), queued=len(queue), skipped=len(jobs) - len(queue),
         gpus=[{'slot': g, **inv[g]} for g in args.gpus], run_id=run_id,
         policy='dynamic per-GPU queue; matched_score test inference only')

    def launch(slot, job):
        exp, scene = job['experiment'], job['scene']
        stamp = now()
        log_path = exp / 'logs' / f'reeval_{stamp}_{scene}_gpu{slot}.log'
        cmd = [args.python, str(ROOT / 'scripts/da_variant_reeval.py'),
               '--experiment', str(exp), '--scene', scene, '--gpu', '0',
               '--batch', str(args.batch)]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': inv[slot]['uuid'],
               'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'PYTHONUNBUFFERED': '1'}
        f = log_path.open('w')
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        running[slot] = {'job': job, 'proc': proc, 'file': f,
                         'log': log_path, 't0': time.monotonic()}
        emit('launch', experiment=str(exp.relative_to(ROOT)), scene=scene,
             gpu_slot=slot, uuid=inv[slot]['uuid'], pid=proc.pid,
             log=str(log_path), remaining=len(queue))

    for slot in args.gpus:
        if queue:
            launch(slot, queue.pop(0))
    while queue or running:
        for slot, state in list(running.items()):
            rc = state['proc'].poll()
            if rc is None:
                continue
            state['file'].close()
            job = state['job']
            emit('finish', experiment=str(job['experiment'].relative_to(ROOT)),
                 scene=job['scene'], gpu_slot=slot, returncode=rc,
                 log=str(state['log']), elapsed_s=time.monotonic() - state['t0'])
            if rc:
                failures.append({'experiment': str(job['experiment'].relative_to(ROOT)),
                                 'scene': job['scene'], 'gpu_slot': slot, 'returncode': rc})
            del running[slot]
        for slot in args.gpus:
            if queue and slot not in running:
                launch(slot, queue.pop(0))
        if queue or running:
            time.sleep(args.poll_seconds)
    emit('all_finished', jobs=len(jobs), failures=failures)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
