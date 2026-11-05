#!/usr/bin/env python3
"""Dynamic two-GPU scheduler for UIT matched-score re-evaluation."""
from __future__ import annotations

import argparse, datetime as dt, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = '/home/lyx/miniconda3/envs/DMAD/bin/python'


def now():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def inventory(indices):
    raw = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,name,memory.used',
         '--format=csv,noheader,nounits'], text=True)
    out = {}
    for line in raw.strip().splitlines():
        i, u, n, m = [x.strip() for x in line.split(',')]
        if int(i) in indices:
            out[int(i)] = {'uuid': u, 'name': n, 'used_mb': int(m)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiments', nargs='+', required=True)
    p.add_argument('--run-id', default=None)
    p.add_argument('--python', default=DEFAULT_PYTHON)
    p.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--poll-seconds', type=float, default=10.0)
    args = p.parse_args()
    exps = [(ROOT / e).resolve() if not Path(e).is_absolute() else Path(e).resolve()
            for e in args.experiments]
    run_id = args.run_id or (now() + '_uit_matched_reeval')
    for exp in exps:
        cfg = json.loads((exp / 'config.json').read_text())
        if cfg.get('dataset') != 'uit' or cfg.get('family') != 'appearance':
            raise ValueError(f'wrong dataset/family: {exp}')
    inv = inventory(args.gpus)
    missing = [g for g in args.gpus if g not in inv]
    if missing:
        raise RuntimeError(f'requested GPUs missing: {missing}')
    logs = {e: e / 'logs' / f'reeval_matched_{run_id}.jsonl' for e in exps}
    for log in logs.values():
        log.parent.mkdir(parents=True, exist_ok=True)

    def emit(kind, **kw):
        ev = {'time': now(), 'run_id': run_id, 'kind': kind, **kw}
        line = json.dumps(ev, ensure_ascii=False) + '\n'
        for log in logs.values():
            with log.open('a') as f:
                f.write(line)
        print(line, end='', flush=True)

    queue = [e for e in exps if not (e / 'results/uit_matched.json').is_file()]
    running = {}
    failures = []
    emit('start', jobs=len(exps), queued=len(queue), skipped=len(exps) - len(queue),
         gpus=[{'slot': g, **inv[g]} for g in args.gpus], run_id=run_id)

    def launch(slot, exp):
        stamp = now()
        log_path = exp / 'logs' / f'reeval_matched_{stamp}_gpu{slot}.log'
        cmd = [args.python, str(ROOT / 'scripts/uit_matched_reeval.py'),
               '--experiment', str(exp), '--gpu', '0', '--batch', str(args.batch)]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': inv[slot]['uuid'],
               'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'PYTHONUNBUFFERED': '1'}
        f = log_path.open('w')
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        running[slot] = {'exp': exp, 'proc': proc, 'file': f,
                         'log': log_path, 't0': time.monotonic()}
        emit('launch', experiment=str(exp.relative_to(ROOT)), gpu_slot=slot,
             uuid=inv[slot]['uuid'], pid=proc.pid, log=str(log_path), remaining=len(queue))

    for slot in args.gpus:
        if queue:
            launch(slot, queue.pop(0))
    while queue or running:
        for slot, state in list(running.items()):
            rc = state['proc'].poll()
            if rc is None:
                continue
            state['file'].close()
            exp = state['exp']
            emit('finish', experiment=str(exp.relative_to(ROOT)), gpu_slot=slot,
                 returncode=rc, log=str(state['log']), elapsed_s=time.monotonic() - state['t0'])
            if rc:
                failures.append({'experiment': str(exp.relative_to(ROOT)),
                                 'gpu_slot': slot, 'returncode': rc})
            del running[slot]
        for slot in args.gpus:
            if queue and slot not in running:
                launch(slot, queue.pop(0))
        if queue or running:
            time.sleep(args.poll_seconds)
    emit('all_finished', jobs=len(exps), failures=failures)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
