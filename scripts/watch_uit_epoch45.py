#!/usr/bin/env python3
"""At epoch45, stop the old top-k UIT evaluation and preserve final weights.

V3/V4 were started before the matched-score evaluator patch.  Their post-
training evaluation would use the wrong top-k readout.  As soon as
ckpt/uit_epoch45.pth exists, terminate the runner (it has already saved the
epoch45 weights), let the scheduler move on, and copy epoch45 to the
uit_final.pth expected by the matched re-evaluation.
"""
from __future__ import annotations

import argparse, datetime as dt, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def now():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def main_pid(exp: Path, scheduler_pid: int | None) -> int | None:
    raw = subprocess.check_output(['ps', '-eo', 'pid,ppid,args'], text=True)
    for line in raw.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, args = parts
        if str(exp) not in args or 'run_scene.py' not in args:
            continue
        if scheduler_pid is not None and int(ppid) != scheduler_pid:
            continue
        return int(pid)
    return None


def wait_one(exp: Path, scheduler_pid: int | None, poll: float, log):
    ckpt45 = exp / 'ckpt' / 'uit_epoch45.pth'
    final = exp / 'ckpt' / 'uit_final.pth'
    while True:
        if ckpt45.is_file():
            pid = main_pid(exp, scheduler_pid)
            if pid is None:
                log(f'{exp.name}: epoch45 exists, runner already exited; final={"yes" if final.is_file() else "no"}')
            else:
                log(f'{exp.name}: epoch45 exists, terminating old evaluator pid={pid}')
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                for _ in range(120):
                    if main_pid(exp, scheduler_pid) is None:
                        break
                    time.sleep(1)
            if not final.is_file():
                shutil.copyfile(ckpt45, final)
                log(f'{exp.name}: created uit_final.pth from epoch45')
            return
        if scheduler_pid is not None:
            try:
                os.kill(scheduler_pid, 0)
            except ProcessLookupError:
                log(f'{exp.name}: scheduler exited before epoch45; stop watcher')
                return
        time.sleep(poll)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiments', nargs='+', required=True)
    p.add_argument('--scheduler-pid', type=int, default=None)
    p.add_argument('--poll-seconds', type=float, default=30)
    args = p.parse_args()
    exps = [Path(e).resolve() for e in args.experiments]
    logs = []
    for exp in exps:
        lp = exp / 'logs' / f'epoch45_switch_{now()}.log'
        lp.parent.mkdir(parents=True, exist_ok=True)
        logs.append((exp, lp.open('a')))

    def log(msg):
        print(msg, flush=True)
        for _, f in logs:
            f.write(msg + '\n'); f.flush()
    # Run each variant in its own process so V3 and V4 are watched concurrently.
    procs = []
    for exp in exps:
        cmd = [sys.executable, str(Path(__file__).resolve()), '--experiments', str(exp),
               '--poll-seconds', str(args.poll_seconds)]
        if args.scheduler_pid:
            cmd += ['--scheduler-pid', str(args.scheduler_pid)]
        # Avoid recursive spawning: use a hidden flag via env.
        if os.environ.get('UIT_EPOCH45_CHILD'):
            wait_one(exp, args.scheduler_pid, args.poll_seconds, log)
            return
        env = {**os.environ, 'UIT_EPOCH45_CHILD': '1'}
        procs.append(subprocess.Popen(cmd, cwd=ROOT, env=env))
    for proc in procs:
        proc.wait()
    for _, f in logs:
        f.close()


if __name__ == '__main__':
    main()
