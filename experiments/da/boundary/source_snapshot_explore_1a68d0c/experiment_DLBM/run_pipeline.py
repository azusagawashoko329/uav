"""Run probe -> shared DLBM boundary learning -> freshly initialized final distillation."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_DLBM.common import ALL_SCENES, parse_scenes, write_json


def run(command, log_path):
    command = [str(part) for part in command]
    rendered = '+ ' + ' '.join(command)
    print(rendered, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write(rendered + '\n')
        log.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def train_command(args, scene, output_dir, epochs, seed, taps, max_steps):
    command = [
        sys.executable, ROOT / 'train.py', '--scene', scene, '--gpu', args.gpu,
        '--img', args.img, '--taps', *taps, '--code_dim', args.code_dim,
        '--depth', args.student_depth, '--tap_align', 'absolute', '--k', args.k,
        '--loss', args.loss, '--epochs', epochs, '--bs', args.train_bs, '--lr', args.train_lr,
        '--wd', args.train_wd, '--workers', args.workers, '--seed', seed, '--out', output_dir,
    ]
    if max_steps:
        command += ['--max_steps', max_steps]
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, help='run name under experiment_DLBM/runs/')
    parser.add_argument('--stage', choices=['all', 'probe_boundary', 'probe', 'boundary', 'final'],
                        default='all', help='probe_boundary runs stages 1+2 and deliberately skips final')
    parser.add_argument('--scenes', default=','.join(ALL_SCENES))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--final_seed', type=int, default=None,
                        help='default seed+1000; proves final is a fresh initialization')
    parser.add_argument('--probe_epochs', type=int, default=10,
                        help='all-layer normal RD warmup per scene; checkpoint is reusable by boundary experiments')
    parser.add_argument('--boundary_epochs', type=int, default=5)
    parser.add_argument('--final_epochs', type=int, default=15)
    parser.add_argument('--train_bs', type=int, default=64)
    parser.add_argument('--boundary_bs', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--img', type=int, default=256)
    parser.add_argument('--code_dim', type=int, default=64)
    parser.add_argument('--student_depth', type=int, default=12)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--loss', choices=['topk', 'full'], default='topk')
    parser.add_argument('--train_lr', type=float, default=5e-4)
    parser.add_argument('--train_wd', type=float, default=5e-2)
    parser.add_argument('--boundary_lr', type=float, default=3e-4)
    parser.add_argument('--boundary_wd', type=float, default=1e-2)
    parser.add_argument('--model_dim', type=int, default=96)
    parser.add_argument('--model_depth', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--probe_max_steps', type=int, default=0)
    parser.add_argument('--boundary_max_steps', type=int, default=0)
    parser.add_argument('--infer_steps', type=int, default=100)
    parser.add_argument('--final_max_steps', type=int, default=0)
    args = parser.parse_args()

    scenes = parse_scenes(args.scenes)
    final_seed = args.final_seed if args.final_seed is not None else args.seed + 1000
    run_dir = ROOT / 'experiment_DLBM' / 'runs' / args.exp
    probe_dir = run_dir / 'probe' / 'ckpt'
    boundary_dir = run_dir / 'boundary'
    final_dir = run_dir / 'final' / 'ckpt'
    policy_path = boundary_dir / 'policies.json'
    log_path = run_dir / 'run.log'
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / 'pipeline_config.json', {**vars(args), 'final_seed_resolved': final_seed})

    if args.stage in ('all', 'probe_boundary', 'probe'):
        for scene in scenes:
            run(train_command(
                args, scene, probe_dir, args.probe_epochs, args.seed, list(range(12)),
                args.probe_max_steps,
            ), log_path)

    if args.stage in ('all', 'probe_boundary', 'boundary'):
        run([
            sys.executable, ROOT / 'experiment_DLBM' / 'train_boundary.py',
            '--probe_dir', probe_dir, '--out', boundary_dir / 'dlbm.pth',
            '--policies', policy_path, '--scenes', ','.join(scenes), '--gpu', args.gpu,
            '--seed', args.seed, '--epochs', args.boundary_epochs, '--bs', args.boundary_bs,
            '--workers', args.workers, '--lr', args.boundary_lr, '--wd', args.boundary_wd,
            '--k', args.k, '--model_dim', args.model_dim, '--model_depth', args.model_depth,
            '--heads', args.heads, '--max_steps', args.boundary_max_steps,
            '--infer_steps', args.infer_steps,
        ], log_path)

    if args.stage in ('all', 'final'):
        if not policy_path.exists():
            raise FileNotFoundError(f'missing learned policies: {policy_path}')
        with policy_path.open(encoding='utf-8') as handle:
            policy_file = json.load(handle)
        for scene in scenes:
            policy = policy_file['scenes'].get(scene)
            if policy is None:
                raise KeyError(f'no learned policy for {scene} in {policy_path}')
            boundary = int(policy['boundary'])
            if not 0 <= boundary < 12:
                raise ValueError(f'illegal boundary for {scene}: {boundary}')
            # No checkpoint/init argument is passed. train.py constructs a new OCBE and student;
            # the distinct final seed makes accidental probe-weight reuse directly auditable.
            run(train_command(
                args, scene, final_dir, args.final_epochs, final_seed,
                list(range(boundary, 12)), args.final_max_steps,
            ), log_path)

    print(f'pipeline stage={args.stage} complete: {run_dir}', flush=True)


if __name__ == '__main__':
    main()
