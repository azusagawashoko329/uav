#!/usr/bin/env python3
"""Detached two-physical-GPU dynamic queue for LGGD without LAGA."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.da_laga_scene import SCENES, checked_config, now, save_json, sha


def read(path):
    return json.loads(path.read_text())


def register(exp, paths, rid, producer):
    payload = read(exp/'artifacts.json')
    items = {x['path']:x for x in payload['artifacts']}
    for path in paths:
        if not path.exists():
            continue
        rel = str(path.relative_to(exp))
        items[rel] = {'path':rel,'producer_script':producer,'run_id':rid,'status':'present',
                      'inputs':['config.json'],'size_bytes':path.stat().st_size,'sha256':sha(path),
                      'storage':'local ignored; no external backup' if path.suffix=='.pth' else 'git'}
    payload['artifacts'] = list(items.values())
    save_json(exp/'artifacts.json',payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    manifest = read(args.manifest)
    owner = args.manifest.resolve().parent
    expdirs = [ROOT/x for x in manifest['experiments']]
    rid = manifest['run_id']
    audit_path = owner/'preflight.json'
    audit = read(audit_path)
    if audit['status'] != 'passed':
        raise ValueError('preflight not passed')
    for rel,digest in audit['code_hashes'].items():
        if sha(ROOT/rel) != digest:
            raise ValueError(f'pinned code/config changed: {rel}')
    inventory = {}
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
    for line in raw.strip().splitlines():
        index,uuid,name = [x.strip() for x in line.split(',')]
        inventory[int(index)] = {'uuid':uuid,'name':name}
    slots = manifest['gpus']
    if len(slots)!=2 or len(set(slots))!=2 or any(g not in inventory for g in slots):
        raise ValueError('requires two distinct physical GPUs')
    jobs,skipped = [],[]
    for scene in sorted(SCENES,key=lambda s:-audit['data'][s]['train_frames']):
        for exp in expdirs:
            cfg = checked_config(exp)
            job = {'experiment':str(exp.relative_to(ROOT)),'scene':scene,'seed':cfg['seed']}
            metric = exp/'results'/f'{scene}.json'
            if metric.exists():
                m = read(metric)
                if (m['seed']!=cfg['seed'] or m['epochs']!=30 or
                        sha(exp/m['checkpoint'])!=m['checkpoint_sha256'] or m['guard']['match']['alpha']!=0):
                    raise ValueError(f'invalid completed scene: {exp}/{scene}')
                skipped.append(job)
            else:
                jobs.append(job)
    if args.dry_run:
        print(json.dumps({'new_scene_jobs':len(jobs),'skipped':len(skipped),'gpus':inventory,
                          'queue':jobs,'policy':'one task per configured GPU, no occupancy gate'}))
        return 0
    lock = (owner/'.laga_scheduler.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.resume:
        source = manifest['source_commit']
        if not source:
            raise ValueError('resume needs original pinned source commit')
    else:
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
            raise ValueError('initial launch requires committed clean working tree')
        source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        manifest['source_commit'] = source
        save_json(args.manifest,manifest)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    (owner/'.laga_scheduler.pid').write_text(str(os.getpid())+'\n')
    event_paths = [exp/'logs'/f'scheduler_{rid}.jsonl' for exp in expdirs]

    def emit(kind,**fields):
        event = {'time':now(),'run_id':rid,'kind':kind,**fields}
        line = json.dumps(event,ensure_ascii=False)+'\n'
        for path in event_paths:
            with path.open('a') as f:
                f.write(line)
        print(line,end='',flush=True)

    emit('start',pid=os.getpid(),sid=os.getsid(0),source_commit=source,new_jobs=len(jobs),
         skipped_jobs=len(skipped),gpus=inventory,policy='force two physical GPUs; immediate refill; ignore occupancy')
    for exp in expdirs:
        cfg,record = checked_config(exp),read(exp/'record.json')
        if not record['runs'] or record['runs'][-1]['run_id']!=rid:
            record['runs'].append({'run_id':rid,'source_commit':source,'seed':cfg['seed'],'status':'running',
                                    'started_at':now(),'ended_at':None,'scene_jobs':[],
                                    'command':[sys.executable,'scripts/da_laga_scheduler.py','--manifest',str(args.manifest)],
                                    'log':f'logs/scheduler_{rid}.jsonl','data_root':cfg['training']['data_root']})
        record.update(status='running',source_commit=source)
        save_json(exp/'record.json',record)
    running,failures,completed = {},[],0

    def launch(gpu,job):
        exp = ROOT/job['experiment']
        path = exp/'logs'/f'{job["scene"]}.log'
        if path.exists() and not args.resume:
            raise FileExistsError(path)
        command = [sys.executable,str(exp/'scripts/run_scene.py'),'--scene',job['scene'],
                   '--run-id',rid,'--source-commit',source,'--preflight',str(audit_path)]
        if args.resume:
            command.append('--resume')
        stream = path.open('a')
        env = {**os.environ,'CUDA_VISIBLE_DEVICES':inventory[gpu]['uuid'],
               'CUDA_DEVICE_ORDER':'PCI_BUS_ID','PYTHONUNBUFFERED':'1'}
        process = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                   stdout=stream,stderr=subprocess.STDOUT)
        running[gpu] = {'process':process,'stream':stream,'job':job,'log':path,
                        'started_at':now(),'monotonic':time.monotonic()}
        emit('launch',**job,gpu=gpu,uuid=inventory[gpu]['uuid'],pid=process.pid,
             log=str(path.relative_to(ROOT)),command=command,remaining=len(jobs))

    def update(state,code):
        job = state['job']
        exp,scene = ROOT/job['experiment'],job['scene']
        record = read(exp/'record.json')
        entry = {'scene':scene,'status':'completed' if code==0 else 'failed','returncode':code,
                 'started_at':state['started_at'],'ended_at':now(),'log':str(state['log'].relative_to(exp))}
        if code==0:
            metric = read(exp/'results'/f'{scene}.json')
            entry.update(checkpoint_sha256=metric['checkpoint_sha256'],split_fingerprint=metric['split_fingerprint'])
        record['runs'][-1]['scene_jobs'].append(entry)
        scene_files = [exp/'results'/f'{s}.json' for s in SCENES]
        if all(p.exists() for p in scene_files):
            metrics = {s:read(exp/'results'/f'{s}.json') for s in SCENES}
            summary = {'variant':'lggd_no_laga','seed':job['seed'],'baseline':checked_config(exp)['baseline'],
                       'macro_auc_percent':sum(m['auc_percent'] for m in metrics.values())/7,
                       'macro_eer':sum(m['eer'] for m in metrics.values())/7,'scenes':metrics,
                       'match':'ablation_match alpha0 in both training and test'}
            save_json(exp/'results/summary.json',summary)
            (exp/'results/summary.md').write_text(f'# HAP-RD w/o LAGA / seed{job["seed"]}\n\n'
                f'Seven DA scenes completed. Matched macro AUC: {summary["macro_auc_percent"]:.4f}%; '
                f'EER: {summary["macro_eer"]:.6f}.\n')
            record.update(status='completed',result_summary='results/summary.md')
            record['runs'][-1].update(status='completed',ended_at=now())
        elif code!=0:
            record['status']='failed'
            record['runs'][-1]['status']='failed'
        save_json(exp/'record.json',record)
        paths = [state['log'],exp/'results'/f'{scene}.json',exp/'results'/f'{scene}_scores.csv',
                 exp/'results'/f'{scene}_status.json',exp/'ckpt'/f'{scene}_final.pth',
                 exp/'ckpt'/f'{scene}_resume.pth']
        register(exp,paths,rid,'scripts/da_laga_scene.py')
        if record['status']=='completed':
            register(exp,[exp/'results/summary.json',exp/'results/summary.md'],rid,'scripts/da_laga_scheduler.py')

    for gpu in slots:
        if jobs:
            launch(gpu,jobs.pop(0))
    while running or jobs:
        for gpu,state in list(running.items()):
            code = state['process'].poll()
            if code is None:
                continue
            state['stream'].close()
            emit('finish',**state['job'],gpu=gpu,pid=state['process'].pid,returncode=code,
                 elapsed_s=time.monotonic()-state['monotonic'])
            del running[gpu]
            # Refill before hashing checkpoints, keeping the physical GPU occupied.
            if jobs:
                launch(gpu,jobs.pop(0))
            if code==0:
                completed+=1
            else:
                failures.append({**state['job'],'returncode':code})
            update(state,code)
        if running:
            time.sleep(2)
    emit('all_finished',completed=completed,skipped=len(skipped),failures=failures)
    for exp,path in zip(expdirs,event_paths):
        register(exp,[path],rid,'scripts/da_laga_scheduler.py')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
