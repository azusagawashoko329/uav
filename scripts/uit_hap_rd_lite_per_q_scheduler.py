#!/usr/bin/env python3
"""Detached dynamic queue for three whole-UIT HAP-RD-Lite-Per-Q models."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import json

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.uit_hap_rd_lite_per_q_runner import checked_config,read,save_json,sha,now


def register(exp,paths,rid,producer):
    payload=read(exp/'artifacts.json');items={x['path']:x for x in payload['artifacts']}
    for path in paths:
        if not path.is_file():continue
        rel=str(path.relative_to(exp))
        items[rel]={'path':rel,'producer_script':producer,'run_id':rid,'inputs':['config.json'],
                    'status':'present','size_bytes':path.stat().st_size,'sha256':sha(path),
                    'storage':'local ignored; no external backup' if path.suffix=='.pth' else 'git'}
    payload['artifacts']=list(items.values());save_json(exp/'artifacts.json',payload)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();manifest=read(args.manifest);owner=args.manifest.resolve().parent
    rid=manifest['run_id'];preflight=owner/'preflight.json';audit=read(preflight)
    if audit['status']!='passed':raise ValueError('preflight failed')
    for rel,h in audit['code_hashes'].items():
        if sha(ROOT/rel)!=h:raise ValueError(f'pinned source/config changed: {rel}')
    inventory={}
    for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True).strip().splitlines():
        g,uuid,name=[x.strip() for x in line.split(',')];inventory[int(g)]={'uuid':uuid,'name':name}
    slots=manifest['gpus']
    if len(slots)!=2 or len(set(slots))!=2 or any(g not in inventory for g in slots):raise ValueError('exactly two physical GPUs required')
    expdirs=[ROOT/rel for rel in manifest['experiments']];jobs=[];skipped=[]
    for exp in expdirs:
        cfg=checked_config(exp);result=exp/'results/uit.json'
        job={'experiment':str(exp.relative_to(ROOT)),'variant':cfg['variant'],'seed':cfg['seed']}
        if result.exists():
            m=read(result)
            if m['variant']!=cfg['variant'] or m['seed']!=cfg['seed'] or m['epochs']!=45 or sha(exp/m['checkpoint'])!=m['checkpoint_sha256']:
                raise ValueError(f'invalid existing result: {exp}')
            skipped.append(job)
        else:jobs.append(job)
    if args.dry_run:
        print(json.dumps({'jobs':jobs,'new_jobs':len(jobs),'skipped':len(skipped),'gpus':inventory,'policy':'force two physical GPUs; no occupancy gate'}));return 0
    lock=(owner/'.uit_hap_rd_lite_per_q.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.resume:
        source=manifest['source_commit']
        if not source:raise ValueError('resume must preserve original source')
    else:
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('initial launch requires clean committed HEAD')
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        manifest['source_commit']=source;save_json(args.manifest,manifest)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    (owner/'.uit_hap_rd_lite_per_q.pid').write_text(str(os.getpid())+'\n')
    event_paths=[e/'logs'/f'scheduler_{rid}.jsonl' for e in expdirs]

    def emit(kind,**fields):
        line=json.dumps({'time':now(),'run_id':rid,'kind':kind,**fields})+'\n'
        for p in event_paths:
            with p.open('a') as f:f.write(line)
        print(line,end='',flush=True)

    emit('start',pid=os.getpid(),sid=os.getsid(0),source_commit=source,new_jobs=len(jobs),skipped=len(skipped),
         gpus=inventory,policy='force two GPUs; immediate refill; ignore occupancy')
    for exp in expdirs:
        cfg=checked_config(exp);record=read(exp/'record.json')
        if (exp/'results/uit.json').exists():continue
        if not record['runs'] or record['runs'][-1]['run_id']!=rid:
            record['runs'].append({'run_id':rid,'source_commit':source,'seed':cfg['seed'],'status':'running',
              'started_at':now(),'ended_at':None,'command':[sys.executable,'scripts/uit_hap_rd_lite_per_q_scheduler.py','--manifest',str(args.manifest)],
              'log':f'logs/scheduler_{rid}.jsonl','data_root':cfg['training']['data_root'],
              'split_fingerprint':audit['data']['split_fingerprint'],'environment':{'torch':audit['torch'],'gpus':inventory}})
        record.update(status='running',source_commit=source);save_json(exp/'record.json',record)
    running={};completed=0;failures=[]

    def launch(gpu,job):
        exp=ROOT/job['experiment'];path=exp/'logs/run.log'
        if path.exists() and not args.resume:raise FileExistsError(path)
        command=[sys.executable,str(exp/'scripts/run_uit.py'),'--run-id',rid,'--source-commit',source,'--preflight',str(preflight)]
        if args.resume:command.append('--resume')
        stream=path.open('a');env={**os.environ,'CUDA_VISIBLE_DEVICES':inventory[gpu]['uuid'],
                                  'CUDA_DEVICE_ORDER':'PCI_BUS_ID','PYTHONUNBUFFERED':'1'}
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        running[gpu]={'process':process,'stream':stream,'job':job,'path':path,'start':now(),'t0':time.monotonic()}
        emit('launch',**job,gpu=gpu,uuid=inventory[gpu]['uuid'],pid=process.pid,command=command,
             log=str(path.relative_to(ROOT)),remaining=len(jobs))

    def update(state,code):
        exp=ROOT/state['job']['experiment'];record=read(exp/'record.json');run=record['runs'][-1]
        if code==0:
            m=read(exp/'results/uit.json')
            if m['guard']['block_order']!='LLLG' or m['source_commit']!=source or m['epochs']!=45:raise ValueError('completed run guard mismatch')
            summary={'dataset':'UIT-ADrone','variant':m['variant'],'seed':m['seed'],'auc_percent':m['auc_percent'],
                     'eer':m['eer'],'frames':m['frames'],'match':m['match'],'guard':m['guard']}
            save_json(exp/'results/summary.json',summary)
            (exp/'results/summary.md').write_text(f'# {m["guard"]["label"]} / UIT / seed{m["seed"]}\n\n'
                f'LLLG+GATE, 45 epochs completed. Training-matched AUC: {m["auc_percent"]:.4f}%; EER: {m["eer"]:.6f}.\n')
            record.update(status='completed',result_summary='results/summary.md')
            run.update(status='completed',ended_at=now(),checkpoint_sha256=m['checkpoint_sha256'])
        else:
            record['status']='failed';run.update(status='failed',ended_at=now(),returncode=code)
        run['task_started_at']=state['start'];run['task_log']='logs/run.log'
        save_json(exp/'record.json',record)
        register(exp,[state['path'],*sorted((exp/'results').glob('*')),*sorted((exp/'ckpt').glob('*.pth'))],
                 rid,'scripts/uit_hap_rd_lite_per_q_runner.py')

    for gpu in slots:
        if jobs:launch(gpu,jobs.pop(0))
    register(owner,[args.manifest.resolve()],rid,'scripts/uit_hap_rd_lite_per_q_scheduler.py')
    while jobs or running:
        for gpu,state in list(running.items()):
            code=state['process'].poll()
            if code is None:continue
            state['stream'].close();emit('finish',**state['job'],gpu=gpu,pid=state['process'].pid,returncode=code,elapsed_s=time.monotonic()-state['t0'])
            del running[gpu]
            # Start the next model before registering/hashing the completed job.
            if jobs:launch(gpu,jobs.pop(0))
            if code==0:completed+=1
            else:failures.append({**state['job'],'returncode':code})
            update(state,code)
        if running:time.sleep(2)
    emit('all_finished',completed=completed,skipped=len(skipped),failures=failures)
    for exp,path in zip(expdirs,event_paths):register(exp,[path],rid,'scripts/uit_hap_rd_lite_per_q_scheduler.py')
    return 1 if failures else 0


if __name__=='__main__':
    sys.exit(main())
