#!/usr/bin/env python3
"""Detached two-GPU queue for the DA publication chain; no occupancy gate.

The manifest is written before launch, all tasks run the experiment's own wrapper.
One global queue keeps one scene process on each configured physical GPU.
"""
from __future__ import annotations
import argparse, datetime as dt, fcntl, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.da_publication_scene import SCENES, save_json, sha

def now():return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def event_json(p):return json.loads(p.read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
    manifest=event_json(a.manifest);base=(ROOT/manifest['base']).resolve();rid=manifest['run_id']
    expdirs=[(ROOT/x).resolve() for x in manifest['experiments']]
    inventory={}
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
    for line in raw.strip().splitlines():
        g,u,n=[x.strip() for x in line.split(',')];inventory[int(g)]={'uuid':u,'name':n}
    slots=manifest['gpus']
    if len(slots)!=2 or len(set(slots))!=2 or any(g not in inventory for g in slots):raise ValueError('requires two distinct GPUs')
    preflight=base/'publication_preflight.json';audit=event_json(preflight)
    if audit['status']!='passed':raise RuntimeError('preflight did not pass')
    for rel,h in audit['code_hashes'].items():
        if sha(ROOT/rel)!=h:raise RuntimeError(f'code changed after preflight: {rel}')
    jobs=[];skipped=[];costs=manifest['scene_train_frames']
    for scene in sorted(SCENES,key=lambda s:-costs[s]):
        for e in expdirs:
            c=event_json(e/'config.json');m=e/'results'/f'{scene}.json';ck=e/'ckpt'/f'{scene}_final.pth'
            job={'experiment':str(e.relative_to(ROOT)),'scene':scene,'variant':c['variant'],'seed':c['seed']}
            if m.exists():
                metric=event_json(m)
                if metric['seed']!=c['seed'] or metric.get('epochs')!=30 or not ck.exists() or sha(ck)!=metric['checkpoint_sha256']:
                    raise RuntimeError(f'invalid existing completion {e}/{scene}')
                skipped.append(job)
            else:jobs.append(job)
    if a.dry_run:
        print(json.dumps({'total_jobs':len(jobs)+len(skipped),'new_jobs':len(jobs),'skipped':len(skipped),'gpus':inventory,'policy':'force configured GPUs; no occupancy check'}));return
    lock=(base/'.publication_scheduler.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not a.resume and subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('working tree must be clean before initial launch')
    manifest['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save_json(a.manifest,manifest)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    (base/'.publication_scheduler.pid').write_text(str(os.getpid())+'\n')
    logs=[base/f'{rid}.scheduler.jsonl']+[e/'logs'/f'scheduler_{rid}.jsonl' for e in expdirs]
    def emit(kind,**kw):
        item={'time':now(),'run_id':rid,'kind':kind,**kw};line=json.dumps(item,ensure_ascii=False)+'\n'
        for p in logs:
            p.parent.mkdir(exist_ok=True,parents=True)
            with p.open('a') as f:f.write(line)
        print(line,end='',flush=True)
    emit('start',pid=os.getpid(),sid=os.getsid(0),source_commit=manifest['source_commit'],
         new_jobs=len(jobs),skipped_jobs=len(skipped),gpus=inventory,policy='force two GPUs, immediate refill; ignore occupancy')
    emit('reuse_skip',jobs=skipped)
    failures=[];running={};completed=0
    for e in expdirs:
        c=event_json(e/'config.json')
        if c.get('reused_from'):continue
        r=event_json(e/'record.json')
        if not r['runs'] or r['runs'][-1]['run_id']!=rid:
            r['runs'].append({'run_id':rid,'source_commit':manifest['source_commit'],'status':'running',
                'seed':c['seed'],'script':'scripts/da_publication_scheduler.py','command':[sys.executable,'scripts/da_publication_scheduler.py','--manifest',str(a.manifest)],
                'started_at':now(),'ended_at':None,'log':f'logs/scheduler_{rid}.jsonl',
                'data_root':c['training']['data_root'],'scene_jobs':[],'environment':{'gpus':inventory,'torch':audit['torch']}})
        r['status']='running';r['source_commit']=manifest['source_commit'];save_json(e/'record.json',r)
    def launch(g,j):
        e=ROOT/j['experiment'];log=e/'logs'/f'{j["scene"]}.log'
        if log.exists() and not a.resume:raise FileExistsError(f'log exists without --resume: {log}')
        cmd=[sys.executable,str(e/'scripts/run_scene.py'),'--scene',j['scene'],'--run-id',rid,
             '--source-commit',manifest['source_commit'],'--source-hashes',str(preflight)]
        if a.resume:cmd+=['--resume']
        env={**os.environ,'CUDA_VISIBLE_DEVICES':inventory[g]['uuid'],'CUDA_DEVICE_ORDER':'PCI_BUS_ID','PYTHONUNBUFFERED':'1'}
        f=log.open('a');p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        running[g]={'process':p,'file':f,'job':j,'log':str(log.relative_to(ROOT)),'start':now(),'t0':time.monotonic()}
        emit('launch',**j,gpu=g,uuid=inventory[g]['uuid'],pid=p.pid,log=str(log.relative_to(ROOT)),command=cmd,remaining=len(jobs))
    def update_scene(state,rc):
        j=state['job'];e=ROOT/j['experiment'];r=event_json(e/'record.json');entry={'scene':j['scene'],'status':'completed' if rc==0 else 'failed',
            'started_at':state['start'],'ended_at':now(),'log':state['log'],'returncode':rc}
        if rc==0:
            m=event_json(e/'results'/f'{j["scene"]}.json');entry['split_fingerprint']=m['split_fingerprint']
            r['runs'][-1]['environment']=m['environment'];entry['checkpoint_sha256']=m['checkpoint_sha256']
        r['runs'][-1]['scene_jobs'].append(entry)
        done=all((e/'results'/f'{s}.json').exists() for s in SCENES)
        if done:
            metrics={s:event_json(e/'results'/f'{s}.json') for s in SCENES}
            summary={'variant':j['variant'],'seed':j['seed'],'scenes':metrics,
                     'macro_auc_percent':sum(m['auc_percent'] for m in metrics.values())/7,
                     'macro_eer':sum(m['eer'] for m in metrics.values())/7,'match':'publication_match'}
            save_json(e/'results/summary.json',summary)
            (e/'results/summary.md').write_text(f'# {e.name}\n\nSeven DA scenes completed. Matched AUC: {summary["macro_auc_percent"]:.4f}%; EER: {summary["macro_eer"]:.6f}.\n')
            r['status']='completed';r['result_summary']='results/summary.md';r['runs'][-1]['status']='completed';r['runs'][-1]['ended_at']=now()
        elif rc!=0:r['status']='failed';r['runs'][-1]['status']='failed'
        save_json(e/'record.json',r)
        arts=event_json(e/'artifacts.json');items={x['path']:x for x in arts['artifacts']}
        paths=[e/'results'/f'{j["scene"]}.json',e/'results'/f'{j["scene"]}_scores.csv',e/'results'/f'{j["scene"]}_status.json',
               ROOT/state['log'],e/'ckpt'/f'{j["scene"]}_final.pth',e/'ckpt'/f'{j["scene"]}_resume.pth']
        if done:paths += [e/'results/summary.json',e/'results/summary.md']
        for p in paths:
            if not p.exists():continue
            rel=str(p.relative_to(e));items[rel]={'path':rel,'producer_script':'scripts/da_publication_scene.py','run_id':rid,
                'status':'present','size_bytes':p.stat().st_size,'sha256':sha(p),'inputs':['config.json'],
                'storage':'local ignored; no external backup registered' if 'ckpt/' in rel else 'git'}
        arts['artifacts']=list(items.values());save_json(e/'artifacts.json',arts)
    for g in slots:
        if jobs:launch(g,jobs.pop(0))
    while running or jobs:
        for g,state in list(running.items()):
            rc=state['process'].poll()
            if rc is None:continue
            state['file'].close();emit('finish',**state['job'],gpu=g,pid=state['process'].pid,returncode=rc,elapsed_s=time.monotonic()-state['t0'])
            # Refill the free GPU before hashing and registering completed artifacts.
            del running[g]
            if jobs:launch(g,jobs.pop(0))
            if rc==0:completed+=1
            else:failures.append({**state['job'],'returncode':rc})
            update_scene(state,rc)
        if running or jobs:time.sleep(2)
    emit('all_finished',completed=completed,skipped=len(skipped),failures=failures)
    for e in expdirs:
        p=e/'logs'/f'scheduler_{rid}.jsonl';arts=event_json(e/'artifacts.json')
        arts['artifacts']=[x for x in arts['artifacts'] if x['path']!=str(p.relative_to(e))]+[{'path':str(p.relative_to(e)),
            'producer_script':'scripts/da_publication_scheduler.py','run_id':rid,'status':'present','size_bytes':p.stat().st_size,'sha256':sha(p),'storage':'git','inputs':[str(a.manifest)]}]
        save_json(e/'artifacts.json',arts)
    return 1 if failures else 0
if __name__=='__main__':sys.exit(main())
