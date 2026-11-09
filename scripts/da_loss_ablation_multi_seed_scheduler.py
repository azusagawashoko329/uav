#!/usr/bin/env python3
"""Detached-friendly dynamic two-GPU scheduler for DA Loss seed42/43/44 cells."""
from __future__ import annotations
import argparse, datetime as dt, glob, json, os, subprocess, sys, time
from pathlib import Path
SCENES=['bike','crossroads','farmland','highway','railway','solar','vehicle']
DEFAULT_PYTHON='/home/lyx/miniconda3/envs/DMAD/bin/python'
def now(): return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def inventory(indices):
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total','--format=csv,noheader,nounits'],text=True); out={}
 for line in raw.strip().splitlines():
  idx,uuid,name,used,total=[x.strip() for x in line.split(',')]
  if int(idx) in indices: out[int(idx)]={'uuid':uuid,'name':name,'used_mb':int(used),'total_mb':int(total)}
 return out
def scene_costs(root,scenes):
 names={'bike':'Bike Roundabout','crossroads':'Crossroads','farmland':'Farmland Inspection','highway':'Highway','railway':'Railway Inspection','solar':'Solar Panel Inspection','vehicle':'Vehicle Roundabout'}; out={}
 for s in scenes:
  paths=glob.glob(str(root/names[s]/'sequence*'/'train'/'*'/'*.jpg'))
  if not paths: raise FileNotFoundError(f'no train frames for {s}')
  out[s]=len(paths)
 return out
def main():
 p=argparse.ArgumentParser(); p.add_argument('--repo-root',type=Path,required=True); p.add_argument('--experiments',nargs='+',required=True); p.add_argument('--run-id',default=None); p.add_argument('--python',default=DEFAULT_PYTHON); p.add_argument('--gpus',nargs='+',type=int,default=[0,1]); p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch',type=int,default=64); p.add_argument('--workers',type=int,default=8); p.add_argument('--scenes',nargs='+',choices=SCENES,default=SCENES); p.add_argument('--max-steps',type=int,default=0); p.add_argument('--poll-seconds',type=float,default=5); p.add_argument('--max-memory-mb',type=int,default=2000); p.add_argument('--resume',action='store_true'); p.add_argument('--dry-run',action='store_true'); a=p.parse_args()
 root=a.repo_root.resolve(); exps=[(root/e).resolve() if not e.is_absolute() else e.resolve() for e in map(Path,a.experiments)]; run_id=a.run_id or now()+'_da_loss_multiseed'
 cfgs={}
 for e in exps:
  cfg=json.loads((e/'config.json').read_text()); cfgs[e]=cfg
  if cfg.get('dataset')!='da' or cfg.get('family')!='appearance' or int(cfg.get('seed')) not in (42,43,44): raise ValueError(f'config mismatch or unsupported seed: {e}')
  if int(cfg.get('training',{}).get('epochs'))!=a.epochs or int(cfg.get('training',{}).get('batch'))!=a.batch: raise ValueError(f'training mismatch: {e}')
 data_root=Path(cfgs[exps[0]]['training']['data_root']); costs=scene_costs(data_root,a.scenes); order=sorted(a.scenes,key=lambda s:(-costs[s],SCENES.index(s)))
 jobs=[{'experiment':e,'scene':s,'experiment_id':cfgs[e]['id'],'seed':int(cfgs[e]['seed'])} for s in order for e in exps]
 inv=inventory(a.gpus); missing=[g for g in a.gpus if g not in inv]
 if missing: raise RuntimeError(f'missing GPUs: {missing}')
 busy={g:inv[g]['used_mb'] for g in a.gpus if inv[g]['used_mb']>a.max_memory_mb}
 if busy: raise RuntimeError(f'GPUs busy above threshold: {busy}')
 def display(e):
  try:return str(e.relative_to(root))
  except ValueError:return str(e)
 logs={e:e/'logs'/f'scheduler_{run_id}.jsonl' for e in exps}
 for l in logs.values(): l.parent.mkdir(parents=True,exist_ok=True)
 def emit(kind,**kw):
  line=json.dumps({'time':now(),'run_id':run_id,'kind':kind,**kw},ensure_ascii=False)+'\n'; print(line,end='',flush=True)
  for l in logs.values(): l.open('a').write(line)
 if a.dry_run:
  print(json.dumps({'kind':'dry_run_ok','run_id':run_id,'jobs':len(jobs),'seeds':sorted({j['seed'] for j in jobs}),'gpus':[{'slot':g,**inv[g]} for g in a.gpus],'scene_order':order,'scene_train_frames':costs,'job_list':[{'experiment':display(j['experiment']),'scene':j['scene'],'seed':j['seed']} for j in jobs]},ensure_ascii=False)); return 0
 queue=[]; skipped=[]
 for j in jobs:
  e,s=j['experiment'],j['scene']; done=(e/'results'/f'{s}.json').is_file() and (e/'ckpt'/f'{s}_final.pth').is_file()
  (skipped if a.resume and done else queue).append(j)
 emit('start',experiments=[display(e) for e in exps],gpus=[{'slot':g,**inv[g]} for g in a.gpus],jobs=len(jobs),epochs=a.epochs,batch=a.batch,workers=a.workers,seeds=sorted({j['seed'] for j in jobs}),scene_order=order,scene_train_frames=costs,skipped=len(skipped),policy='dynamic per-GPU queue; mixed seed42/43/44')
 if skipped: emit('resume_skip',jobs=[{'experiment':display(j['experiment']),'scene':j['scene'],'seed':j['seed']} for j in skipped])
 running={}; failures=[]; completed=[]
 def launch(slot,j):
  e,s,seed=j['experiment'],j['scene'],j['seed']; stamp=now(); log=e/'logs'/f'{stamp}_{s}_gpu{slot}.log'; cmd=[a.python,str(root/'scripts/da_loss_ablation_runner.py'),'--experiment',str(e),'--scene',s,'--gpu','0','--seed',str(seed),'--epochs',str(a.epochs),'--batch',str(a.batch),'--workers',str(a.workers)]
  if a.max_steps: cmd+=['--max-steps',str(a.max_steps)]
  if a.resume: cmd.append('--overwrite')
  env={**os.environ,'CUDA_VISIBLE_DEVICES':inv[slot]['uuid'],'CUDA_DEVICE_ORDER':'PCI_BUS_ID','PYTHONUNBUFFERED':'1'}; f=log.open('w'); proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL); running[slot]={'job':j,'proc':proc,'file':f,'path':log,'t0':time.monotonic()}; emit('launch',experiment=display(e),experiment_id=j['experiment_id'],scene=s,seed=seed,gpu_slot=slot,uuid=inv[slot]['uuid'],name=inv[slot]['name'],pid=proc.pid,log=str(log),command=cmd,remaining_after_launch=len(queue))
 for slot in a.gpus:
  if queue: launch(slot,queue.pop(0))
 while queue or running:
  for slot,state in list(running.items()):
   rc=state['proc'].poll()
   if rc is None: continue
   state['file'].close(); j=state['job']; emit('finish',experiment=display(j['experiment']),experiment_id=j['experiment_id'],scene=j['scene'],seed=j['seed'],gpu_slot=slot,pid=state['proc'].pid,returncode=rc,log=str(state['path']),elapsed_s=time.monotonic()-state['t0']); (completed if rc==0 else failures).append(j); del running[slot]
  for slot in a.gpus:
   if queue and slot not in running: launch(slot,queue.pop(0))
  if queue or running: time.sleep(a.poll_seconds)
 emit('all_finished',jobs=len(jobs),completed=len(completed),skipped=len(skipped),failures=failures); return 1 if failures else 0
if __name__=='__main__': sys.exit(main())
