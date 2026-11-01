#!/usr/bin/env python3
"""Dynamic UIT scheduler: queue on configured physical GPUs regardless of occupancy."""
import argparse, datetime as dt, json, os, subprocess, sys, time
from pathlib import Path

def now(): return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def inventory(indices):
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used','--format=csv,noheader,nounits'],text=True)
 out={}
 for line in raw.splitlines():
  i,u,n,m=[x.strip() for x in line.split(',')]
  if int(i) in indices: out[int(i)]={'uuid':u,'name':n,'memory_mb':int(m)}
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--variants',nargs='+',required=True);p.add_argument('--run-id',required=True);p.add_argument('--gpus',nargs='+',type=int,default=[0,1]);p.add_argument('--epochs',type=int,default=45);p.add_argument('--batch',type=int,default=64);p.add_argument('--workers',type=int,default=8);p.add_argument('--seed',type=int,default=42);p.add_argument('--poll-seconds',type=float,default=15);p.add_argument('--max-memory-mb',type=int,default=2000);args=p.parse_args();root=args.repo_root.resolve();dirs={v:root/'experiments/uit/appearance'/v for v in args.variants}
 logs=[d/'logs'/f'scheduler_{args.run_id}.jsonl' for d in dirs.values()]
 def emit(kind,**kw):
  e={'time':now(),'run_id':args.run_id,'kind':kind,**kw};line=json.dumps(e,ensure_ascii=False)+'\n'
  for f in logs: f.parent.mkdir(exist_ok=True);f.open('a').write(line)
  print(line,end='',flush=True)
 queue=list(args.variants);running={};failures=[];emit('start',jobs=len(queue),gpus=args.gpus,policy='force configured GPUs; ignore external occupancy')
 while queue or running:
  for g,s in list(running.items()):
   rc=s['p'].poll()
   if rc is None: continue
   s['f'].close();v=s['v'];emit('finish',variant=v,gpu=g,pid=s['p'].pid,returncode=rc,log=str(s['log']))
   if rc: failures.append({'variant':v,'gpu':g,'returncode':rc})
   running.pop(g)
  inv=inventory(args.gpus)
  for g in args.gpus:
   if not queue or g in running or g not in inv: continue
   v=queue.pop(0);d=dirs[v];stamp=now();log=d/'logs'/f'{stamp}_uit_gpu{g}.log';cmd=[sys.executable,str(d/'scripts/run_scene.py'),'--output',str(d),'--gpu','0','--seed',str(args.seed),'--epochs',str(args.epochs),'--batch',str(args.batch),'--workers',str(args.workers)];env={**os.environ,'CUDA_VISIBLE_DEVICES':inv[g]['uuid'],'CUDA_DEVICE_ORDER':'PCI_BUS_ID','PYTHONUNBUFFERED':'1'};f=log.open('w');pobj=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);running[g]={'v':v,'p':pobj,'f':f,'log':log};emit('launch',variant=v,gpu=g,uuid=inv[g]['uuid'],name=inv[g]['name'],memory_mb=inv[g]['memory_mb'],pid=pobj.pid,log=str(log),command=cmd)
 if queue: emit('waiting',queued=queue,occupied=list(running),inventory=inv)
 if queue or running: time.sleep(args.poll_seconds)
 emit('all_finished',jobs=len(args.variants),failures=failures);return 1 if failures else 0
if __name__=='__main__': sys.exit(main())
