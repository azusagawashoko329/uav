#!/usr/bin/env python3
"""Dynamic two-GPU scheduler for the five UIT-ADrone appearance variants."""
from __future__ import annotations
import argparse, datetime as dt, json, os, subprocess, sys, time
from pathlib import Path
VARIANTS=['20261030_uit_4block_plain_s42','20261030_uit_4block_gl768_s42','20261030_uit_4block_gl192_s42','20261030_uit_gl192_dense128_s42','20261030_uit_gl192_dense128_rank2_s42']
def now(): return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def probe(gpus):
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True);out={}
 for line in raw.strip().splitlines():
  i,u,t=(int(x.strip()) for x in line.split(','));
  if i in gpus: out[str(i)]={'used_mb':u,'total_mb':t}
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo-root',type=Path,required=True);ap.add_argument('--python',default='/home/lyx/miniconda3/envs/DMAD/bin/python');ap.add_argument('--gpus',nargs='+',type=int,default=[0,1]);ap.add_argument('--poll-seconds',type=float,default=2);ap.add_argument('--max-memory-mb',type=int,default=2000);ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--batch',type=int,default=64);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--dry-run',action='store_true');args=ap.parse_args();root=args.repo_root.resolve();dirs={v:root/'experiments'/'uit'/'appearance'/v for v in VARIANTS};
 for d in dirs.values(): (d/'logs').mkdir(parents=True,exist_ok=True);(d/'results').mkdir(parents=True,exist_ok=True)
 status=probe(args.gpus);print(json.dumps({'kind':'gpu_probe','gpus':args.gpus,'status':status,'max_memory_mb':args.max_memory_mb}),flush=True);busy={g:status[str(g)] for g in args.gpus if status.get(str(g),{}).get('used_mb',999999)>args.max_memory_mb};
 if busy: raise RuntimeError(f'GPUs busy: {busy}')
 if args.dry_run: print(json.dumps({'kind':'dry_run_ok','jobs':len(VARIANTS),'variants':VARIANTS}),flush=True);return 0
 run_id=now()+'_uit_five_variant_matrix';logs={v:dirs[v]/'logs'/f'scheduler_{run_id}.jsonl' for v in VARIANTS}
 def emit(kind,**kw):
  e={'time':now(),'run_id':run_id,'kind':kind,**kw};line=json.dumps(e,ensure_ascii=False)+'\n';
  for f in logs.values(): f.open('a').write(line)
  print(line,end='',flush=True)
 queue=list(VARIANTS);running={};failures=[]
 def launch(gpu,v):
  d=dirs[v];stamp=now();lp=d/'logs'/f'{stamp}_uit_gpu{gpu}.log';cmd=[args.python,str(d/'scripts'/'run_scene.py'),'--output',str(d),'--gpu',str(gpu),'--seed',str(args.seed),'--epochs',str(args.epochs),'--batch',str(args.batch),'--workers',str(args.workers)];log=lp.open('w');p=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'UAV_UIT_GPU':str(gpu)});running[gpu]={'variant':v,'proc':p,'log':log,'path':lp};emit('launch',variant=v,gpu=gpu,pid=p.pid,log=str(lp),command=cmd)
 for gpu in args.gpus:
  if queue: launch(gpu,queue.pop(0))
 while running:
  for gpu,s in list(running.items()):
   rc=s['proc'].poll()
   if rc is None: continue
   s['log'].close();emit('finish',variant=s['variant'],gpu=gpu,pid=s['proc'].pid,returncode=rc,log=str(s['path']));
   if rc: failures.append({'variant':s['variant'],'gpu':gpu,'returncode':rc})
   del running[gpu]
   if queue:launch(gpu,queue.pop(0))
  if running:time.sleep(args.poll_seconds)
 emit('all_finished',jobs=len(VARIANTS),failures=failures);return 1 if failures else 0
if __name__=='__main__':sys.exit(main())
