"""Latency audit of the seven trained LCDB checkpoints."""
from __future__ import annotations

import copy
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.bottleneck_lcdb import build_lcdb_bottleneck
from model.loss import anomaly_score
from model.student_structural import FourBlockGlobalLocalDecoder


def stamp(): return datetime.now(timezone.utc).isoformat()


def taps(batch, device, i):
    g = torch.Generator(device=device).manual_seed(9100 + i * 13 + batch)
    return [torch.randn(batch, 256, 768, device=device, generator=g) for _ in range(12)]


def measure(fn, device, warmup, samples, repeats, batch):
    with torch.inference_mode():
        for _ in range(warmup): fn()
        torch.cuda.synchronize(device); values=[]; rounds=[]
        for _ in range(repeats):
            cur=[]
            for _ in range(samples):
                t=time.perf_counter(); fn(); torch.cuda.synchronize(device)
                cur.append((time.perf_counter()-t)*1000)
            values.extend(cur); rounds.append(cur)
    values.sort(); p50=values[len(values)//2]
    return dict(latency_p50_ms=p50, latency_p95_ms=values[min(len(values)-1,int(.95*len(values)))],
                serial_fps=1000*batch/p50, round_p50_ms=[sorted(x)[len(x)//2] for x in rounds],
                batch=batch,samples=samples,repeats=repeats)


def main():
    cfg=json.loads((EXP/'config.json').read_text()); device=torch.device('cuda:0'); torch.cuda.set_device(device)
    torch.set_num_threads(cfg['cpu_threads']); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    env=dict(started_at=stamp(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             cuda_device_name=torch.cuda.get_device_name(device),gpu=0,torch=torch.__version__,cuda=torch.version.cuda,
             python=platform.python_version(),dtype='float32',tf32=False,cpu_threads=torch.get_num_threads())
    result=dict(environment=env,scenes=[])
    for i,scene in enumerate(cfg['scenes']):
        state=torch.load(EXP/'ckpt'/f'{scene}_final.pth',map_location='cpu',weights_only=False)
        base=build_lcdb_bottleneck(hidden=128).to(device).eval(); student=FourBlockGlobalLocalDecoder(dim=192,teacher_dim=768).to(device).eval()
        base.load_state_dict(state['ocbe_state_dict'],strict=True); student.load_state_dict(state['student_state_dict'],strict=True)
        original=copy.deepcopy(base).eval(); cached=copy.deepcopy(base).eval(); fused=copy.deepcopy(base).eval()
        cached.cache_depth_basis_for_inference(fuse_projection=False); fused.cache_depth_basis_for_inference(fuse_projection=True)
        scene_result=dict(scene=scene,auc_percent=json.loads((EXP/'results'/f'{scene}.json').read_text())['auc_percent'],timings={},verification={})
        for batch in cfg['batches']:
            x=taps(batch,device,i)
            with torch.inference_mode():
                y0,y1,y2=original(x),cached(x),fused(x); s0,s1,s2=student(y0),student(y1),student(y2)
                a0,a1,a2=anomaly_score(x,s0),anomaly_score(x,s1),anomaly_score(x,s2)
                scene_result['verification'][str(batch)]=dict(cached_code_max_abs_diff=float((y0-y1).abs().max().cpu()),fused_code_max_abs_diff=float((y0-y2).abs().max().cpu()),fused_score_max_abs_diff=float((a0-a2).abs().max().cpu()))
            funcs={'original':original,'cached_loop':cached,'fused':fused}
            samples=cfg['samples'] if batch==1 else cfg['batch16_samples']
            scene_result['timings'][str(batch)]={scope:{name:measure((lambda m=m:student(m(x))) if scope=='subnet' else (lambda m=m:anomaly_score(x,student(m(x)))),device,cfg['warmup'],samples,cfg['repeats'],batch) for name,m in funcs.items()} for scope in ('subnet','score')}
        result['scenes'].append(scene); del base,original,cached,fused,student; torch.cuda.empty_cache()
    result['finished_at']=stamp(); (EXP/'results'/'lcdb_trained_gpu0.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'event':'completed','scenes':len(result['scenes'])}),flush=True)


if __name__=='__main__':main()
