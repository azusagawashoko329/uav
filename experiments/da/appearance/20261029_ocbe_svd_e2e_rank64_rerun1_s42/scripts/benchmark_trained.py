"""Trained-checkpoint latency audit for this experiment."""
import json, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[5]
EXP = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from model.bottleneck_depthrank_learnable import build_learnable_depthrank2_bottleneck
from model.bottleneck_rank2_svd import build_end_to_end_factorized_rank2_bottleneck
from model.student_structural import FourBlockGlobalLocalDecoder
from model.loss import anomaly_score

def stamp(): return datetime.now(timezone.utc).isoformat()
def taps(batch, device, i):
    g = torch.Generator(device=device).manual_seed(9100 + i * 13 + batch)
    return [torch.randn(batch, 256, 768, device=device, generator=g) for _ in range(12)]
def measure(fn, device, warmup, samples, repeats, batch):
    with torch.inference_mode():
        for _ in range(warmup): fn()
        torch.cuda.synchronize(device); values=[]
        for _ in range(repeats):
            for _ in range(samples):
                t=time.perf_counter(); fn(); torch.cuda.synchronize(device)
                values.append((time.perf_counter()-t)*1000)
    values.sort(); p50=values[len(values)//2]
    return {'latency_p50_ms':p50, 'latency_p95_ms':values[min(len(values)-1,int(.95*len(values)))],
            'serial_fps':1000*batch/p50, 'batch':batch, 'samples':samples, 'repeats':repeats}
CFG=json.loads((EXP/'config.json').read_text())
variant=CFG['variant']
device=torch.device('cuda:0'); torch.cuda.set_device(device)
torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
result={'environment':{'started_at':stamp(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'cuda_device_name':torch.cuda.get_device_name(device),'gpu':0,'torch':torch.__version__,'cuda':torch.version.cuda,
 'dtype':'float32','tf32':False},'variant':variant,'scenes':[]}
for i, scene in enumerate(CFG['scenes']):
    state=torch.load(EXP/'ckpt'/f'{scene}_final.pth',map_location='cpu',weights_only=False)
    if variant.startswith('rank2_'):
        model=build_learnable_depthrank2_bottleneck(n_taps=12,code_dim=64,fuse_hidden=128,basis_init='random_orthonormal',basis_seed=42)
    elif variant.startswith('svd_e2e_'):
        model=build_end_to_end_factorized_rank2_bottleneck(rank=64,n_taps=12,code_dim=64,hidden=128,basis_init='random_orthonormal',basis_seed=42)
    else: raise ValueError(variant)
    model.load_state_dict(state['ocbe_state_dict'],strict=True); model=model.to(device).eval()
    student=FourBlockGlobalLocalDecoder(dim=192,teacher_dim=768).to(device).eval()
    student.load_state_dict(state['student_state_dict'],strict=True)
    model.cache_depth_basis_for_inference(fuse_projection=True)
    scene_result={'scene':scene,'auc_percent':json.loads((EXP/'results'/f'{scene}.json').read_text())['auc_percent'],'timings':{},'verification':{}}
    for batch in [1,16]:
        x=taps(batch,device,i)
        with torch.inference_mode():
            y=model(x); s=student(y); a=anomaly_score(x,s)
            scene_result['verification'][str(batch)]={'code_finite':bool(torch.isfinite(y).all()),'score_finite':bool(torch.isfinite(a).all())}
        n=100 if batch==1 else 30
        scene_result['timings'][str(batch)]={'core':measure(lambda:student(model(x)),device,30,n,4,batch),
          'score':measure(lambda:anomaly_score(x,student(model(x))),device,30,n,4,batch)}
    result['scenes'].append(scene_result); del model,student; torch.cuda.empty_cache()
result['finished_at']=stamp()
out=EXP/'results'/f'{variant}_trained_gpu0.json'; out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'event':'completed','variant':variant,'scenes':7}),flush=True)
