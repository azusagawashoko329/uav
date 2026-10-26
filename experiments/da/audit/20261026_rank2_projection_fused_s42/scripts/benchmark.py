"""Locate and remove Rank2 projection overhead without changing its FP32 math."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import itertools
import json
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.bottleneck import build_bottleneck
from model.bottleneck_depthrank_learnable import build_learnable_depthrank2_bottleneck
from model.student_structural import FourBlockGlobalLocalDecoder
from model.loss import anomaly_score


def stamp():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_helpers():
    path = ROOT/'experiments/da/audit/20261026_rank2_cacheq_fp32_s42/scripts/benchmark.py'
    spec = importlib.util.spec_from_file_location('cacheq_validation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def loop_project(taps, q):
    modes = []
    for i in range(0,len(taps),3):
        a,b,c = taps[i:i+3]
        modes.append(q[0,0]*a+q[0,1]*b+q[0,2]*c)
        modes.append(q[1,0]*a+q[1,1]*b+q[1,2]*c)
    return torch.cat(modes,dim=-1)


def setup(gpu):
    c = json.loads((EXP/'config.json').read_text())
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(device)
    assert torch.cuda.get_device_name(device)==c['gpu_name']
    torch.set_num_threads(c['cpu_threads'])
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    import triton
    env=dict(started_at=stamp(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             gpu=torch.cuda.get_device_name(device),cuda_ordinal=gpu,torch=torch.__version__,
             triton=triton.__version__,cuda=torch.version.cuda,python=platform.python_version(),
             dtype='float32',tf32=False,cpu_threads=torch.get_num_threads(),config_sha256=digest(EXP/'config.json'),
             nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.used,utilization.gpu','--format=csv'],text=True))
    return c,device,env


def build_models(device):
    torch.manual_seed(42)
    base=build_learnable_depthrank2_bottleneck(basis_init='random_orthonormal',basis_seed=42).to(device).eval()
    fused=copy.deepcopy(base)
    torch.manual_seed(42)
    dense=build_bottleneck(n_taps=12,fuse_hidden=128).to(device).eval()
    torch.manual_seed(42)
    student=FourBlockGlobalLocalDecoder(dim=192).to(device).eval()
    return dict(dense128=dense,cached_loop=base,fused=fused),student


def restore_q(models,scene,c,device):
    path=ROOT/c['source_experiment']/'results'/f'{scene}_basis.json'
    assert digest(path)==c['basis_sha256'][scene]
    history=json.loads(path.read_text())['history'][-1]
    assert history['epoch']==15
    for name in ['cached_loop','fused']:
        model=models[name]
        with torch.no_grad():
            model.raw_depth_basis.copy_(torch.tensor(history['raw_q'],device=device))
        model.cache_depth_basis_for_inference(fuse_projection=name=='fused')
    return dict(path=str(path.relative_to(ROOT)),sha256=digest(path),raw_q=history['raw_q'])


def verify_fused_contract(device):
    checks=legacy_helpers().verify_cache_lifecycle()
    from model.depthrank_projection import project_depth_rank2
    errors=[]
    with torch.inference_mode():
        # Non-multiple tile sizes and noncontiguous teacher-shaped slices detect
        # incorrect group/direction/channel ordering and masked tail accesses.
        for b,p,ch,groups in [(1,7,17,4),(2,5,33,2),(1,1,8,1),(2,13,19,4)]:
            taps=[torch.randn(b,p+5,ch,device=device)[:,5:] for _ in range(groups*3)]
            q=torch.randn(2,3,device=device)
            a,bout=loop_project(taps,q),project_depth_rank2(taps,q)
            torch.testing.assert_close(a,bout,atol=0,rtol=0)
            errors.append(float((a-bout).abs().max().cpu()))
    model=build_learnable_depthrank2_bottleneck(embed_dim=17,code_dim=4,fuse_hidden=8,
                   basis_init='random_orthonormal',basis_seed=42).to(device).eval()
    ref=copy.deepcopy(model)
    keys=set(model.state_dict())
    model.cache_depth_basis_for_inference(fuse_projection=True)
    assert keys==set(model.state_dict())
    taps=[torch.randn(2,7,17,device=device) for _ in range(12)]
    with torch.inference_mode():
        torch.testing.assert_close(model(taps),ref(taps),atol=0,rtol=0)
    # Fused inference must not intercept gradient-enabled evaluation/training.
    model(taps).square().sum().backward(); ref(taps).square().sum().backward()
    for a,b in zip(model.parameters(),ref.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=0,rtol=0)
    assert model.raw_depth_basis.grad.abs().max()>0
    model.train()
    assert model._inference_depth_projector is None
    model.eval().cache_depth_basis_for_inference(fuse_projection=True)
    model.load_state_dict(ref.state_dict(),strict=True)
    assert model._inference_depth_projector is None
    model.cache_depth_basis_for_inference(fuse_projection=True).double()
    assert model._inference_depth_projector is None
    checks.update(fused_mask_and_layout_cases=len(errors),fused_max_abs_diff=max(errors),
                  fused_eval_gradients_bitwise_equal=True,fused_checkpoint_keys_unchanged=True,
                  fused_training_clears_projector=True,fused_load_clears_projector=True,
                  fused_dtype_change_clears_projector=True)
    return checks


def measure(functions,batch,c,device):
    names=list(functions)
    orders=list(itertools.permutations(names)) if len(names)==3 else [tuple(names),tuple(reversed(names))]
    times={k:[] for k in names}; rates={k:[] for k in names}
    with torch.inference_mode():
        for fn in functions.values():
            for _ in range(c['warmup']):fn()
        torch.cuda.synchronize(device)
        for repeat in range(c['repeats']):
            order=orders[repeat%len(orders)]
            for name in order:
                values=[];fn=functions[name]
                for _ in range(c['samples']):
                    start=time.perf_counter();fn();torch.cuda.synchronize(device)
                    values.append((time.perf_counter()-start)*1000)
                times[name].append(values)
            for name in order:
                fn=functions[name];torch.cuda.synchronize(device);start=time.perf_counter()
                for _ in range(c['samples']):fn()
                torch.cuda.synchronize(device)
                rates[name].append(batch*c['samples']/(time.perf_counter()-start))
    summarize=legacy_helpers().summary_stats
    return {name:summarize(times[name],rates[name],batch) for name in names}


def timing_main(c,device,env):
    from model.depthrank_projection import project_depth_rank2
    helpers=legacy_helpers()
    checks=verify_fused_contract(device)
    dump(EXP/'results/verification.json',checks)
    models,student=build_models(device)
    assert sum(p.numel() for p in models['fused'].parameters())+sum(p.numel() for p in student.parameters())==3761222
    result=dict(environment=env,config=c,scenes=[])
    for scene in c['scenes']:
        entry=dict(scene=scene,q_source=restore_q(models,scene,c,device),timings={},verification={})
        for k,v in models['cached_loop'].state_dict().items():
            assert torch.equal(v,models['fused'].state_dict()[k])
        for batch in c['batches']:
            taps=helpers.input_taps(batch,device)
            with torch.inference_mode():
                norm=[layer(t) for layer,t in zip(models['cached_loop'].tap_norms,taps)]
                q=models['cached_loop'].effective_depth_basis()
                projected=loop_project(norm,q)
                optimized=project_depth_rank2(norm,q)
                torch.testing.assert_close(projected,optimized,atol=0,rtol=0)
                code0,code1=models['cached_loop'](taps),models['fused'](taps)
                torch.testing.assert_close(code0,code1,atol=0,rtol=0)
                r0,r1=student(code0),student(code1)
                for x,y in zip(r0,r1):torch.testing.assert_close(x,y,atol=0,rtol=0)
                s0,s1=anomaly_score(taps,r0),anomaly_score(taps,r1)
                torch.testing.assert_close(s0,s1,atol=0,rtol=0)
                dense_input=torch.cat(norm,dim=-1)
                entry['verification'][str(batch)]=dict(projection_max_abs_diff=float((projected-optimized).abs().max().cpu()),
                    code_max_abs_diff=float((code0-code1).abs().max().cpu()),
                    reconstruction_max_abs_diff=max(float((x-y).abs().max().cpu()) for x,y in zip(r0,r1)),
                    score_max_abs_diff=float((s0-s1).abs().max().cpu()))
            scopes={
                'pack_or_project':dict(dense128=lambda:torch.cat(norm,dim=-1),
                                      cached_loop=lambda:loop_project(norm,q),fused=lambda:project_depth_rank2(norm,q)),
                'fuse_mlp':dict(dense128=lambda:models['dense128'].fuse(dense_input),
                               cached_loop=lambda:models['cached_loop'].fuse(projected),
                               fused=lambda:models['fused'].fuse(optimized)),
                'ocbe':{k:(lambda model=m:model(taps)) for k,m in models.items()},
                'subnet':{k:(lambda model=m:student(model(taps))) for k,m in models.items()},
                'subnet_with_score':{k:(lambda model=m:anomaly_score(taps,student(model(taps)))) for k,m in models.items()},
            }
            entry['timings'][str(batch)]={k:measure(fns,batch,c,device) for k,fns in scopes.items()}
            if scene==c['scenes'][0]:
                entry.setdefault('common_components',{})[str(batch)]={
                    'tap_norms':measure({'common':lambda:[layer(t) for layer,t in zip(models['cached_loop'].tap_norms,taps)]},batch,c,device),
                    'student':measure({'common':lambda:student(code0)},batch,c,device)}
            m=entry['timings'][str(batch)]['subnet']
            print(json.dumps(dict(event='measured',scene=scene,batch=batch,
                       latency_ms={k:v['latency_p50_ms'] for k,v in m.items()},
                       fps={k:v['serial_fps'] for k,v in m.items()})),flush=True)
        result['scenes'].append(entry);result['last_updated_at']=stamp()
        dump(EXP/'results/measurements.json',result)
    result['timing_finished_at']=stamp()
    dump(EXP/'results/measurements.json',result)
    return result


def profile_main(c,device,env):
    models,_=build_models(device)
    restore_q(models,c['scenes'][0],c,device)
    taps=legacy_helpers().input_taps(1,device)
    profiles={}
    for name,model in models.items():
        with torch.inference_mode():
            for _ in range(30):model(taps)
            torch.cuda.synchronize(device)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(20):model(taps)
                torch.cuda.synchronize(device)
        ops=[dict(name=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,
                  self_cuda_us=e.self_cuda_time_total) for e in prof.key_averages() if e.key.startswith('aten::')]
        kernels=[e for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
        profiles[name]=dict(forwards=20,cuda_kernel_count=len(kernels),
                           kernel_count_per_forward=len(kernels)/20,
                           cuda_kernel_duration_sum_us=sum(e.time_range.elapsed_us() for e in kernels),
                           kernel_names=dict(Counter(e.name for e in kernels)),aten_operators=ops)
    from model.depthrank_projection import _project_triplets
    compiled=list(_project_triplets.cache[device.index].values())
    ptx=compiled[-1].asm['ptx']
    (EXP/'results/fused_projection.ptx').write_text(ptx)
    profiles['assembly']=dict(path='results/fused_projection.ptx',sha256=digest(EXP/'results/fused_projection.ptx'),
                              fp32_fma_count=len(re.findall(r'\bfma\.[a-z.]*f32\b',ptx)))
    assert profiles['assembly']['fp32_fma_count']==0
    result=dict(environment=env,profiles=profiles,finished_at=stamp())
    dump(EXP/'results/profiler.json',result)
    print(json.dumps(dict(event='profile_completed',counts={k:v['kernel_count_per_forward'] for k,v in profiles.items() if 'forwards' in v})),flush=True)


def report(result):
    rows=[];aggregates={}
    for scene in result['scenes']:
        for batch,scopes in scene['timings'].items():
            for scope,modes in scopes.items():
                for name,metric in modes.items():
                    rows.append(dict(scene=scene['scene'],batch=int(batch),scope=scope,implementation=name,
                                     **{k:v for k,v in metric.items() if not isinstance(v,list)}))
    with (EXP/'results/summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    for batch in result['config']['batches']:
        aggregates[str(batch)]={}
        for scope in result['scenes'][0]['timings'][str(batch)]:
            aggregates[str(batch)][scope]={}
            for name in ['dense128','cached_loop','fused']:
                value=mean(s['timings'][str(batch)][scope][name]['latency_p50_ms'] for s in result['scenes'])
                aggregates[str(batch)][scope][name]=dict(mean_scene_p50_ms=value,fps_from_mean=1000*batch/value)
    result['aggregates']=aggregates
    profile=json.loads((EXP/'results/profiler.json').read_text())
    lines=['# Rank2投影融合：定位并修复执行开销','','GL192＋Dense128＋共享可学习随机正交初始化Q，seed42。恢复七场景epoch15 raw_Q，比较缓存后原逐元素循环与保持FP32乘加顺序的融合投影。Dense128为同轮参考。训练结构、Q值、loss和所有原state_dict键均不改。','',
           '## 计时口径','','RTX 5880 Ada、FP32、TF32/AMP关闭、256×256、batch1/16。每函数预热30次、六轮各100次，三个条件以全部六种顺序轮换。Latency为每次CUDA同步的墙钟p50；FPS=1000×batch/Latency。连续队列FPS另存CSV/JSON。Profiler在全部计时结束后的独立子进程执行。','',
           '旧完整checkpoint未保存，本次恢复各场景最终Q，其他权重使用完全配对的同构初始化。检查函数等价性与执行开销，不重新评测AUC。','',
           '## 单帧主结果：纯OCBE＋学生','','按七场景p50取算术均值，FPS按该均值换算；七个Q不表示七份硬件独立样本。','',
           '| 条件 | OCBE ms | 子网 ms | FPS | CUDA kernel/OCBE |','|---|---:|---:|---:|---:|']
    for name in ['dense128','cached_loop','fused']:
        x=aggregates['1'];k=profile['profiles'][name]['kernel_count_per_forward']
        lines.append(f"| {name} | {x['ocbe'][name]['mean_scene_p50_ms']:.4f} | {x['subnet'][name]['mean_scene_p50_ms']:.4f} | {x['subnet'][name]['fps_from_mean']:.1f} | {k:g} |")
    lines+=['','## 逐场景单帧配对','','| 场景 | Dense ms | 原Rank2 ms | 融合Rank2 ms | 原FPS | 融合FPS |','|---|---:|---:|---:|---:|---:|']
    for s in result['scenes']:
        t=s['timings']['1']['subnet'];a,b=t['cached_loop'],t['fused']
        lines.append(f"| {s['scene']} | {t['dense128']['latency_p50_ms']:.4f} | {a['latency_p50_ms']:.4f} | {b['latency_p50_ms']:.4f} | {a['serial_fps']:.1f} | {b['serial_fps']:.1f} |")
    lines+=['','## 子步骤隔离','','以下是各阶段单独调用的耗时，不能直接相加替代pipeline实测。Dense128的pack_or_project仅执行concat，Rank2两种实现执行Q投影；它们输出宽度不同，后续MLP随之缩小。','',
            '| 步骤 | Dense128 ms | 原Rank2 ms | 融合Rank2 ms |','|---|---:|---:|---:|']
    for scope in ['pack_or_project','fuse_mlp','ocbe','subnet','subnet_with_score']:
        v=aggregates['1'][scope]
        lines.append('| '+scope+' | '+' | '.join(f"{v[k]['mean_scene_p50_ms']:.4f}" for k in ['dense128','cached_loop','fused'])+' |')
    lines+=['','## 验证、解释与使用','','原投影包含24次逐元素乘法、16次逐元素加法及一次拼接；融合版先stack十二层，再用一个kernel计算四组所有patch/通道并直接写出[group,direction,channel]顺序。使用显式round-to-nearest乘法/加法，保持原先(乘积0＋乘积1)＋乘积2的顺序。PTX单独归档并检查无FP32 FMA。','',
            '验证包括14组完整尺寸投影、OCBE、12个学生输出及异常分数，要求逐位一致；还包括非整块尺寸、非连续输入、梯度开启的eval、训练恢复、加载权重和精度转换的缓存失效。误差/状态见verification.json与measurements.json。','',
            '启用：加载权重 → to(device).eval() → cache_depth_basis_for_inference(fuse_projection=True) → inference_mode下前向。默认缓存接口仍为原循环以保持兼容；显式融合只支持CUDA FP32并依赖Triton，训练及梯度开启路径继续使用原实现。设备/精度转换、重新训练或载入新权重后重新准备。','',
            'Rank2训练参数仍为3,761,222，主要矩阵算术仍为1.957199872 GFLOPs。Q投影本身另有约0.00786432 GFLOPs逐元素运算（batch1），此前主要FLOPs口径未计它；融合减少执行次数和中间张量，不减少这部分数学运算。','',
            '桌面GPU合成特征、未锁定时钟或绑核、不包含教师/预处理/H2D；不是完整模型FPS。历史AUC 82.9599未重算，不能将本次初始化网络测速视作新AUC结果。','']
    (EXP/'results/summary.md').write_text('\n'.join(lines))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--mode',choices=['timing','profile'],default='timing');args=ap.parse_args()
    c,device,env=setup(args.gpu)
    print(json.dumps(dict(event='started',mode=args.mode,**env)),flush=True)
    if args.mode=='profile':
        profile_main(c,device,env);return
    record=json.loads((EXP/'record.json').read_text())
    run=dict(run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_fused_timing',status='running',source_commit=env['source_commit'],
             started_at=env['started_at'],ended_at=None,script='scripts/benchmark.py',log='logs/timing.log',
             command=[sys.executable,str(Path(__file__).relative_to(ROOT)),'--gpu',str(args.gpu),'--mode','timing'],
             seed=42,data_root=None,data_split='synthetic teacher-shaped tensors; archived seven final Qs',environment=env,outputs=[])
    record.update(status='running',source_commit=env['source_commit'],runs=[run]);dump(EXP/'record.json',record)
    try:
        result=timing_main(c,device,env)
        command=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT)),'--gpu',str(args.gpu),'--mode','profile']
        with (EXP/'logs/profile.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        profile=json.loads((EXP/'results/profiler.json').read_text())
        record['runs'].append(dict(run_id=datetime.fromisoformat(profile['environment']['started_at']).strftime('%Y%m%dT%H%M%SZ')+'_fused_profile',
             status='completed',source_commit=env['source_commit'],started_at=profile['environment']['started_at'],ended_at=profile['finished_at'],
             command=command,seed=42,script='scripts/benchmark.py',log='logs/profile.log',data_root=None,
             data_split='bike final Q; synthetic GPU tensors; no timing after profiler',environment=profile['environment'],exit_code=0,
             outputs=['results/profiler.json','results/fused_projection.ptx']))
        report(result);result['finished_at']=stamp();dump(EXP/'results/measurements.json',result)
        run.update(status='completed',ended_at=result['finished_at'],exit_code=0,outputs=['results/measurements.json','results/verification.json','results/summary.csv','results/summary.md'])
        record.update(status='completed',result_summary='results/summary.md')
        print(json.dumps(dict(event='completed',subnet=result['aggregates']['1']['subnet'])),flush=True)
    except Exception as exc:
        run.update(status='failed',ended_at=stamp(),reason=repr(exc));record['status']='failed';raise
    finally:
        dump(EXP/'record.json',record)


if __name__=='__main__':main()
