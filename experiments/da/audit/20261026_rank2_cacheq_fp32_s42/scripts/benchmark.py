"""Paired inference audit: only cache the final effective shared Rank2 Q."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.bottleneck import build_bottleneck
from model.bottleneck_depthrank import build_depthrank2_bottleneck
from model.bottleneck_depthrank_learnable import build_learnable_depthrank2_bottleneck
from model.student_structural import FourBlockGlobalLocalDecoder
from model.loss import anomaly_score


def stamp():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_cache_lifecycle():
    """Check the gradient and checkpoint contracts that an inference cache risks."""
    torch.manual_seed(42)
    model = build_learnable_depthrank2_bottleneck(embed_dim=8, code_dim=4,
                fuse_hidden=8, basis_init='random_orthonormal', basis_seed=42).eval()
    reference = copy.deepcopy(model)
    taps = [torch.randn(2, 4, 8) for _ in range(12)]
    original_keys = set(model.state_dict())
    original_params = sum(p.numel() for p in model.parameters())
    model.cache_depth_basis_for_inference()
    assert set(model.state_dict()) == original_keys
    assert sum(p.numel() for p in model.parameters()) == original_params
    with torch.inference_mode():
        torch.testing.assert_close(model(taps), reference(taps), atol=0, rtol=0)
    # eval() alone does not disable autograd; the cached detached Q must not
    # interrupt gradients when gradient-enabled evaluation is requested.
    model(taps).square().sum().backward()
    reference(taps).square().sum().backward()
    for a, b in zip(model.parameters(), reference.parameters()):
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
    assert model.raw_depth_basis.grad.abs().max() > 0
    model.train()
    assert model._inference_depth_basis is None
    try:
        model.cache_depth_basis_for_inference()
    except RuntimeError:
        pass
    else:
        raise AssertionError('Training mode accepted a detached inference cache')
    model.eval().cache_depth_basis_for_inference()
    model.load_state_dict(reference.state_dict(), strict=True)
    assert model._inference_depth_basis is None
    model.cache_depth_basis_for_inference().double()
    assert model._inference_depth_basis is None
    model.float().cache_depth_basis_for_inference()
    with torch.no_grad():
        model.raw_depth_basis[0, 0].add_(0.1)
    with torch.inference_mode():
        model(taps)
    assert model._inference_depth_basis is None
    return dict(checkpoint_keys_unchanged=True, parameter_count_unchanged=True,
                cached_cpu_output_bitwise_equal=True, eval_gradients_bitwise_equal=True,
                train_clears_cache=True, training_cache_rejected=True,
                checkpoint_load_clears_cache=True, dtype_change_clears_cache=True,
                in_place_parameter_change_invalidates_cache=True)


def summary_stats(rounds, rates, batch):
    values = np.array(rounds).ravel()
    p50 = float(np.median(values))
    return dict(latency_p50_ms=p50, latency_p95_ms=float(np.percentile(values, 95)),
                latency_mean_ms=float(values.mean()), latency_std_ms=float(values.std()),
                round_p50_ms=[float(np.median(r)) for r in rounds],
                serial_fps=1000 * batch / p50,
                queued_fps=float(np.median(rates)), throughput_rounds_fps=rates)


def paired_measure(functions, batch, config, device):
    """Interleave AB/BA rounds so a slow drift cannot favor the cached copy."""
    samples, repeats = config['samples'], config['repeats']
    timings = {name: [] for name in functions}
    rates = {name: [] for name in functions}
    with torch.inference_mode():
        for fn in functions.values():
            for _ in range(config['warmup']):
                fn()
        torch.cuda.synchronize(device)
        for repeat in range(repeats):
            order = list(functions)
            if repeat % 2:
                order.reverse()
            for name in order:
                fn = functions[name]
                values = []
                for _ in range(samples):
                    start = time.perf_counter()
                    fn()
                    torch.cuda.synchronize(device)
                    values.append((time.perf_counter() - start) * 1000)
                timings[name].append(values)
            for name in order:
                fn = functions[name]
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                for _ in range(samples):
                    fn()
                torch.cuda.synchronize(device)
                rates[name].append(batch * samples / (time.perf_counter() - start))
    return {name: summary_stats(timings[name], rates[name], batch) for name in functions}


def input_taps(batch, device):
    # Match the original teacher's patch slice: five prefix tokens are skipped.
    return [torch.randn(batch, 261, 768, device=device)[:, 5:] for _ in range(12)]


def profile_ocbe(model, taps, device, steps=10):
    try:
        with torch.inference_mode():
            for _ in range(10):
                model(taps)
            torch.cuda.synchronize(device)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(steps):
                    model(taps)
                torch.cuda.synchronize(device)
        ops = {e.key: e.count for e in prof.key_averages() if e.key.startswith('aten::')}
        kernels = Counter(e.name for e in prof.events()
                          if e.device_type == torch.autograd.DeviceType.CUDA)
        return dict(status='completed', forwards=steps, aten_operator_counts=ops,
                    cuda_kernel_count=sum(kernels.values()), cuda_kernel_names=dict(kernels))
    except Exception as exc:
        # GPU profiling can be unavailable independently of inference timing.
        return dict(status='failed', reason=repr(exc), forwards=steps)


def write_report(result):
    rows = []
    for scene in result['scenes']:
        for batch, scopes in scene['timings'].items():
            for scope, modes in scopes.items():
                for mode, metrics in modes.items():
                    rows.append(dict(scene=scene['scene'], batch=int(batch), scope=scope,
                                     mode=mode, **{k: v for k, v in metrics.items() if not isinstance(v, list)}))
    with (EXP/'results/summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    lines = ['# Rank2推理缓存Q：配对测速', '',
             '唯一改动是预先计算、缓存训练完成后的有效正交Q。原四组循环、通道融合、学生和异常评分计算均保持原样；训练默认不开启缓存。', '',
             'GL192＋Dense128＋共享可学习随机初始化Q，seed42；恢复七场景epoch15的原始Q。旧训练未保存完整checkpoint，其余权重按相同seed同构初始化；只用于配对资源审计，不重新评测AUC。', '',
             'GPU：'+result['environment']['gpu']+'；FP32、TF32/AMP关闭、输入256×256、CPU四线程。每项预热30次、六轮×100次，轮间交替AB/BA。Latency为逐次CUDA同步的墙钟中位数，FPS=1000×batch/Latency。另测连续队列吞吐。', '',
             '主表只包含OCBE＋学生纯前向，不包含教师和top-k评分。带评分以及batch16数据另存JSON/CSV。缓存前后使用完全相同的权重与输入。', '',
             '| 场景 | 原OCBE ms | 缓存OCBE ms | 原子网 ms | 缓存子网 ms | 原FPS | 缓存FPS | 子网延迟下降 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    gains = []
    for scene in result['scenes']:
        t = scene['timings']['1']
        a, b = t['subnet']['uncached'], t['subnet']['cached']
        gain = (1-b['latency_p50_ms']/a['latency_p50_ms'])*100
        gains.append(gain)
        lines.append(f"| {scene['scene']} | {t['ocbe']['uncached']['latency_p50_ms']:.4f} | {t['ocbe']['cached']['latency_p50_ms']:.4f} | {a['latency_p50_ms']:.4f} | {b['latency_p50_ms']:.4f} | {a['serial_fps']:.1f} | {b['serial_fps']:.1f} | {gain:.2f}% |")
    lines += ['', f'七场景配对延迟下降比例的算术平均：{np.mean(gains):.2f}%。这里七个场景只是七个最终Q的检查，不构成硬件独立重复样本。', '',
              '## 同轮参考与隔离实验', '',
              '| 条件 | OCBE ms | 子网 ms | 子网FPS |', '|---|---:|---:|---:|']
    for name, scopes in result['references'].items():
        m = scopes['subnet'][name]
        lines.append(f"| {name} | {scopes['ocbe'][name]['latency_p50_ms']:.4f} | {m['latency_p50_ms']:.4f} | {m['serial_fps']:.1f} |")
    qtime = result['basis_only']
    lines += ['', f"独立Q获取：重复GS {qtime['uncached']['latency_p50_ms']:.4f}ms，缓存读取 {qtime['cached']['latency_p50_ms']:.4f}ms。独立计时与pipeline中位数不能直接相加或相减作精确分解。", '',
              '## 一致性与使用方法', '',
              '调用顺序：加载该场景权重 → to(device/dtype) → eval() → cache_depth_basis_for_inference() → inference_mode下前向。缓存不写入state_dict，不增加训练参数。恢复train、加载新权重、设备/精度转换时清除缓存；梯度开启的eval仍走可微正交化，原始Q发生标准原地更新时缓存失效。', '',
              '七场景batch1/16校验Q、OCBE输出、学生12个重建输出和最终异常分数；详细误差见measurements.json。梯度、训练恢复、旧checkpoint兼容和缓存失效检查见verification.json。', '',
              '推理数学关系是Q_cached = GS(Q_raw_final)，原算子次序和系数不变。因此预期输出等价；本次没有完整旧权重，也没有在测试集重算AUC，不能把历史82.9599当成新评测值。', '',
              '本次训练参数仍为3,761,222，主要矩阵运算仍为1.957199872 GFLOPs；运行时缓存仅6个FP32系数（24字节）。没有改变Rank2压缩方向、学生结构或loss，也没有加入投影向量化、compile、量化等其他优化。', '',
              '## 剩余开销', '',
              '缓存只消除GS。每组三层特征的两个任意方向仍分别执行三次乘法、两次加法，四组共24次逐元素乘法、16次逐元素加法，再拼接。固定Mean＋Trend实现用更简单的加减与缩放，不能把它的速度直接当成缓存任意Q后的速度。', '',
              'Profiler的算子次数与CUDA kernel计数见profiler.json；该次profiling不用于主表时延。主要FLOPs沿用先前审计口径，不包含归一化、逐元素Q运算、激活和评分。', '',
              '限制：桌面GPU、合成GPU驻留特征，不含图像预处理、传输和教师前向；没有锁定GPU时钟/CPU绑核。与历史2.157ms/463.7FPS的横向差异可能含环境波动，收益以本次交替配对测量为准。', '']
    (EXP/'results/summary.md').write_text('\n'.join(lines))
    result['summary'] = dict(mean_paired_subnet_latency_reduction_percent=float(np.mean(gains)),
                             scenarios=7, trainable_parameters=3761222,
                             core_major_gflops=1.957199872, cache_bytes=24)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    config = json.loads((EXP/'config.json').read_text())
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    assert torch.cuda.get_device_name(device) == config['gpu_name']
    torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    env = dict(source_commit=source, started_at=stamp(), gpu=torch.cuda.get_device_name(device),
               cuda_ordinal=args.gpu, torch=torch.__version__, cuda=torch.version.cuda,
               python=platform.python_version(), dtype='float32', tf32=False,
               cpu_threads=torch.get_num_threads(), config_sha256=digest(EXP/'config.json'),
               nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.used,utilization.gpu','--format=csv'],text=True))
    print(json.dumps(dict(event='started', **env)), flush=True)
    record = json.loads((EXP/'record.json').read_text())
    run = dict(run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_cacheq',
               status='running', source_commit=source, started_at=env['started_at'], ended_at=None,
               command=[sys.executable, str(Path(__file__).relative_to(ROOT)), '--gpu', str(args.gpu)],
               seed=42, script='scripts/benchmark.py', log='logs/run.log',
               data_root=None, data_split='synthetic patch features; archived seven final Qs, no AUC retest',
               environment=env, outputs=[])
    record.update(status='running', source_commit=source, runs=[run])
    dump(EXP/'record.json', record)
    try:
        checks = verify_cache_lifecycle()
        dump(EXP/'results/verification.json', checks)
        result = dict(environment=env, config=config, references={}, scenes=[])
        source_exp = ROOT/config['source_experiment']
        for scene in config['scenes']:
            path = source_exp/'results'/f'{scene}_basis.json'
            assert digest(path) == config['basis_sha256'][scene]
            history = json.loads(path.read_text())['history'][-1]
            assert history['epoch'] == 15
            torch.manual_seed(42)
            original = build_learnable_depthrank2_bottleneck(basis_init='random_orthonormal', basis_seed=42).to(device).eval()
            with torch.no_grad():
                original.raw_depth_basis.copy_(torch.tensor(history['raw_q'],device=device))
            optimized = copy.deepcopy(original).cache_depth_basis_for_inference()
            student = FourBlockGlobalLocalDecoder(dim=192).to(device).eval()
            assert sum(p.numel() for p in original.parameters())+sum(p.numel() for p in student.parameters()) == 3761222
            assert set(original.state_dict()) == set(optimized.state_dict())
            for k, v in original.state_dict().items():
                assert torch.equal(v, optimized.state_dict()[k])
            entry = dict(scene=scene, q_source=str(path.relative_to(ROOT)), q_source_sha256=digest(path),
                         raw_q=history['raw_q'], timings={}, verification={})
            with torch.inference_mode():
                q = original.effective_depth_basis()
                torch.testing.assert_close(q, optimized.effective_depth_basis(), atol=0, rtol=0)
                torch.testing.assert_close(q, torch.tensor(history['effective_q'],device=device), atol=1e-6, rtol=1e-6)
                entry['effective_q'] = q.cpu().tolist()
            for batch in config['batches']:
                taps = input_taps(batch, device)
                with torch.inference_mode():
                    a, b = original(taps), optimized(taps)
                    ra, rb = student(a), student(b)
                    sa, sb = anomaly_score(taps,ra), anomaly_score(taps,rb)
                    torch.testing.assert_close(a,b,atol=0,rtol=0)
                    for x,y in zip(ra,rb):
                        torch.testing.assert_close(x,y,atol=0,rtol=0)
                    torch.testing.assert_close(sa,sb,atol=0,rtol=0)
                    entry['verification'][str(batch)] = dict(
                        ocbe_max_abs_diff=float((a-b).abs().max().cpu()),
                        reconstruction_max_abs_diff=max(float((x-y).abs().max().cpu()) for x,y in zip(ra,rb)),
                        score_max_abs_diff=float((sa-sb).abs().max().cpu()))
                scopes = {
                    'ocbe': {'uncached':lambda:original(taps), 'cached':lambda:optimized(taps)},
                    'subnet': {'uncached':lambda:student(original(taps)), 'cached':lambda:student(optimized(taps))},
                    'subnet_with_score': {'uncached':lambda:anomaly_score(taps,student(original(taps))),
                                          'cached':lambda:anomaly_score(taps,student(optimized(taps)))},
                }
                entry['timings'][str(batch)] = {scope:paired_measure(functions,batch,config,device)
                                                 for scope,functions in scopes.items()}
                t=entry['timings'][str(batch)]['subnet']
                print(json.dumps(dict(event='measured', scene=scene, batch=batch,
                           uncached_ms=t['uncached']['latency_p50_ms'], cached_ms=t['cached']['latency_p50_ms'],
                           uncached_fps=t['uncached']['serial_fps'], cached_fps=t['cached']['serial_fps'])),flush=True)
                if scene == config['scenes'][0] and batch == 1:
                    result['basis_only'] = paired_measure({'uncached':original.effective_depth_basis,
                                                          'cached':optimized.effective_depth_basis},1,config,device)
                    profiler = {name:profile_ocbe(m,taps,device) for name,m in [('uncached',original),('cached',optimized)]}
                    dump(EXP/'results/profiler.json', profiler)
                    for name, reference in [('dense128',build_bottleneck(n_taps=12,fuse_hidden=128)),
                                             ('fixed_mean_trend',build_depthrank2_bottleneck())]:
                        reference = reference.to(device).eval()
                        result['references'][name] = {
                            'ocbe':paired_measure({name:lambda:reference(taps)},1,config,device),
                            'subnet':paired_measure({name:lambda:student(reference(taps))},1,config,device)}
                    # Device changes must invalidate a prepared cache too.
                    scratch = copy.deepcopy(original).cpu().eval().cache_depth_basis_for_inference().to(device)
                    assert scratch._inference_depth_basis is None
                    checks['device_change_clears_cache'] = True
                    dump(EXP/'results/verification.json', checks)
            result['scenes'].append(entry)
            result['last_updated_at'] = stamp()
            dump(EXP/'results/measurements.json',result)
        result['finished_at'] = stamp()
        write_report(result)
        dump(EXP/'results/measurements.json',result)
        run.update(status='completed', ended_at=result['finished_at'], exit_code=0,
                   outputs=['results/measurements.json','results/verification.json','results/profiler.json',
                            'results/summary.csv','results/summary.md'])
        record.update(status='completed',result_summary='results/summary.md')
        print(json.dumps(dict(event='completed', **result['summary'])),flush=True)
    except Exception as exc:
        run.update(status='failed',ended_at=stamp(),reason=repr(exc))
        record['status']='failed'
        raise
    finally:
        dump(EXP/'record.json',record)


if __name__ == '__main__':
    main()
