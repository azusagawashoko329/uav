"""Validate the two-GPU architecture audit and write Chinese tables."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[5]
SCENES = ['bike', 'crossroads', 'farmland', 'highway', 'railway', 'solar', 'vehicle']


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def historical_auc(spec):
    exp = ROOT / spec['experiment']
    paths = [exp / 'results' / (s + '.json') for s in SCENES]
    if all(p.is_file() for p in paths):
        return mean(json.loads(p.read_text())['auc_percent'] for p in paths), 'seven scene JSON'
    if spec['id'] == 'baseline12':
        # Preserve the historical result rather than substituting a conversational
        # 84.81 estimate. No AUC is re-evaluated during this performance audit.
        with (exp / 'results.csv').open() as f:
            rows = list(csv.DictReader(f))
        return mean(float(r['topk_AUC']) for r in rows), 'historical results.csv rounded to 2dp; runtime source unknown'
    return None, 'unavailable'


def main():
    provenance = {
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'started_at': datetime.now(timezone.utc).isoformat(),
        'producer_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    print(json.dumps({'event': 'started', **provenance}), flush=True)
    config = json.loads((EXP / 'config.json').read_text())
    by_id = {s['id']: s for s in config['variants']}
    gpus = [json.loads((EXP / f'results/gpu{i}.json').read_text()) for i in (0, 1)]
    for gpu in gpus:
        if 'finished_at' not in gpu:
            raise ValueError('Benchmark process has not finished')
    gpu1_by_id = {r['id']: r for r in gpus[1]['variants']}
    for r in gpus[0]['variants']:
        other = gpu1_by_id[r['id']]
        for key in ('core_parameters', 'full_parameters', 'core_flops', 'full_flops', 'model_size'):
            if r[key] != other[key]:
                raise ValueError(f'Static architecture metrics differ across GPUs: {r["id"]} {key}')
    rows = []
    for gpu in gpus:
        if {r['id'] for r in gpu['variants']} != set(by_id):
            raise ValueError('Incomplete variant coverage')
        for r in gpu['variants']:
            s = by_id[r['id']]
            if r['core_parameters'] != s['expected_core_parameters']:
                raise ValueError('Parameter mismatch')
            if r['full_parameters'] != r['core_parameters'] + gpu['teacher_parameters']:
                raise ValueError('Teacher missing from full parameter count')
            auc, auc_source = historical_auc(s)
            for batch in config['batches']:
                b = r['batches'][str(batch)]
                if b['clean_output_max_abs_diff'] > 1e-6:
                    raise ValueError('Diagnostic removal changed output')
                c, f = b['core_score'], b['full_score']
                row = dict(
                    id=r['id'], label=r['label'], cuda_ordinal=gpu['environment']['cuda_ordinal'],
                    gpu=gpu['environment']['cuda_device_name'], batch=batch,
                    ocbe_parameters=r['ocbe_parameters'], student_parameters=r['student_parameters'],
                    core_parameters=r['core_parameters'], full_parameters=r['full_parameters'],
                    ocbe_gflops=r['ocbe_flops']['flops']/1e9,
                    student_gflops=r['student_flops']['flops']/1e9,
                    core_gflops=r['core_flops']/1e9, full_gflops=r['full_flops']/1e9,
                    core_gmac=r['core_flops']/2e9,
                    core_pth_mib=r['model_size']['core_pth_bytes']/2**20,
                    core_payload_mib=r['model_size']['core_state_tensor_bytes']/2**20,
                    full_pth_mib=r['model_size']['full_pth_bytes']/2**20,
                    full_payload_mib=r['model_size']['full_state_tensor_bytes']/2**20,
                    core_pth_mb=r['model_size']['core_pth_bytes']/1e6,
                    full_pth_mb=r['model_size']['full_pth_bytes']/1e6,
                    ocbe_latency_ms=b['ocbe']['latency_p50_ms'],
                    student_latency_ms=b['student']['latency_p50_ms'],
                    core_without_score_ms=b['core']['latency_p50_ms'],
                    core_latency_ms=c['latency_p50_ms'], core_p95_ms=c['latency_p95_ms'],
                    core_serial_fps=c['serial_fps_from_p50'], core_queued_fps=c['queued_throughput_fps'],
                    full_latency_ms=f['latency_p50_ms'], full_p95_ms=f['latency_p95_ms'],
                    full_round_p50_min_ms=min(f['round_p50_ms']),
                    full_round_p50_max_ms=max(f['round_p50_ms']),
                    full_serial_fps=f['serial_fps_from_p50'], full_queued_fps=f['queued_throughput_fps'],
                    raw_diagnostic_core_ms=b.get('core_score_with_diagnostics', c)['latency_p50_ms'],
                    auc_percent=auc, auc_source=auc_source,
                    weight_source=r['weight_source']['kind'], experiment=s['experiment'],
                )
                rows.append(row)
    dump(EXP / 'results/summary.json', {'configuration': config, 'report_provenance': provenance, 'rows': rows,
                                       'teacher_parameters': gpus[0]['teacher_parameters'],
                                       'teacher_gflops': gpus[0]['teacher_flops']['flops']/1e9})
    with (EXP / 'results/summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    r0 = [r for r in rows if r['cuda_ordinal'] == 0 and r['batch'] == 1]
    lines = ['# 从原始学生到DGT：轻量化性能统一审计', '',
             '本次是现有模型的统一FP32推理重测，未重新训练或选择AUC。完整机器可读数据见summary.csv与gpu0.json/gpu1.json。', '',
             '![参数与每帧计算量：教师和可训练子网](parameters_flops.png)', '',
             '![两张GPU的单帧时延](latency_comparison.png)', '',
             '## 基线和比较口径', '',
             '约89M的基线对应12个原始学生block、4个蒸馏tap、Dense512；精确OCBE＋学生参数为89,079,104。原始4block＋12个重建头＋Dense512为40,258,880。原始4block通过减少block数量轻量化，GL再改变block内部结构，两者需要分开列出。', '',
             '冻结教师为85,641,216个参数，在推理中仍然需要运行。以下core=OCBE＋学生；full=教师＋OCBE＋学生；两条pipeline的Latency/FPS均包含实际top-5异常分数。按用户最新定义，蒸馏外观分支即整个模型；行为分支已取消。', '',
             '历史基线只重建、评分4个tap；其余架构保留12个重建头并评分12层。这里比较各版本原本的推理流程，不把它们改成相同输出头；因此参数/FLOPs/时延变化同时包含已落地的输出接口变化，不能全部归因于block数目。组件附表另列不含评分的网络前向。', '',
             'FLOPs口径：1MAC=2FLOPs，仅计GEMM、卷积和attention的QK/AV。LayerNorm、激活、softmax、逐元素Q运算、top-k、偏置和数据搬运不在该FLOPs数值内，实际时延包含这些算子。它不是完整标量指令数。', '',
             '输入256×256，FP32，TF32/AMP关闭，batch=1/16；每项预热30次，3轮×100次。Latency为每次CUDA同步的墙钟p50，另报告p95；串行FPS=1000×batch/Latency。批量吞吐另用连续队列测量。', '',
             '部分DGT变体训练代码每次前向包含诊断量转CPU。主表删除这些诊断统计和拷贝，逐版本batch1/16验证输出一致；原始含诊断Latency保留附表。没有做模型剪枝、融合、量化、compile或精度替换。', '',
             '## 参数、FLOPs和Model Size', '',
             '大小为实际torch.save纯state_dict的序列化长度，内存中测量；排除优化器和训练metadata，MiB=2^20字节。CSV另给FP32 tensor payload、MB；原始GPU JSON保留字节数。它是部署权重存储量，不等于运行显存。', '',
             '| 版本 | OCBE参数M | core参数M | full参数M | OCBE GFLOPs | 学生 GFLOPs | core GFLOPs | full GFLOPs | core大小MiB | full大小MiB | 历史AUC |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in r0:
        lines.append(f"| {r['label']} | {r['ocbe_parameters']/1e6:.6f} | {r['core_parameters']/1e6:.6f} | {r['full_parameters']/1e6:.6f} | {r['ocbe_gflops']:.4f} | {r['student_gflops']:.4f} | {r['core_gflops']:.4f} | {r['full_gflops']:.4f} | {r['core_pth_mib']:.2f} | {r['full_pth_mib']:.2f} | {r['auc_percent']:.4f} |")
    lines += ['', '历史12block基线results.csv仅保留两位小数，宏平均以该文件计算；这不是对话中84.81的复现，也不改写旧实验AUC。当前审计主要比较资源和速度。loss-only的ResidualDepth两版本推理结构相同，应把微小时延差看成测量波动，不能归因于loss降低推理复杂度。']
    lines += ['', '### 模型重建与权重来源', '',
              '参数结构按config登记的源实验重建，逐个断言精确参数量。下表的初始化模型仅用于结构资源审计，不能把其输出解释为已训练模型的检测能力；历史AUC来自原实验。完整checkpoint SHA256、路径在gpu0.json。', '',
              '| 版本 | 本次使用权重 | checkpoint SHA256前12位 |',
              '|---|---|---|']
    for r in gpus[0]['variants']:
        w = r['weight_source']
        source = '已训练bike checkpoint' if w['kind']=='trained_bike_checkpoint' else 'seed42同构初始化（旧权重本机缺失）'
        lines.append(f"| {r['label']} | {source} | {w.get('sha256', '—')[:12]} |")
    selected = ['baseline12', 'original4', 'gl768', 'gl192_dense512',
                'gl192_dense128', 'rank2_random', 'dgt', 'slot16']
    selection = {r['id']: r for r in r0}
    baseline = selection['baseline12']
    lines += ['', '## 主要轻量化路径：相对89M基线', '',
              '时延加速比采用RTX 5880 Ada、batch=1、含异常评分的pipeline实测；加速比大于1才表示更快。', '',
              '| 版本 | core参数减少 | core FLOPs减少 | full参数减少 | full FLOPs减少 | core加速比 | full加速比 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for k in selected:
        r = selection[k]
        reductions = [(1-r[key]/baseline[key])*100 for key in
                      ('core_parameters', 'core_gflops', 'full_parameters', 'full_gflops')]
        lines.append(f"| {r['label']} | " + ' | '.join(f'{v:.2f}%' for v in reductions)
                     + f" | {baseline['core_latency_ms']/r['core_latency_ms']:.2f}× | {baseline['full_latency_ms']/r['full_latency_ms']:.2f}× |")
    rank, dgt, slot = (selection[k] for k in ('rank2_random', 'dgt', 'slot16'))
    lines += ['',
              f"从共享可学习Rank2到DGT：OCBE参数减少{(1-dgt['ocbe_parameters']/rank['ocbe_parameters'])*100:.2f}%，但OCBE主要算术FLOPs变化{(dgt['ocbe_gflops']/rank['ocbe_gflops']-1)*100:+.2f}%；RTX 5880单帧core时延{rank['core_latency_ms']:.3f}→{dgt['core_latency_ms']:.3f}ms，full时延{rank['full_latency_ms']:.3f}→{dgt['full_latency_ms']:.3f}ms。减少共享权重不能直接等价为计算量或推理时延同比下降。",
              '',
              f"Slot16的core参数为{slot['core_parameters']/1e6:.4f}M、core计算为{slot['core_gflops']:.4f}GFLOPs；完整模型仍为{slot['full_parameters']/1e6:.4f}M、{slot['full_gflops']:.4f}GFLOPs。冻结教师本身即{gpus[0]['teacher_parameters']/1e6:.4f}M、{gpus[0]['teacher_flops']['flops']/1e9:.4f}GFLOPs。", '']
    for gpu in gpus:
        ordinal = gpu['environment']['cuda_ordinal']
        lines += ['', f"## {gpu['environment']['cuda_device_name']}（CUDA {ordinal}）", '',
                  '### batch=1：单帧Latency和串行FPS', '',
                  '| 版本 | core p50 ms | core p95 ms | core FPS | full p50 ms | full p95 ms | full FPS |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['cuda_ordinal'] == ordinal and r['batch'] == 1:
                lines.append(f"| {r['label']} | {r['core_latency_ms']:.3f} | {r['core_p95_ms']:.3f} | {r['core_serial_fps']:.1f} | {r['full_latency_ms']:.3f} | {r['full_p95_ms']:.3f} | {r['full_serial_fps']:.1f} |")
        lines += ['', '### batch=16：吞吐', '',
                  '下列Latency是整个16帧批次；队列FPS单独实测。它不能当成batch=1的响应速度。', '',
                  '| 版本 | core批次p50 ms | core队列FPS | full批次p50 ms | full队列FPS |',
                  '|---|---:|---:|---:|---:|']
        for r in rows:
            if r['cuda_ordinal'] == ordinal and r['batch'] == 16:
                lines.append(f"| {r['label']} | {r['core_latency_ms']:.3f} | {r['core_queued_fps']:.1f} | {r['full_latency_ms']:.3f} | {r['full_queued_fps']:.1f} |")
        lines += ['', '### batch=1组件耗时与诊断开销', '',
                  '含诊断版本另外实测原始forward；本身没有诊断的版本，最后一列沿用core含分数数值，没有额外测一遍相同函数。', '',
                  '| 版本 | OCBE ms | 学生 ms | core无分数 ms | core含分数 ms | 原始含CPU诊断 ms |',
                  '|---|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['cuda_ordinal'] == ordinal and r['batch'] == 1:
                lines.append(f"| {r['label']} | {r['ocbe_latency_ms']:.3f} | {r['student_latency_ms']:.3f} | {r['core_without_score_ms']:.3f} | {r['core_latency_ms']:.3f} | {r['raw_diagnostic_core_ms']:.3f} |")
        lines += ['', '独立教师耗时：' + '; '.join(f"batch={b} p50={t['latency_p50_ms']:.3f}ms" for b,t in gpu['teacher'].items()) + '。独立组件p50不能直接相加替代pipeline实测。']
    lines += ['', '## 机制解释和限制', '',
              '令每patch四组输入各为1536维，DGT两条共享投影的主乘法量为4×1536×64×2=786,432 MAC/patch；Dense Rank2首层为6144×128=786,432 MAC/patch，二者相等。DGT还需256→64输出投影及跨组融合，而Dense Rank2末层为128→64。共享投影减少权重存储，但不能省去四组各自的前向计算。Slot16降低组内投影宽度，才减少这部分乘法量；历史AUC也有所损失。', '',
              '缩窄学生维度减少block内近似二次的通道计算；12个重建头只随学生宽度线性下降，它们始终输出768维。实际Latency还受GPU并行度、kernel启动、内存访问、布局转换及评分影响，因此参数/FLOPs/FPS需要一起看。', '',
              '冻结教师与12个768维输出的评分成本在轻量版本中占比变大。所有版本都是桌面GPU测量，不能代替Jetson或无人机平台的部署测试；没有读盘、图像处理和H2D。', '',
              '原始GPU JSON的peak_full_allocated_bytes记录包含benchmark保留的教师中间特征、原实现与去诊断副本等，不能当作独立部署模型的峰值显存；本报告不使用该字段评估显存。Model Size专指纯权重存储。', '',
              '有现存训练checkpoint的版本加载bike权重；老版本缺权重使用固定seed同构初始化。没有输入依赖的稀疏路由，所以本测试用于稠密架构资源比较；不声称重新验证这些初始化模型的AUC。', '',
              '所有学生FLOPs均与Linear/Conv/MHA独立公式交叉核验。teacher验证使用261个token（含5个prefix），学生使用256个patch；不存在把teacher当成256token而漏算前缀的情况。', '',
              'PyTorch 2.1在inference_mode下可能绕过FLOP分派，计数为零。因此只在统计FLOPs时开启梯度分派、执行一次前向且不执行反向；实际时延全部在eval＋inference_mode下测量。两张GPU的静态参数、运算量与序列化大小逐项核对一致。', '',
              '教师统一提取全部12层patch后，基线选择第6/8/10/12层，其余方法使用全部12层。统一提取保留了同样的教师计算图，基线可能多保留少量未用中间激活。teacher本身不提前退出。', '',
              '本次两张GPU同时独立测量；CPU各用4线程，但没有固定时钟或CPU绑核。三轮重复用于反映短时波动，并不能把不足几个百分点的速度差自动解释为结构收益；CSV包含完整pipeline三轮median范围。', '',
              '线性层FLOPs=2×N×Cin×Cout；MHA FLOPs=8×N×D²+4×N²×D；DWConv FLOPs=2×N×D×k²。teacher还包括patch embedding卷积。', '']
    (EXP / 'results/summary.md').write_text('\n'.join(lines))
    # Standalone figures separate fixed teacher cost and trainable subnet cost.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    yy = np.arange(len(selected))
    for ax, core_key, teacher_cost, label in (
        (axes[0], 'core_parameters', gpus[0]['teacher_parameters']/1e6, 'Parameters (M)'),
        (axes[1], 'core_gflops', gpus[0]['teacher_flops']['flops']/1e9, 'Major arithmetic GFLOPs / frame'),
    ):
        core = [selection[k][core_key] / (1e6 if core_key=='core_parameters' else 1) for k in selected]
        ax.barh(yy, core, label='OCBE + student', color='#de854b')
        ax.barh(yy, [teacher_cost]*len(yy), left=core, label='Frozen teacher', color='#8fabc6')
        ax.set_yticks(yy, selected); ax.set_xlabel(label); ax.grid(axis='x', alpha=.2)
        ax.legend(fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title('Deployment weights include the teacher')
    axes[1].set_title('Per-frame arithmetic includes the teacher')
    fig.tight_layout(); fig.savefig(EXP/'results/parameters_flops.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, gpu in zip(axes, gpus):
        ordinal = gpu['environment']['cuda_ordinal']
        rr = {r['id']: r for r in rows if r['cuda_ordinal']==ordinal and r['batch']==1}
        yy = np.arange(len(selected))
        ax.barh(yy-.18, [rr[k]['core_latency_ms'] for k in selected], height=.36, label='OCBE + student + score')
        ax.barh(yy+.18, [rr[k]['full_latency_ms'] for k in selected], height=.36, label='Entire model + score')
        ax.set_yticks(yy, selected); ax.set_xlabel('Synchronized batch=1 p50 latency (ms)')
        ax.set_title(gpu['environment']['cuda_device_name']); ax.legend(fontsize=8); ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis(); fig.tight_layout()
    fig.savefig(EXP/'results/latency_comparison.png', dpi=180); plt.close(fig)
    print(json.dumps({'event': 'completed', 'finished_at': datetime.now(timezone.utc).isoformat(),
                      'variants': len(r0), 'rows': len(rows), 'summary': str(EXP/'results/summary.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
