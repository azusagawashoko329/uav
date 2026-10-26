"""Validate the two-GPU architecture audit and write Chinese tables."""
from __future__ import annotations

import csv
import hashlib
import json
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
    config = json.loads((EXP / 'config.json').read_text())
    by_id = {s['id']: s for s in config['variants']}
    gpus = [json.loads((EXP / f'results/gpu{i}.json').read_text()) for i in (0, 1)]
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
                    full_serial_fps=f['serial_fps_from_p50'], full_queued_fps=f['queued_throughput_fps'],
                    raw_diagnostic_core_ms=b.get('core_score_with_diagnostics', c)['latency_p50_ms'],
                    auc_percent=auc, auc_source=auc_source,
                    weight_source=r['weight_source']['kind'], experiment=s['experiment'],
                )
                rows.append(row)
    dump(EXP / 'results/summary.json', {'configuration': config, 'rows': rows,
                                       'teacher_parameters': gpus[0]['teacher_parameters'],
                                       'teacher_gflops': gpus[0]['teacher_flops']['flops']/1e9})
    with (EXP / 'results/summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    r0 = [r for r in rows if r['cuda_ordinal'] == 0 and r['batch'] == 1]
    lines = ['# 从原始学生到DGT：轻量化性能统一审计', '',
             '本次是现有模型的统一FP32推理重测，未重新训练或选择AUC。完整机器可读数据见summary.csv与gpu0.json/gpu1.json。', '',
             '## 基线和比较口径', '',
             '约89M的基线对应12个原始学生block、4个蒸馏tap、Dense512；精确OCBE＋学生参数为89,079,104。原始4block＋12个重建头＋Dense512为40,258,880。原始4block通过减少block数量轻量化，GL再改变block内部结构，两者需要分开列出。', '',
             '冻结教师为85,641,216个参数，在推理中仍然需要运行。以下core=OCBE＋学生；full=教师＋OCBE＋学生；两条pipeline的Latency/FPS均包含实际top-5异常分数。YOLO/KNN/行为融合不在本测试范围。', '',
             'FLOPs口径：1MAC=2FLOPs，仅计GEMM、卷积和attention的QK/AV。LayerNorm、激活、softmax、逐元素Q运算、top-k、偏置和数据搬运不在该FLOPs数值内，实际时延包含这些算子。它不是完整标量指令数。', '',
             '输入256×256，FP32，TF32/AMP关闭，batch=1/16；每项预热30次，3轮×100次。Latency为每次CUDA同步的墙钟p50，另报告p95；串行FPS=1000×batch/Latency。批量吞吐另用连续队列测量。', '',
             'DGT训练代码每次前向包含诊断量转CPU。主表删除这些诊断统计和拷贝，逐版本batch1/16验证输出一致；原始含诊断Latency保留附表。没有做模型剪枝、融合、量化、compile或精度替换。', '',
             '## 参数、FLOPs和Model Size', '',
             '大小为实际torch.save纯state_dict的序列化长度，内存中测量；排除优化器和训练metadata，MiB=2^20字节。CSV另给FP32 tensor payload、MB及字节口径。', '',
             '| 版本 | OCBE参数M | core参数M | full参数M | OCBE GFLOPs | 学生 GFLOPs | core GFLOPs | full GFLOPs | core大小MiB | full大小MiB | 历史AUC |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in r0:
        lines.append(f"| {r['label']} | {r['ocbe_parameters']/1e6:.6f} | {r['core_parameters']/1e6:.6f} | {r['full_parameters']/1e6:.6f} | {r['ocbe_gflops']:.4f} | {r['student_gflops']:.4f} | {r['core_gflops']:.4f} | {r['full_gflops']:.4f} | {r['core_pth_mib']:.2f} | {r['full_pth_mib']:.2f} | {r['auc_percent']:.4f} |")
    lines += ['', '历史12block基线results.csv仅保留两位小数，宏平均以该文件计算；这不是对话中84.81的复现，也不改写旧实验AUC。当前审计主要比较资源和速度。loss-only的ResidualDepth两版本推理结构相同，应把微小时延差看成测量波动，不能归因于loss降低推理复杂度。']
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
                  '| 版本 | OCBE ms | 学生 ms | core无分数 ms | core含分数 ms | 原始含CPU诊断 ms |',
                  '|---|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['cuda_ordinal'] == ordinal and r['batch'] == 1:
                lines.append(f"| {r['label']} | {r['ocbe_latency_ms']:.3f} | {r['student_latency_ms']:.3f} | {r['core_without_score_ms']:.3f} | {r['core_latency_ms']:.3f} | {r['raw_diagnostic_core_ms']:.3f} |")
        lines += ['', '独立教师耗时：' + '; '.join(f"batch={b} p50={t['latency_p50_ms']:.3f}ms" for b,t in gpu['teacher'].items()) + '。独立组件p50不能直接相加替代pipeline实测。']
    lines += ['', '## 机制解释和限制', '',
              '共享权重降低存储量，但同一投影被四组重复调用，所以DGT的OCBE FLOPs不会按71.5%的参数比例下降。Slot16才同时减少共享投影的每次计算，但历史AUC有所损失。', '',
              '缩窄学生维度减少block内近似二次的通道计算；12个重建头只随学生宽度线性下降，它们始终输出768维。实际Latency还受GPU并行度、kernel启动、内存访问、布局转换及评分影响，因此参数/FLOPs/FPS需要一起看。', '',
              '冻结教师与12个768维输出的评分成本在轻量版本中占比变大。所有版本都是桌面GPU测量，不能代替Jetson或无人机平台的部署测试；没有读盘、图像处理和H2D。', '',
              '有现存训练checkpoint的版本加载bike权重；老版本缺权重使用固定seed同构初始化。没有输入依赖的稀疏路由，所以本测试用于稠密架构资源比较；不声称重新验证这些初始化模型的AUC。', '',
              '所有学生FLOPs均与Linear/Conv/MHA独立公式交叉核验。teacher验证使用261个token（含5个prefix），学生使用256个patch；不存在把teacher当成256token而漏算前缀的情况。', '',
              '线性层FLOPs=2×N×Cin×Cout；MHA FLOPs=8×N×D²+4×N²×D；DWConv FLOPs=2×N×D×k²。teacher还包括patch embedding卷积。', '']
    (EXP / 'results/summary.md').write_text('\n'.join(lines))
    print(json.dumps({'variants': len(r0), 'rows': len(rows), 'summary': str(EXP/'results/summary.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
