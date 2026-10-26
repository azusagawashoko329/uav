"""Paired single-CUDA-process follow-up with recorded CPU affinity."""
from pathlib import Path
import importlib.util
import os
import subprocess

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / 'experiments/da/audit/20261026_rank2_projection_fused_s42/scripts/benchmark.py'
spec = importlib.util.spec_from_file_location('fused_rank2_audit', SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.EXP = EXP
audit.__file__ = __file__

original_setup = audit.setup


def setup(gpu):
    config, device, env = original_setup(gpu)
    affinity = sorted(os.sched_getaffinity(0))
    assert affinity == config['cpu_affinity'], (affinity, config['cpu_affinity'])
    env.update(cpu_affinity=affinity, pid=os.getpid(),
               gpu_topology=subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True),
               validation_scope='single CUDA ordinal; cross-device checks archived in launchfix audit')
    return config, device, env


original_report = audit.report


def report(result):
    original_report(result)
    path = EXP / 'results/summary.md'
    text = path.read_text().replace('batch1/16', 'batch1').replace('14组完整尺寸', '7组完整尺寸')
    text = text.replace('先stack十二层', '先按[tap,channel]顺序concat十二层')
    text = text.replace('无固定时钟/CPU绑核', '未锁定GPU/CPU时钟')
    text = text.replace('未锁定时钟或绑核', '固定CPU亲和性但未锁定时钟')
    text += '''
## 单CUDA进程与CPU亲和性

本轮与launchfix使用同一融合数学和预编译launcher。所有计时、数值和梯度检查只访问CUDA ordinal 0（RTX 5880 Ada）。跨卡/stream/cache-hit验证已在launchfix审计通过，不在计时进程再次访问第二张卡。进程由taskset绑定CPU20–23，属于5880对应的NUMA节点1；记录实际亲和性和nvidia-smi拓扑，子进程继承相同亲和性。模型、七场景Q、三种同轮对照、六轮排列和计时方式均不变。

此轮同时改变CUDA上下文准备和CPU亲和性，不能分别归因两者造成的绝对时延变化；主要结论只使用同轮配对差距。旧记录保留，不用跨轮FPS相除表示提速。
'''
    path.write_text(text)


audit.setup = setup
audit.report = report

if __name__ == '__main__':
    audit.main()
