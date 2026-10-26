"""Repeat the paired Q-cache audit without starting a profiler in this process."""
from pathlib import Path
import hashlib
import importlib.util
import json

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / 'experiments/da/audit/20261026_rank2_cacheq_fp32_s42'
spec = importlib.util.spec_from_file_location('cacheq_matched_audit', SOURCE/'scripts/benchmark.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
# The original implementation stays immutable. Only profiling is replaced by
# an external evidence reference; every numerical check and timing loop is reused.
audit.EXP = EXP
audit.__file__ = __file__


def external_profile(model, taps, device, steps=10):
    path = SOURCE/'results/profiler.json'
    name = 'uncached' if model._inference_depth_basis is None else 'cached'
    return dict(status='external_reference', source=str(path.relative_to(ROOT)),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                reason='No profiler is started in the timing process',
                previously_measured_profile=json.loads(path.read_text())[name])


original_report = audit.write_report


def write_report(result):
    original_report(result)
    path = EXP/'results/summary.md'
    with path.open('a') as f:
        f.write('\n## 计时与Profiler分离的复核\n\n'
                '初轮在bike单帧计时后启动Profiler，随后同进程单帧绝对耗时抬高。'
                '本轮完全不启动Profiler，复用完全相同的六轮AB/BA计时、原始最终Q、'
                '输出及梯度验证和参考模型，只从上轮独立进程引用kernel计数证据。'
                '缓存收益以本轮配对数据为准；上轮绝对时延不用于填表。\n')


audit.profile_ocbe = external_profile
audit.write_report = write_report

if __name__ == '__main__':
    audit.main()
