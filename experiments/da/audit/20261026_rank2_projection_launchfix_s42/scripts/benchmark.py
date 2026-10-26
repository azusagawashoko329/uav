"""Final single-frame audit after removing repeated JIT launch dispatch."""
from pathlib import Path
import importlib.util

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT/'experiments/da/audit/20261026_rank2_projection_fused_s42/scripts/benchmark.py'
spec = importlib.util.spec_from_file_location('fused_rank2_audit', SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.EXP = EXP
audit.__file__ = __file__

original_verify = audit.verify_fused_contract


def verify():
    checks = original_verify(audit.torch.device('cuda:0'))
    # CUDA contexts and streams must remain correct when a model is used on a
    # different device from the process's current default device.
    from model.depthrank_projection import project_depth_rank2
    torch = audit.torch
    with torch.inference_mode():
        current = torch.cuda.current_device()
        for device in [torch.device('cuda:0'), torch.device('cuda:1')]:
            q = torch.randn(2,3,device=device)
            taps = [torch.randn(1,7,17,device=device) for _ in range(12)]
            for _ in range(2):
                a,b = audit.loop_project(taps,q),project_depth_rank2(taps,q)
                torch.testing.assert_close(a,b,atol=0,rtol=0)
                assert torch.cuda.current_device()==current
        stream = torch.cuda.Stream(device=0)
        with torch.cuda.stream(stream):
            q = torch.randn(2,3,device='cuda:0')
            taps = [torch.randn(1,7,17,device='cuda:0') for _ in range(12)]
            torch.testing.assert_close(audit.loop_project(taps,q),project_depth_rank2(taps,q),atol=0,rtol=0)
    checks.update(launcher_cache_hits_bitwise_equal=True,cross_device_launch_correct=True,
                  current_device_preserved=True,nondefault_stream_correct=True)
    return checks


def verify_with_device(device):
    return verify()


original_report = audit.report


def report(result):
    original_report(result)
    path=EXP/'results/summary.md'
    s=path.read_text().replace('batch1/16','本轮仅batch1').replace('14组完整尺寸','7组完整尺寸')
    s=s.replace('先stack十二层','先按[tap,channel]顺序concat十二层')
    s+='''
## 缩短主机启动路径的复核

上一轮融合投影已使OCBE kernel数从57减到18，CUDA kernel时长总和约97us，与Dense128约100us接近，但墙钟子网仍慢约0.095ms。本轮仅优化融合执行的启动和打包：用等价flat concat替代stack，按(device,total,channels,groups)缓存已编译kernel的launcher，避免每帧重复JIT签名构造与设备分派。缓存不保存Tensor指针或Q值，始终传入当前权重对应的Q。

本轮聚焦batch1，六种顺序轮换的配对协议、七场景最终Q、参考模型、输出与梯度检查均复用上轮；没有重新训练或改Q/学生/loss。额外核对CUDA0/1切换、launcher cache命中及非默认stream，不改变调用方的当前设备。Profiler仍由计时结束后的独立子进程采集。
'''
    path.write_text(s)


audit.verify_fused_contract=verify_with_device
audit.report=report

if __name__=='__main__':audit.main()
