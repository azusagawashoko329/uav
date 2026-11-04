# HSTforU（Avenue/PVT2-b2 structure；DA 配置未确认）

- 论文：Le et al., *HSTforU: anomaly detection in aerial and ground-based videos with hierarchical spatio-temporal transformer for U-net*, Applied Intelligence 55(4), 2025, DOI `10.1007/s10489-024-06042-4`。
- 代码：`/data1/data8tb/liangyuxuan/Comparison method/HSTforU/`（HEAD `9548473`）。
- 变体：**当前假设**为 Avenue/PVT2-b2（`models/HSTforU_Avenue.py`），PVT2 encoder + TimeSformer + UNet decoder。
- 输入：4 帧 `256×256` RGB；输出下一帧；`T=4, K=1, stride=1`。
- 权重：random init（无官方 checkpoint）；FLOPs/latency 与权重无关。

> **DA 配置未确认**：repo 只有 `configs/scripts/{ped2,avenue,shanghaitech}.yaml`，没有 Drone-Anomaly 的 yaml；`models/__init__.py` 默认还是 Ped2/PVT2-b1。这个 148.970M 是 Avenue/PVT2-b2 结构，不能自动称为 DA 论文配置。`paper_matched=false`。

## 依赖处理

HSTforU 代码使用 mmcv 1.x API，但当前 Python 3.13 环境只能装 `mmcv-lite 2.2.0`。已加兼容 shim：

- `mmcv.runner.load_checkpoint`：no-op（不加载 checkpoint）；
- `mmcv.utils.parrots_wrapper._BatchNorm`：映射到 `torch.nn.modules.batchnorm._BatchNorm`；
- `mmcv.cnn.UPSAMPLE_LAYERS`：轻量 registry；
- `build_upsample_layer`：`DeconvModule` 直接实例化。

这是**适配层**，不是上游 mmcv 1.x 原环境；结果标 `adapted`，与原版环境可能有细微差异。

## 实测（RTX 5880 Ada，FP32，50 warmup + 3×500）

| 项 | 值 |
|---|---:|
| Inference / Trainable Params | 148.970M / 148.970M |
| FLOPs | 168.209G / output score |
| Latency@BS1 P50 | 64.258ms |
| FPS@BS1 | 15.5 |
| FPS@BS16 | —（batch 标量无效） |
| FLOPs verified | yes（手写 216 模块误差 0；profiler matmul 0%） |
| Status | `partial/unverified`；`paper_matched=false`（Avenue/PVT2-b2 假设，评分是 batch 标量代理） |

- 结果：`comparison/da/HSTforU/default/benchmark_v2.json`
- benchmark：`comparison/benchmarks/bench_v2_extra.py --method hstforu`
- **评分口径**：官方 `test.py` 对 `target` 计算 PSNR，再用 `anomaly_score` 做逐视频 min-max；当前 `_hstforu` 用 `F.mse_loss(out, zeros)` 代理，既不是 target PSNR 也不是逐视频归一化。因此当前 latency 只能作网络前向参考，不能称为完整 anomaly-score latency。
- **输出形状**：当前 `fn` 返回 **一个 batch 标量**；BS1 有效，BS16 原始批吞吐是 batch 聚合分数，不能按 16 个每样本分数计，主表已置 `—`。
- AUC 仍引用用户表/论文的 66.96，不把这里的效率数字当成官方 AUC 配置。
