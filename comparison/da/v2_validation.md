# v2 FLOPs 计数器验证记录（2026-11-04）

协议：[BENCHMARK_PROTOCOL.md](../BENCHMARK_PROTOCOL.md) v2。本轮先用 GANomaly（反卷积）、MemAE（3D 卷积/反卷积）、GLoRD V0（Attention）做小规模验证，再跑正式 50 warmup + 3×500。

## 1. 计数器实现

- 主计数器：`torch.utils.flop_counter.FlopCounterMode`（TorchDispatchMode），覆盖 `Conv2d/ConvTranspose2d/Conv3d/ConvTranspose3d`、`Linear`（addmm/mm）、`MatMul/BMM`、融合 SDPA。
- 补丁：`aten._native_multi_head_attention` 不在 PyTorch 注册表里，`comparison/benchmarks/flops_counter.py` 增加 `native_mha_flop`（QKV + QK + AV + 输出投影）。
- 独立公式：`hand_module_flops` 对 Conv/ConvTranspose/Linear/MultiheadAttention 逐模块手算；与主计数器**分开跑**，不重复累加。
- Profiler：`torch.profiler(with_flops=True)` 只比较**共同支持**的算子类别；`ConvTranspose`、`Conv3d`、融合 Attention、native MHA 等盲区跳过，不用 profiler 总数做验收。
- 去重：module hook / matmul / SDPA 包装是分开的计数路径；按 module + 调用 + shape 对齐，`flops_duplicate_check` 记录是否存在 `convolution` 与 `_convolution` 同图等风险。
- 单位：`1 MAC = 2 FLOPs`。

## 2. 玩具算子验证

| 算子 | 手算期望 FLOPs | 计数器 | 差异 |
|---|---:|---:|---:|
| `ConvTranspose2d(3→4,k3,s2,p1,op1)` | 13,824 | 13,824 | 0 |
| `Conv3d(2→3,k3,s1,p1)` | 82,944 | 82,944 | 0 |
| `ConvTranspose3d(2→3,k3,s2,p1,op1)` | 10,368 | 10,368 | 0 |
| `Linear(8→16)` | 1,024 | 1,024 | 0 |
| `MultiheadAttention(8,2)` | 2,560 | 2,560 | 0 |
| Conv2d + Linear + MHA 组合 | 16,896 | 16,896 | 0 |

组合模型：手写公式 3 个模块全部匹配（max rel diff = 0）；profiler 共同类别 `conv` 差异 0；`matmul` 因 native MHA 被 profiler 分解而跳过。

## 3. 实际模型验证

| 模型 | 输入 | 计数器 Full FLOPs | 手写模块 | 手写最大误差 | Profiler 总数 | Profiler 盲区 |
|---|---|---:|---:|---:|---:|---|
| GANomaly `NetG` | 1×3×256×256 | **16.427778048G** | 21 模块 | 0 | 10.951852032G | 漏 `ConvTranspose2d` 5.475926016G |
| MemAE | 1×3×16×256×256 | **70.808240128G** | 8 模块 | 0 | 1.050624G | 漏 `Conv3d/ConvTranspose3d` 69.757616128G |
| GLoRD V0 | 1×3×256×256 | **95.106764800G** | 92 模块 | 0 | 90.204910640G | native MHA 被部分分解；v1 少 4.901854160G |
| ANDT adapted 6-frame | 1×6×3×256×256 | **74.489796608G** | 16 模块 | 0 | 73.812554496G | ConvTranspose2d 盲区；matmul 共同子集 0% |

ANDT 的适配方式是 `de_dense(feat[:,0])` 预测下一帧（发布代码 decoder 只支持 1 帧），因此这一行是 **adapted**，不是 official；详见 [ANDT/default/README.md](ANDT/default/README.md)。

DINOv3 teacher 的融合 Attention 还做了独立公式核对：12 blocks × `4·L²·E`（L=261 tokens，E=768）= **2,511,212,544 FLOPs**，与计数器 `aten._scaled_dot_product_efficient_attention` 完全一致。

结论：

- GANomaly 的 v1 10.952G 只覆盖 Conv2d/Linear，**漏了反卷积**；v2 16.428G 与实际 G 推理图一致。
- MemAE 的 v1 0.065664G/帧 ×16 = 1.050624G 只覆盖 memory 的 matmul，**漏了全部 3D 卷积/反卷积**；v2 全窗口 70.808G。
- GLoRD V0 的 v1 90.205G 漏了 student native MHA；v2 95.107G，与 2026-10-26 独立审计的 95.1067648G 一致。
- 四个模型的所有手写模块与主计数器误差均为 0；没有 `hand_only` 模块；`flops_ops_missing=[]`，`flops_verified=true`。
- Profiler 总数只作参考，**没有**用于 Full FLOPs 验收；这避免了 MemAE 这类正确结果被不完整 profiler 判失败。

## 4. 正式测量

执行脚本：

```bash
CUDA_VISIBLE_DEVICES=GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6 \
python comparison/benchmarks/bench_v2_ours.py

CUDA_VISIBLE_DEVICES=GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6 \
python comparison/benchmarks/bench_v2_external.py --method ganomaly
# ... --method memae / mkd

CUDA_VISIBLE_DEVICES=GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6 \
python comparison/benchmarks/bench_v2_andt.py
```

- ANDT 为 adapted 6-frame：74.490G / 4.470ms / 223.6 FPS@BS1 / 210.4 FPS@BS16；不是官方代码。
- 预热 50 次；BS1 3×500 次（每次同步，P50/Mean/P95）；BS16 3×500 个 batch（只在每轮结束同步）。
- 结果只写对应方法/变体目录的 `benchmark_v2.json`；旧 `legacy_v1/benchmark_*.json` 未改动。
- 日志里的 `wrote ...` 是文件重排前的路径；文件已按方法目录归档，JSON/日志内容本身未改。
- 汇总见 [comparison_summary.md](comparison_summary.md) §3。

## 5. 复现坑（已修复，保留记录）

MKD 前两次尝试失败，未写入正式结果：

1. `make_arch('B', cfg_b, ...)` 把 `cfg` 传成了 list；上游接口要求 dict `{'A': ..., 'B': ...}`，报 `TypeError: list indices must be integers or slices, not str`。
2. 上游 `VGG.forward` 在 `target_layer=11` 调 `x.register_hook(...)`；在 `torch.no_grad()`/`inference_mode()` 下报 `RuntimeError: cannot register a hook on a tensor that doesn't require gradient`。v2 改为 hook-free 的 `_cloner_features()`，按相同 taps `[2,5,8,11,14,17,20,23,26,29,32,35,38]` 收集特征。

最终成功日志：`comparison/da/MKD/default/benchmark_v2.log`。
