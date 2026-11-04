# ANDT（adapted 6-frame measured；official code still partial）

- 论文：Jin et al., *Anomaly Detection in Aerial Videos With Transformers*, IEEE TGRS 2022（arXiv:2209.13363）；Drone-Anomaly 数据集与 ANDT baseline。
- 本地代码：`/data1/data8tb/liangyuxuan/Comparison method/ANDT/`（HEAD `f2e8c4f70123`；`record.json.source.local_path` 指向这里）。
- 对比条目目录只放记录，不复制上游代码。
- v2 结果：`comparison/da/ANDT/default/benchmark_v2.json`（**adapted 6-frame**，不是官方代码）。

## “论文的 6 帧模型”是什么

ANDT 是**时序**方法，不是单帧分类器：

```text
输入窗口：frame[t-5], frame[t-4], ..., frame[t]   （6 个连续帧）
   │
   ├─ patch embedding + 时空位置编码
   ├─ Transformer encoder（2 层、6 heads）
   └─ decoder 预测 frame[t+1]
异常分数 = 预测帧与真实 frame[t+1] 的差异
```

- 一个 6 帧窗口只输出 **1 个**帧级分数（预测的下一帧），不是 6 个输出帧；v2 计量记为 `T=6, K=1, stride=1`；
- 论文 Table IV 的三种设置：1 帧重建自己、6 帧重建自己、**6 帧预测下一帧（最终 ANDT）**；
- 论文 Table V 的输入帧数 ablation：2→6 帧逐步变好，超过 6 变差，**最终选 6 帧**；
- 因此主表里 ANDT 的 Input 写 **256×256×6**，不能按单帧 256×256×1 算 FLOPs/latency。

## 参数（按代码结构）

- `b16` 配置：`num_layers=2`、`heads=6`、`patch=16`、`emb_dim=768`。
- `num_frames=6`（论文 ablation 最佳）、`mlp=4096`：**70.347M** total/trainable。
- `num_frames=6`、`mlp=1536`（config 默认）：**62.478M**。
- `num_frames=1`、`mlp=4096`：**69.364M**。

## v2 adapted 6-frame 结果

发布代码的 `Decoder.forward` 只支持 1 帧，不能直接跑论文的 6 帧输入。v2 使用最小适配：

1. 去掉 `pdb.set_trace()` 和 forward 里的硬编码 `.cuda()`；
2. `Decoder.forward` 改为 `self.decoder(self.de_dense(x[:, 0]))`（`de_dense` 是代码里已定义但被注释掉的 768→256×16×16 投影）；
3. 输入 `(B, 6, 3, 256, 256)`，输出预测的下一帧 `(B, 3, 256, 256)`。

实测（RTX 5880 Ada，FP32，50 warmup + 3×500）：

| 项 | 值 |
|---|---:|
| Input | 256×256×6 |
| T / K / stride | 6 / 1 / 1 |
| Inference Params | 70.347M |
| Training Trainable | 70.347M |
| FLOPs | **74.490G / output score** |
| Latency@BS1 P50 | **5.015ms / output score** |
| FPS@BS1 | **199.4** |
| FPS@BS16 | **197.5** |
| FLOPs verified | yes（hand 16 模块误差 0；profiler matmul 0%；ConvTranspose 盲区跳过） |

- 适配器：`comparison/da/ANDT/default/scripts/andt_adapted_model.py`
- benchmark：`comparison/benchmarks/bench_v2_andt.py`
- 结果：`comparison/da/ANDT/default/benchmark_v2.json`
- 权重状态：random init；官方 checkpoint 不可用。FLOPs/latency 对权重不敏感。
- **评分口径**：当前 `bench_v2_andt.py` 用 `F.mse_loss(out, zeros)` 作为产生标量的代理，不是官方 `MSE(pred, true_next_frame)`；因此 latency 只能作 adapted/近似的评分耗时，不能称为官方 anomaly-score latency。
- **论文 decoder 与本地代码不一致**：论文（arXiv:2209.13363）写的是两个 FC 层把特征升到 `8×8×512`，再用 5 层卷积/上采样重建 `256×256×3`；本地 active 代码是 `view(...,256,16,16)`，`de_dense` 是单层 `768→256*16*16`，两者都不是论文的 `8×8×512` 两 FC 路径。因此 `de_dense(feat[:,0])` 只是能跑通的最小适配，`paper_matched=false`。
- `frame_coverage=1.0` 是近似值：滑动窗口预测下一帧时，视频开头 5 帧没有预测分数。

## 官方代码为什么仍不能冒充

发布代码 `ANDT/ANDT/model.py` 的 `forward` 是未整理变体：

- `pdb.set_trace()` 会中断；
- forward 里 `.cuda()` 硬编码；
- decoder `x.view(x.size(0), 256, 16, 16)` 只对应 **1 帧**的 256 个 patch；`num_frames=6` 时 `feat` 是 `1 + 256*6 = 1537` 个 token，`view` 形状不匹配；
- `de_dense` 路径被注释；论文 decoder 是 `2 FC → 8×8×512 → 5 conv/upsample`，而本地 `de_dense` 是 `768→256×16×16` 单层投影，**adapted 版本不是论文 decoder 的逐层复现**。

因此：

- **adapted 数字不能当 official**；
- AUC 仍引用 ANDT 论文/用户表的官方 6 帧模型（68.64），不能把这里的效率数字与官方 AUC 说成同一实现；
- 若后续拿到作者完整 6 帧代码/decoder，需要重新跑并替换这一行。
