# DAD-FSM（precomputed segmentation + score-inclusive follow-up）

- 论文：Fakhry et al., *Drone Video Anomaly Detection by Future Segmentation and Spatio-Temporal Relational Modeling*（repo：`Comparison method/DAD-FSM`）。
- 官方代码：`/data1/data8tb/liangyuxuan/Comparison method/DAD-FSM/`，HEAD `602fed9ac93e6a3f2a291e9b1f23bcca775df961`。
- 输入：4 帧 RGB + 4 帧分割图，224×224；`T=4, K=1, stride=1`。
- 参数：123.979M（ResNet50 随机初始化；官方预训练权重未随 repo 提供）。

## 1. 分割图来源

官方测试 `train_ourmodel_danomaly.py:85-86`：

```python
batch_pred = model(batch_data[:, :4], batch_seg[:, :4])
loss = loss_func_mse(seg_data_256[:, 4], batch_pred)
```

- `batch_seg` 来自 `../Drone_Anomaly_seg/...`，是**离线预计算**的分割图；
- repo 没有生成分割图的网络/脚本；`data_utils.py` 里只有可选的 OpenCV MOG 背景减除，且 DADFSM loader 没有使用它；
- 因此当前效率只覆盖检测网络，**不包含分割图生成**，标 `§ precomputed segmentation input`。

## 2. 旧结果（MSE-to-zero proxy）

旧 `benchmark_v2.json` 的 `_dadfsm` 用：

```python
return F.mse_loss(out, torch.zeros_like(out))
```

这不是官方 target；old latency 只能作 detector forward + zero-target proxy 的参考：

| 项 | 值 |
|---|---:|
| P50 | 12.684ms |
| FPS@BS1 | 78.5 |
| FPS@BS16 | 195.6 |

## 3. score-inclusive 结果

`comparison/benchmarks/bench_v2_scored.py` 传入与官方 target 同形状的随机张量（64×64；**不是真实标签**，只用于测量相同形状的评分计算量）并计算：

```python
pred = model(rgb, seg)
score = F.mse_loss(pred, future_seg_target)
```

写 `benchmark_v2_scored.json`，不覆盖旧 JSON：

| 项 | 旧 zero proxy | **score-inclusive target MSE** |
|---|---:|---:|
| P50 | 12.684ms | **17.032ms** |
| Mean | — | 17.157ms |
| P95 | — | 17.921ms |
| FPS@BS1 | 78.5 | **58.3** |
| FPS@BS16 | 195.6 | **293.7** |
| Params | 123.979M | 123.979M |
| FLOPs | 71.194G | 71.194G |

- 评分增加约 4.35ms P50（per-sample target MSE）。
- **BS16 A/B 核验**：同一 GPU、相同 BS16 输入、old/scored 交替 3 轮（`comparison/da/fairness_bs16_ab.json`，50 warmup + 3×500 batches）得到旧版 **280.8 FPS**、scored **280.7 FPS**（scored/old 时间比 1.000，二者等价）。旧 JSON 的 **195.6** 是低 GPU 时钟/其它负载下的 run，不能用来声称“评分更快”；scored 的 293.7 与本轮 A/B 的 280.7 差异同样属 run-to-run 时钟差异。引用 BS16 时必须带运行条件，主排序仍以 BS1 P50 为主。
- 71.194G 仍是 v2 MAC 计数；MSE 是 elementwise，不计入该 FLOPs 数字。
- **完整在线成本**还应包含分割图生成；repo 没有来源，暂不能测，当前只能报 detector-only。
- 文件：`benchmark_v2_scored.json`、`benchmark_v2_scored.log`；旧 `benchmark_v2.json` 保留。
