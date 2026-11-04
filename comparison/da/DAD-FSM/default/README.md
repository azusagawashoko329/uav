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

`comparison/benchmarks/bench_v2_scored.py` 传入真实的未来分割 target（64×64）并计算：

```python
pred = model(rgb, seg)
score = F.mse_loss(pred, future_seg_target)
```

写 `benchmark_v2_scored.json`，不覆盖旧 JSON：

| 项 | 旧 zero proxy | **score-inclusive target MSE** |
|---|---:|---:|
| P50 | 12.684ms | **13.576ms** |
| Mean | — | 13.630ms |
| P95 | — | 14.057ms |
| FPS@BS1 | 78.5 | **73.4** |
| FPS@BS16 | 195.6 | **289.1** |
| Params | 123.979M | 123.979M |
| FLOPs | 71.194G | 71.194G |

- 评分增加约 0.89ms P50；BS16 的上升来自本轮实测的 batch 调度差异，引用时以 BS1 P50 为主。
- 71.194G 仍是 v2 MAC 计数；MSE 是 elementwise，不计入该 FLOPs 数字。
- **完整在线成本**还应包含分割图生成；repo 没有来源，暂不能测，当前只能报 detector-only。
- 文件：`benchmark_v2_scored.json`、`benchmark_v2_scored.log`；旧 `benchmark_v2.json` 保留。
