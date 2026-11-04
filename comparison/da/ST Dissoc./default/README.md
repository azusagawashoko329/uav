# ST Dissoc.（score-inclusive follow-up）

- 论文：Chang et al., *Video Anomaly Detection with Spatio-Temporal Dissociation*（repo：`Comparison method/ST Dissoc.`）。
- 官方代码：`/data1/data8tb/liangyuxuan/Comparison method/ST Dissoc./`，HEAD `1206c2a7c4c4e0b47c21c498fadb8e1e3973dd56`。
- 输入：5 帧 256×256 RGB，按通道拼接为 `[B,15,256,256]`；`T=5, K=1, stride=1`。
- 参数：6.120M；`padding=False` 按 0 兼容处理。

## 1. 旧结果（stage-E forward only）

旧 `benchmark_v2.json` 只执行：

```python
self.m(x, None, ['E'])
# -> static_decoder, pred_decoder, cluster_map
```

没有执行官方 eval 的最终评分，因此旧 latency/FPS 只能读作 **cluster-map-only**：

| 项 | 值 |
|---|---:|
| P50 | 3.668ms |
| FPS@BS1 | 269.7 |
| FPS@BS16 | 288.2 |

## 2. 官方评分路径

官方 `arch/solver/Solver_PredRes.py:185-220`：

```python
static_decoder, pred_decoder, loss_cluster_map = model(batch_in, None, ['E'])
pred_recon = pred_decoder + static_decoder
pred_target = static_in + motion_target
loss_pixelwise_cl_re = loss_map(
    torch.mean((pred_recon - pred_target) ** 2, [1], keepdim=True),
    loss_cluster_map,
)
score = reciprocal_metric(loss_pixelwise_cl_re)   # 1 / loss
```

`loss_map` 在 `arch/module/eval_utils.py:48`：`mean(loss_recon_map * loss_cluster_map, [1,2,3])`。

## 3. score-inclusive 结果

`comparison/benchmarks/bench_v2_scored.py` 把上述 `pred_recon`、`pred_target`、`loss_map`、reciprocal 全部放进 forward，写 `benchmark_v2_scored.json`，不覆盖旧 JSON：

| 项 | 旧 forward-only | **score-inclusive** |
|---|---:|---:|
| P50 | 3.668ms | **4.280ms** |
| Mean | — | 4.322ms |
| P95 | — | 4.458ms |
| FPS@BS1 | 269.7 | **231.4** |
| FPS@BS16 | 288.2 | **452.3** |
| Params | 6.120M | 6.120M |
| FLOPs | 60.612G | 60.612G |

- 评分增加约 0.61ms P50 / BS1 下降约 14%。
- **BS16 A/B 核验**：同一 GPU、相同 BS16 输入、old/scored 交替 3 轮（`comparison/da/fairness_bs16_ab.json`，50 warmup + 3×500 batches）得到旧版 **426.6 FPS**、scored **422.2 FPS**（scored/old 时间比 1.010，即评分图在 BS16 几乎不增加成本）。因此旧 JSON 的 **288.2** 不能与 scored 的 452.3 直接比较：旧值是在较低 GPU 时钟/其它负载下测得；scored 的 452.3 与本轮 A/B 的 422.2 也属 run-to-run 时钟差异。引用 BS16 时必须带运行条件，主排序仍以 BS1 P50 为主。
- 60.612G 仍是 v2 MAC 计数器对卷积/线性/矩阵乘的计数；`loss_map`、乘法和 reciprocal 是 elementwise，不计入该 FLOPs 数字。
- 文件：`benchmark_v2_scored.json`、日志 `benchmark_v2_scored.log`。
- 旧 `benchmark_v2.json` 保留，作为 forward-only 对照。
