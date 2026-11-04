# MLEP（TF1 部分测量）

- 论文：Liu et al., *Margin Learning Embedded Prediction for Video Anomaly Detection with A Few Anomalies*, IJCAI 2019。
- 代码：`/data1/data8tb/liangyuxuan/Comparison method/MLEP/`（HEAD `22e9663`）。
- TF1 代码；已在 `uav_tf1`（Python 3.7 + TF1.15 + CUDA10/cuDNN7）建图。

## 已得到

| 项 | 值 |
|---|---:|
| 推理图 | `cyclegan_convlstm` prediction net，`use_decoder=True` |
| 输入 | 4 帧 224×224 RGB |
| Inference / Trainable Params | 12.561M / 12.561M |
| FLOPs | 319.513G（TF1 profiler，CPU graph） |
| Latency/FPS | **未完成** |

## 为什么 latency 标 partial

- 官方 `data/pretrains` 的 OneDrive 链接同样不可用；没有 checkpoint。
- TF1.15 CUDA10/cuDNN7 在 RTX 5880（sm_89）上的 GPU session 长时间不返回，无法按 v2 协议完成 BS1/BS16 计时；CPU 计时不符合协议。
- params/FLOPs 对权重不敏感，保留为 `partial/unverified`。

## 文件

- `comparison/da/MLEP/default/benchmark_v2.json`
- 脚本：`comparison/benchmarks/bench_tf1_extra.py --method mlep`
- AUC 仍引用用户表/论文；本目录不提供新的 AUC。
