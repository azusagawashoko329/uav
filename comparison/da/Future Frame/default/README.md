# Future Frame（TF1 部分测量）

- 论文：Liu et al., *Future Frame Prediction for Anomaly Detection — A New Baseline*, CVPR 2018。
- 代码：`/data1/data8tb/liangyuxuan/Comparison method/Future Frame/`（HEAD `9e0f161`）。
- 官方实现是 TensorFlow 1.4/CUDA 8；已在 `uav_tf1`（Python 3.7 + TF1.15 + CUDA10/cuDNN7）建图。

## 已得到

| 项 | 值 |
|---|---:|
| 推理图 | generator（U-Net），不含 FlowNet |
| 输入 | 4 帧 256×256 RGB |
| Inference / Trainable Params | 7.704M / 7.704M |
| FLOPs | 74.476G（TF1 profiler，CPU graph） |
| Latency/FPS | **未完成** |

## 为什么 latency 标 partial

- 官方 `pretrains.tar.gz` 的 OneDrive 链接现在返回 301/401，脚本里的 1drv 直链返回 404；AWS `flownetdata/weights.tar.gz` 也 404。无法拿到官方 generator checkpoint。
- TF1.15 二进制只带 CUDA 10/cuDNN 7；在 RTX 5880（sm_89）上 GPU session 的首次 kernel/PTX 编译长时间不返回，无法按 v2 协议完成 BS1/BS16 计时。CPU 计时不符合协议（不是 5880），因此不填。
- Inference 本身不需要 FlowNet；FLOPs/params 对权重不敏感，所以这两项可保留，但状态只能是 `partial/unverified`。

## 文件

- `comparison/da/Future Frame/default/benchmark_v2.json`
- 脚本：`comparison/benchmarks/bench_tf1_extra.py --method futureframe`
- AUC 仍引用用户表/论文；本目录不提供新的 AUC。
