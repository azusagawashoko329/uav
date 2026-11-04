# MLEP（TF1 官方代码 + TF2 runtime-adapted 测量）

- 论文：Liu et al., *Margin Learning Embedded Prediction for Video Anomaly Detection with A Few Anomalies*, IJCAI 2019。
- 官方代码：`/data1/data8tb/liangyuxuan/Comparison method/MLEP/`（HEAD `22e96634fe43a5b8394413a1aae267b673b3c4ce`）。
- TF1 代码；先用 `uav_tf1`（Python 3.7 + TF1.15 + CUDA10/cuDNN7）建图，后改用 TF2.20 做 runtime-adapted 计时。

## 1. 推理图与口径

| 项 | 值 |
|---|---|
| 推理模块 | `cyclegan_convlstm` prediction net，`use_decoder=True` |
| 输入 | 4 帧 224×224 RGB，`[B, 4, 224, 224, 3]` |
| 输出 | 下一帧 `[B, 224, 224, 3]`；ConvLSTM hidden `[B, 56, 56, 256]` |
| K / stride / frame_coverage | 1 / 1 / 1.0 |
| score_density | 1.0 |
| Inference / Trainable Params | 12.561M / 12.561M |
| Training-only Params | 0 |
| FLOPs | 319.513G/output frame |
| Latency/FPS 口径 | prediction-only（旧 `benchmark_v2_tf2compat.json`）/ **score-inclusive PSNR BS1**（`benchmark_v2_scored.json`；BS16 未测） |

MLEP 的 margin learning 是训练目标；本表只测推理用的 prediction net（`use_decoder=True`），不把训练期的 margin/三元组计算塞进推理 FLOPs。

**评分口径**：官方 `inference.py` 在 prediction net 输出后还要对 ground-truth 下一帧计算 PSNR（`utils.util.psnr_error`）。旧 `benchmark_v2_tf2compat.json` 只到 prediction tensor；`benchmark_v2_scored.json` 已把官方 PSNR 放进计时图，但本轮只完成 BS1（BS16 长时 run 被停止），所以 BS16 留空。

## 2. 为什么官方 TF1.15 的 latency 拿不到

TF1.15 官方 wheel 绑定 **CUDA 10.0 + cuDNN 7.6.5**，RTX 5880 是 **Ada、sm_89**；这套旧 cuBLAS/cuDNN 没有可用的 sm_89 kernel。最小复现与 Future Frame 相同：

- Session 创建、变量初始化、`tf.add` 这类 elementwise GPU 运算可以完成；
- 64×64 `tf.matmul`（cuBLAS 10.0）40s 内不返回；
- 32×32 `tf.nn.conv2d`（cuDNN 7）100s 内不返回；
- 完整 MLEP 图 smoke 5–15 分钟不返回，手动终止。

MLEP 的 encoder/decoder 是卷积、recurrent 是 ConvLSTM（内部也是卷积），所以每次前向都会命中卡死的 cuBLAS/cuDNN 路径。官方 pretrains 的 OneDrive 链接也不可用；但这是独立问题，latency 与权重数值无关，random init 也能计时。官方 `benchmark_v2.json` 因此只保留 params/FLOPs，latency/FPS 为 `null`，状态 `partial/unverified`。

## 3. TF2 runtime-adapted 测量

用 **TF2.20.0（CUDA 12.5.1 build，cuDNN 9，Python 3.13.5）** 运行官方模型代码：

- `models/networks.py`、`models/pix2pix.py` 未改；由 `comparison/benchmarks/tf2_compat_shims.py` 提供 TF1 contrib API：
  - `tensorflow.contrib.layers` 的 `conv2d` / `conv3d` / `conv2d_transpose` / `max_pool2d` / `flatten` / `dropout` / `fully_connected` / `instance_norm` / `l2_regularizer`；
  - `tensorflow.contrib.framework.arg_scope`；
  - `tensorflow.contrib.rnn.ConvLSTMCell`。TF2 没有 contrib 版 ConvLSTMCell，这里按 TF1 contrib 的数学实现：`kernel` + `biases`，输入与 hidden 拼接后做 3×3 SAME 卷积，输出 4 组 gate，顺序 `input, new_input, forget, output`，`forget_bias=1.0`。
- **参数核对**：TF2 图 12,560,835，TF1 图 12,560,835，逐参数完全一致；输出 shape `[B,224,224,3]`、hidden `[B,56,56,256]` 也一致。这是 shim 没有改变官方结构的主要验收证据。
- `dynamic_rnn` 在 TF2 下会打印一条 `meta_optimizer ... layout failed ... cycle` 警告。这是 TF2 的图布局优化器跳过了 recurrent while-loop 的 layout pass，不影响执行；smoke 和完整 1500 次计时都正常完成。
- **FLOPs**：沿用 TF1.15 profiler 在同一图上的 319.513G；TF2 运行时没有另做 v2 手写计数器交叉验证，所以 `flops_verified=false`。
- **计时协议**：RTX 5880 Ada（UUID `GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6`），FP32，BS1 50 warmup + 3×500 次、BS16 50 warmup + 3×500 个 batch；每次 `Session.run` 同步计时。TF `Session.run` 本身是同步的，BS16 的排队方式与 PyTorch 的异步 launch 不完全相同。

| 口径 | Params | FLOPs | Latency@BS1 P50 | FPS@BS1 | FPS@BS16 | Status |
|---|---:|---:|---:|---:|---:|---|
| 官方 TF1.15 | 12.561M | 319.513G | —（CUDA10 在 sm_89 上不可执行） | — | — | `partial/unverified` |
| TF2.20 runtime-adapted，prediction-only | 12.561M | 319.513G | 11.538ms | 82.0 | 75.5 | `partial/unverified`（runtime-adapted） |
| TF2.20 runtime-adapted，**score-inclusive PSNR (BS1, per-sample [B])** | 12.561M | 319.513G | **12.525ms** | **79.6** | —（BS16 未测） | `partial/unverified`（runtime-adapted） |

补充：prediction-only mean 12.188ms、P95 13.822ms；score-inclusive PSNR mean 12.567ms、P95 14.099ms。BS16 的 prediction-only 三轮 500 batch 耗时 105.50 / 106.05 / 105.94s；score-inclusive BS16 在本轮超时未完成（已用 `--skip-bs16` 写 BS1 结果），所以 `fps_bs16=null`，不能拿 prediction-only 的 75.5 冒充 scored BS16。`fps_stream_estimated` 与 BS1 FPS 相同（K=stride=1），没有做连续视频流实测，所以 `fps_stream_measured=null`。

**score-inclusive 文件**：`benchmark_v2_scored.json` 使用官方 PSNR 图（预测帧 vs 与 GT 同形状的随机张量；不是真实标签，只测评分计算量）；旧的 `benchmark_v2_tf2compat.json` 保留为 prediction-only 对照。PSNR 的 elementwise 运算不计入 319.513G TF-profiler FLOPs。

**怎么读这个数**：它是“同一个 MLEP 图、同一张 5880、现代 CUDA 运行时”的 runtime-adapted 结果，不是官方 TF1.15/CUDA10 的 latency。主表引用时必须带 `runtime-adapted` 脚注；不能和论文里的 TF1.15 数字直接混比。

## 4. 复现

```bash
cd /data1/data8tb/liangyuxuan/code/uav
CUDA_VISIBLE_DEVICES=GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6 \
python comparison/benchmarks/bench_tf2_compat.py --method mlep
```

脚本会自动把 `libdevice.10.bc` 复制到临时工作目录（TF2 的 MLIR kernel generator 需要 cwd 里有这个文件），然后建 BS1/BS16 图并计时。

## 5. 文件与限制

- 官方 TF1.15 记录：`benchmark_v2.json`（params/FLOPs，latency null，`partial/unverified`）
- TF2 适配记录：`benchmark_v2_tf2compat.json`（prediction-only，BS1/BS16 原始耗时）
- score-inclusive 记录：`benchmark_v2_scored.json`（PSNR；BS1 完成、BS16 未测）
- 脚本：`comparison/benchmarks/bench_tf2_compat.py`、`comparison/benchmarks/tf2_compat_shims.py`
- 日志：`benchmark_v2_tf2compat.log`、`benchmark_v2_scored.log`（BS1 成功）、`benchmark_v2_scored_full.log`（BS16 超时尝试）
- AUC 仍引用用户表/论文；本目录不提供新的 AUC。
- 未拿到官方 checkpoint；random init 不影响 latency/FLOPs，但复现论文 AUC 需要权重。
- 若后面有 NGC TF1.15 + CUDA11 容器或 sm_75 老卡，可再做“官方 TF1.15 runtime”的交叉验证；当前数字不要冒充官方 TF1.15。
