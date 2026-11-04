# Future Frame（TF1 官方代码 + TF2 runtime-adapted 测量）

- 论文：Liu et al., *Future Frame Prediction for Anomaly Detection — A New Baseline*, CVPR 2018。
- 官方代码：`/data1/data8tb/liangyuxuan/Comparison method/Future Frame/`（HEAD `9e0f161b8d5f1f46556e116577a5596405798343`）。
- 官方实现是 TensorFlow 1.4/CUDA 8；本地先用 `uav_tf1`（Python 3.7 + TF1.15 + CUDA10/cuDNN7）建图，后改用 TF2.20 做 runtime-adapted 计时。

## 1. 推理图与口径

| 项 | 值 |
|---|---|
| 推理模块 | generator（U-Net） |
| FlowNet | **不参与推理**；官方 FlowNetSD 只用于训练 |
| 输入 | 4 帧 256×256 RGB，按通道拼接为 `[B, 256, 256, 12]` |
| 输出 | 下一帧 `[B, 256, 256, 3]` |
| K / stride / frame_coverage | 1 / 1 / 1.0 |
| score_density | 1.0 |
| Inference / Training Trainable / Training-only Params | 7.704M / 14.536M / 6.831M |
| FLOPs | 74.476G/output frame |
| Latency/FPS 口径 | **Prediction-only**（只到 generator 输出；未含官方 PSNR 评分） |

注意：`input_hwt=256x256x4` 里的 `4` 是输入帧数 T；实现上 4 帧是沿 channel 维拼接的 12 通道张量，不是 `[B,4,3,256,256]`。K=1 表示一次前向只产生一个下一帧分数。

**训练/推理参数拆分**：`Codes/train.py` 在 `LAM_ADV != 0` 时训练 generator 和 discriminator（D 有 6,831,489 个可训练参数）；仓库自带的 ped1/ped2/avenue/shanghaitech 配置均为 `LAM_ADV=0.05`。因此官方默认训练参数量是 G+D = 14.536M，推理只用 G = 7.704M，Training-only = D = 6.831M。**DA 专用训练配置不在仓库里**，DA AUC 是否使用 adversarial loss 未确认；若 DA 训练设 `LAM_ADV=0`，则 D 不参与。

**评分口径**：官方 `Codes/inference.py` 在 generator 输出后还要对 ground-truth 下一帧计算 PSNR（`utils.psnr_error`），当前 benchmark 只计时到预测图像张量，没有包含 PSNR。因此主表/本 README 的 latency/FPS 应读作 **Prediction-only**；包含 PSNR 的 per-anomaly-score 计时仍需补测，不能直接称为完整评分耗时。

## 2. 为什么官方 TF1.15 的 latency 拿不到

TF1.15 官方 wheel 绑定 **CUDA 10.0 + cuDNN 7.6.5**，而 RTX 5880 是 **Ada、sm_89**。CUDA 10.0/cuDNN 7.6.5 发布时没有 sm_89，配套的 cuBLAS/cuDNN 没有可用的 Ada kernel。

本机最小复现（`uav_tf1` 环境，RTX 5880）：

| 测试 | 结果 |
|---|---|
| TF 识别 5880 | 成功，compute capability 8.9 |
| Session 创建 + 变量初始化 | 约 2.3s 完成 |
| `tf.add` elementwise GPU 运算 | 正常完成 |
| 64×64 `tf.matmul`（cuBLAS 10.0） | 40s 内不返回，超时终止 |
| 32×32 `tf.nn.conv2d`（cuDNN 7） | 100s 内不返回，超时终止 |
| Future Frame 完整图 smoke | 5–15 分钟不返回，手动终止 |

所以不是建图、Python 代码或权重的问题：**建图和简单 kernel 都能走，一进入 cuBLAS/cuDNN 就卡死**。Future Frame 是 U-Net，几乎每层都命中卷积，连一次前向都跑不完，更无法执行 50 次 warmup + 3×500 次 BS1/BS16 计时。

官方 checkpoint 也不可用（OneDrive 301/401、1drv/AWS 404），但这是独立问题：**latency 和权重数值无关，random init 也能计时**。因此官方 `benchmark_v2.json` 保留 params/FLOPs，latency/FPS 留 `null`，状态 `partial/unverified`。

## 3. TF2 runtime-adapted 测量

为在同一张 5880 上得到可比较的 latency，用 **TF2.20.0（CUDA 12.5.1 build，cuDNN 9，Python 3.13.5）** 运行官方模型代码：

- 模型文件 `unet.py` 未改；只由 `comparison/benchmarks/tf2_compat_shims.py` 提供 TF1 `tensorflow.contrib.layers` 的 `conv2d` / `max_pool2d` / `conv2d_transpose` 等价实现（底层仍是 TF2 的 `tf.nn.conv2d` / `tf.nn.max_pool2d` / `tf.nn.conv2d_transpose`）。
- **参数核对**：TF2 图 7,704,195，TF1 图 7,704,195，完全一致；输出 shape `[B,256,256,3]` 也一致。因此图结构和官方实现对齐。
- **FLOPs**：沿用 TF1.15 profiler 在同一图上的 74.476G；TF2 运行时没有另做 v2 手写计数器交叉验证，所以 `flops_verified=false`。
- **计时协议**：RTX 5880 Ada（UUID `GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6`），FP32，BS1 50 warmup + 3×500 次、BS16 50 warmup + 3×500 个 batch；每次 `Session.run` 同步计时。TF `Session.run` 本身是同步的，BS16 的排队方式与 PyTorch 的异步 launch 不完全相同，这一点只影响 BS16 吞吐的可比性，BS1 P50 是主比较口径。

| 口径 | Params | FLOPs | Latency@BS1 P50 | FPS@BS1 | FPS@BS16 | Status |
|---|---:|---:|---:|---:|---:|---|
| 官方 TF1.15 | 7.704M | 74.476G | —（CUDA10 在 sm_89 上不可执行） | — | — | `partial/unverified` |
| TF2.20 runtime-adapted | 7.704M | 74.476G | **5.467ms** | **180.8** | **185.1** | `partial/unverified`（runtime-adapted） |

补充：mean 5.531ms，P95 6.217ms；BS16 三轮 500 batch 耗时 43.09 / 43.22 / 43.37s。`fps_stream_estimated` 与 BS1 FPS 相同（K=stride=1），没有做连续视频流实测，所以 `fps_stream_measured=null`。

**怎么读这个数**：它回答的是“同一个 Future Frame 图、同一张 5880、现代 CUDA 运行时下有多快”。它不是官方 TF1.15/CUDA10 的 latency，不能和论文里 TF1.15 的数字直接混比；主表引用时必须带 `runtime-adapted` 脚注。它的价值在于：官方 TF1.15 在这张卡上根本无法执行，而这是目前唯一可复现的 5880 实测。

## 4. 复现

```bash
cd /data1/data8tb/liangyuxuan/code/uav
CUDA_VISIBLE_DEVICES=GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6 \
python comparison/benchmarks/bench_tf2_compat.py --method futureframe
```

脚本会自动把 `libdevice.10.bc` 复制到临时工作目录（TF2 的 MLIR kernel generator 需要 cwd 里有这个文件），然后建 BS1/BS16 图并计时。

## 5. 文件与限制

- 官方 TF1.15 记录：`benchmark_v2.json`（params/FLOPs，latency null，`partial/unverified`）
- TF2 适配记录：`benchmark_v2_tf2compat.json`（含 BS1/BS16 原始耗时数组）
- 脚本：`comparison/benchmarks/bench_tf2_compat.py`、`comparison/benchmarks/tf2_compat_shims.py`
- 日志：`benchmark_v2_tf2compat.log`
- AUC 仍引用用户表/论文；本目录不提供新的 AUC。
- 未拿到官方 checkpoint；random init 不影响 latency/FLOPs，但复现论文 AUC 需要权重。
- 若后面有 NGC TF1.15 + CUDA11 容器或 sm_75 老卡，可再做“官方 TF1.15 runtime”的交叉验证；当前数字不要冒充官方 TF1.15。
