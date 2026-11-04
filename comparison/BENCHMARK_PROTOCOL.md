# 轻量化对比协议（DA 主表）

本协议用于 Drone-Anomaly（DA）上多方法/多变体的参数量、FLOPs、时延与吞吐对比。**所有数字必须按同一口径产生**；不能把旧实验里“只算学生/不含教师”的数字混进主表。

## 1. 硬件与软件

- **GPU**：同一张 **RTX 5880 Ada**（本机 UUID `GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6`）；测量时该卡无其他任务。
- **CUDA / PyTorch**：尽量统一；每个方法记录 `torch.__version__`、CUDA 版本和 conda 环境。环境不同不能静默混表。
- **精度**：FP32；关闭 AMP、TF32：
  ```python
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  torch.set_float32_matmul_precision('highest')
  ```
- **模型模式**：`model.eval()`；计时用 `torch.inference_mode()`。

## 2. 参数量

- **Total Params (M)**：推理图从输入到“帧级异常分数”所经过的所有 `nn.Parameter`，**包含冻结的教师/主干/检测器/光流网络**；不含 buffer。
- **Trainable Params (M)**：one-class 训练时 `requires_grad=True` 的参数；冻结教师不计入。
- 训练专用模块（例如 GANomaly 的判别器 D）不计入推理 total，但必须在备注里单列；如果方法把 D 也计入训练参数量，另给 `train_params`。
- 单位 M = 10^6；保留至少 3 位小数。

## 3. FLOPs

- **口径**：一次 BS1 完整前向，从预处理后的输入张量到**一个输出帧的异常分数**；包含教师/时序编码器/预测器/检测器/光流网络/打分头，不包含 JPEG 解码、磁盘 I/O、CPU→GPU 拷贝。
- **工具**：PyTorch profiler `with_flops=True`（`sum(event.flops)`），统一口径；记录工具版本。该口径把乘加按 2 FLOPs 计。若论文给的是 MACs，不能直接与 GFLOPs 混用，需注明。
- **时序模型**：一个窗口产生 K 个有效输出帧时，`FLOPs/frame = 窗口 FLOPs / K`；不能把窗口长度当成单张图像。
- **光流**：在线光流网络计入主 FLOPs；若用预计算光流，主表只计模型，另报“光流预计算开销/帧”和可选端到端时延。

## 4. Latency

### 4.1 主指标：Latency@BS1（P50, ms/output frame）

- `batch_size=1`；每次调用完整推理链路，输出帧级异常分数。
- 预热 50 次；正式测量 500 次；重复 3 轮。
- 每次迭代前后 `torch.cuda.synchronize()`；记录单次耗时；报告中位数（P50）。可附均值/P95。
- 时序方法：一次调用处理一个窗口，若输出 K 个有效检测帧，则 `Latency@BS1 = 窗口时延 / K`；同时保留窗口时延，避免把窗口当单帧。
- 若方法需要光流：主时延包含光流网络；预计算光流时，另报在线端到端时延（模型+光流预计算/读取）。

### 4.2 吞吐：FPS@BS1 与 FPS@BS16

- **FPS@BS1**：由 BS1 的平均时延计算，`1000 / mean_latency_ms`（单位：output frames/s）。
- **FPS@BS16**：batch=16，连续执行多个 batch；**只在整轮结束时同步**，不要每个 batch 强制同步。预热 50 个 batch，正式 500 个 batch，重复 3 轮；`FPS@BS16 = (16 × 500 × K) / 整轮耗时`，其中 K 为每个样本的有效输出帧数。
- 时序模型：batch=16 指 16 个窗口；K 为每个窗口的输出帧数；不能把 16×窗口长度都算成输出帧。
- 报告单帧/单窗口的吞吐时，明确 K。

## 5. 输入规格

- **保持各方法原论文正式配置**，不强制 resize 到 256×256；表格必须有 `Input` 列：`分辨率 × 输入帧数 × 通道`。
- 单帧方法：`H×W×1`；时序方法：`H×W×T`（T = 窗口长度，注明输出 stride/输出帧数）。
- 如果为了跑通 DA 做了适配，`Status` 标 `adapted`，并在备注写清改动（resize、灰度、帧数、预处理）。
- 数据归一化、RGB/BGR、通道顺序按原方法；记录在 Input/Notes。

## 6. 公平比较与 AUC 来源

- 同一 GPU、同一精度、同一 BS、同一 warmup/正式次数、同一计时范围；不同方法的环境差异必须记录。
- **AUC 不能混源**：
  - 外部五种方法（adVAE/GANomaly/MemAE/MKD/ANDT）的 AUC 来自用户提供的对比表/ANDT 论文；ANDT 论文明确 baseline 是在其统一网络/评测协议下跑的，**不一定等于各方法原论文配置**。
  - Ours 的 AUC 来自 `experiments/da/appearance/20261103_variant_matrix_*`（seed42, 30 epoch）。
  - 其余方法（MADViT/LightViT/ASTT/DAD-FSM/HSTforU 等）的 AUC 以用户表格/原论文为准，必须标注来源。
- 轻量化指标（params/FLOPs/latency/FPS）与 AUC 是两条独立证据；不能因为 AUC 来源不同就混用轻量化口径，也不能把“原论文模型的轻量化”当成“ANDT 适配版的轻量化”。

## 7. 表格字段

```
Method | Input (H×W×T) | Total Params (M) | Trainable Params (M) |
FLOPs (G/output frame) | Latency@BS1 P50 (ms/output frame) |
FPS@BS1 | FPS@BS16 | AUC mean | Status | Source/Notes
```

`Status`：`measured` / `paper-only` / `blocked` / `pending` / `adapted`。

## 8. 已知阻塞项（写表时必须保留）

- **adVAE**：用户给的官方仓库 `WangXuhongCN/adVAE` 是**表格数据**版本（ODDS 数据集），不是 DA/ANDT 用的视频模型；不能用它的 FC 模型冒充视频模型。需要 ANDT 作者的实际 adVAE 实现或论文中的图像/特征配置，否则参数量标 `blocked`。
- **ANDT**：仓库 `ANDT/ANDT/model.py` 的 forward 含 `pdb.set_trace()`，且 decoder 对 `num_frames>1` 的 token 形状不匹配（`x.view(b,256,16,16)`），属于代码变体/未整理状态；参数量可按代码结构数，但 FLOPs/latency 需先确认/修补，不能静默当成论文 6 帧模型。
- **GANomaly**：参数量随 `isize` 变化（32→3.2M，128→38.4M，256→144.0M G）；必须锁定输入尺寸和是否含判别器 D。主表默认 `isize=256`，D 仅训练时使用，单列。
- **MemAE**：3D conv 需要窗口；默认 `T=16`、`mem_dim=2000`、彩色 `C=3`。参数量与 T 无关，但 FLOPs/latency 必须按窗口/输出帧折算。
- **MKD**：source VGG16（冻结）+ cloner；默认 `equal_network_size=False` → cloner B（0.344M trainable）。输入尺寸按原论文 MVTec 配置 128×128；若 DA 适配用 256，必须标 `adapted`，不能混。
- **ASTT/DAD-FSM**：时序窗口；必须包含时序编码器/预测器/打分，按输出帧计数；有光流则计入或单列预计算。

## 9. 复现命令（待脚本落地）

```bash
# 统一在 5880 Ada 上跑；脚本会关闭 TF32/AMP，使用 inference_mode
CUDA_VISIBLE_DEVICES=GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6 \
python comparison/benchmarks/bench_ours.py
```

结果写入 `comparison/da/benchmark_ours.json`，由 `comparison/da/comparison_summary.md` 引用。
