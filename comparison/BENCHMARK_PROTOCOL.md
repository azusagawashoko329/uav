# 轻量化对比协议 v2（DA 主表）

- **版本**：v2，2026-11-04。
- **状态**：v2 已对 Ours V0–V5、GANomaly、MemAE、MKD、ANDT(adapted)、LightViT-AD、MADViT Full/Multi-layer/Last-layer、ST Dissoc.、ASTT、DAD-FSM、HSTforU 完成正式测量；Future Frame/MLEP 的官方 TF1.15 记录为 `partial/unverified`（params/FLOPs 有，GPU latency 未完成），另已增加 TF2.20 runtime-adapted 计时（Future Frame 5.467ms、MLEP 11.538ms，见 summary §3.3），但因 `flops_verified=false` 仍不能升 `measured`；adVAE 仍 `blocked`。本文件定义口径，结果见 [da/comparison_summary.md](da/comparison_summary.md) 与 [da/v2_validation.md](da/v2_validation.md)。
- **v1 数据**：已归到各方法目录的 `legacy_v1/`（Ours 在 `comparison/da/ours/legacy_v1/`，外部方法在 `<Method>/default/legacy_v1/`）与 `experiments/da/audit/20261026_lightweight_profile_fp32_s42/`，全部原样归档，只作历史记录；清单和哈希见 [da/LEGACY_v1.md](da/LEGACY_v1.md)，逐条改动见 [protocol_change_log.md](protocol_change_log.md)。
- **正式主表**：[da/comparison_summary.md](da/comparison_summary.md)。v2 重测完成前，数字留空或标 `partial/unverified` / `blocked` / `paper-only`，不得把 v1 数字当正式对比结果。

## 0. 一句话口径

一条正式记录必须同时回答六个问题：

1. **什么模型**：代码 commit、输入 `H×W×T`、权重状态、是否 DA 适配；
2. **多大**：Inference Params、Training Trainable Params、Training-only Params；
3. **多少计算**：可核验的 FLOPs，`1 MAC = 2 FLOPs`，覆盖/漏计算子要列清楚；
4. **多快**：同一计算图、同一精度、BS1 P50/Mean、BS16 连续吞吐；
5. **时序怎么摊**：窗口输入帧数 `T`、有效输出分数数 `K`、窗口步长 `stride`、窗口值/摊销值、独立窗口吞吐/真实视频流吞吐；
6. **AUC 从哪来**：AUC 来源与效率测量的模型配置是否同一版本。

缺任一项，`Status` 不能写 `measured`。

## 1. 硬件、软件、精度

- **GPU**：同一张 **RTX 5880 Ada**，UUID `GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6`；测量时该卡无其他任务。
- **软件**：每个方法记录 `torch.__version__`、CUDA、cuDNN、Python、conda 环境；环境不同不能静默混表。
- **精度**：FP32；关闭 AMP、TF32：
  ```python
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  torch.set_float32_matmul_precision('highest')
  ```
- **模式**：`model.eval()`；计时用 `torch.inference_mode()`。
- **FLOPs 计数模式**：见 §3.4；计数可以在 `torch.no_grad()` 下做，计时仍用 `inference_mode()`，两者必须在同一计算图/同一输入形状上。

## 2. 参数量：三列拆分，禁止再用含糊的 Total/Trainable

| 列 | 定义 | GANomaly 示例 |
|---|---|---:|
| `inference_params_m` | **Inference Params**：推理图实际加载并执行的 `nn.Parameter`，**包含冻结教师/主干/检测器**；不包含只在训练使用的模块（如 D），不包含 buffer | 143.950M（G） |
| `training_trainable_params_m` | **Training Trainable Params**：one-class 训练时 `requires_grad=True` 的全部参数，**包含训练专用模块 D**，不包含冻结教师 | 188.690M（G+D） |
| `training_only_params_m` | **Training-only Params**：只在训练使用、推理不执行的参数，是 Training Trainable 的子集（如 D） | 44.739M（D） |

规则：

- **不得把 Inference Params 叫 Total**，再与 Training Trainable 直接比较。GANomaly 的 `Inference < Training Trainable` 是正常的（推理只用 G，训练要更新 G+D）。
- 冻结教师计入 `inference_params_m`，不计入 `training_trainable_params_m`；如需透明，可另记 `frozen_inference_params_m`（只作拆分，不参与主排序）。
- `training_only_params_m <= training_trainable_params_m`；推理参数和训练可更新参数之间不要求大小关系。
- 单位 M = 10^6，至少 3 位小数；不含 buffer、优化器状态、梯度缓存。
- 论文只给一个 “total” 时，必须按上述定义映射到正确列，并在 Notes 写清论文原始口径；无法映射的标 `blocked`。

## 3. FLOPs：可核验口径，profiler 不能单独作数

### 3.1 计量范围

- **起点**：已经位于 GPU 上的预处理后输入张量（例如 `(B, C, H, W)` 或 `(B, C, T, H, W)`）。
- **终点**：最终异常分数张量（单分数或 `K` 个窗口内分数）。
- **必须包含**：推理必需的 Teacher/Backbone、时序编码器、Decoder、检测器、Scoring/打分头（含 top-k、距离、memory 等打分逻辑中产生乘加的部分）；在线光流网络如果参与推理，也计入。
- **不包含**：JPEG 解码、CPU→GPU 拷贝、磁盘 I/O、离线光流预计算、缓存构建、测试时预处理（resize/归一化/灰度等）。
  - 这些开销单独报告：`preprocess_ms_per_frame`、`offline_flow_ms_per_frame`、`cache_read_ms_per_frame`；需要端到端时另报 `online_end_to_end_ms_per_frame`。
- **时序模型**：先测一个窗口的 FLOPs，再按 §5 摊销到有效输出分数；不能把窗口长度当成单张图像。
- **MKD**：原方法是多层特征评分，不能只用最终层特征距离代替完整模型；重测时必须按原代码的 source/cloner 多层特征与评分实现。当前 v1 的“最终层距离”只能标 `partial/unverified`。
- **GANomaly**：推理图只走 G（`encoder1 → decoder → encoder2`）；D 是 training-only，不计入推理 FLOPs，但要在参数量列和 Notes 里单列。
- **Ours**：包含 DINOv3 教师 + OCBE + student + `anomaly_readouts` 打分；不能只算学生子网。
- **Teacher tap 不是计算裁剪**：tap 是“取哪几层特征”的输出选择，不是只运行这几层。V0 取 `[5,7,9,11]` 时，要得到第 11 层特征，Transformer 仍必须前向 `0..11` 的所有 block；FLOPs 按实际前向层数计，不能按“4 个 tap × 单层”算。能省的是未用 tap 的特征保存/后处理，不是得到第 11 层所必需的前向计算。

### 3.2 单位与基本约定

- **1 MAC = 2 FLOPs**；主表单位 `GFLOPs = 10^9 FLOPs`。
- 主表列名 `FLOPs (G/output score)`：一个有效输出异常分数对应的 FLOPs。
- 如果论文给的是 MACs，必须写明 `MACs`，不能直接和本表 GFLOPs 混用；换算时乘 2。
- FLOPs 是计算量，不是时延；elementwise/norm/pooling/softmax/top-k 等非 MAC 算子计入时延，但不计入 MAC-FLOPs（除非论文明确按“全部算子”统计，且需单独标注）。

### 3.3 必须覆盖的算子

| 算子 | MACs 公式（`MACs × 2 = FLOPs`） | 备注 |
|---|---|---|
| `Conv2d` | `N · Cout · Hout · Wout · (Cin/groups) · Kh · Kw` | 含 depthwise/grouped；`groups` 不能漏 |
| `ConvTranspose2d` | `N · Cin · Hin · Win · (Cout/groups) · Kh · Kw` 减去落到输出边界外的贡献 | stride/padding/output_padding/cropping 会影响精确值；以实际输入输出形状的 hook 计数为准，公式只作上界核对 |
| `Conv3d` | 同 Conv2d，多一个 `Dout` 和 `Kd` | 时序/3D 方法必须覆盖 |
| `ConvTranspose3d` | 同 ConvTranspose2d，多一个 `D` 维 | 3D 解码器必须覆盖 |
| `Linear` | `N · tokens · Cin · Cout` | 含 patch embedding 的线性投影、分类头、打分头 |
| `MatMul` / `BMM` | `prod(batch_dims) · M · N · K` | 包含 `@`、`torch.matmul`、`torch.bmm`、`einsum` 中可归约为矩阵乘的部分 |
| Attention QK | `B · H · Lq · Lk · D` | 多头注意力；`B` 为 batch，`H` 为 heads |
| Attention AV | `B · H · Lq · Lk · Dv` | 与 QK 分开计；不能只算一次 |
| `scaled_dot_product_attention` | QK + AV 两段 | 融合 kernel 也要拆成两段矩阵乘；softmax 本身不计 MAC |
| Patch embedding | 按实际实现计为 `Conv2d` 或 `Linear` | 不能因为“只是预处理”漏掉 |

**非 MAC 算子**（elementwise add/mul、LayerNorm/BatchNorm、GELU/ReLU、pooling、softmax、top-k、cosine distance、memory 读写）默认不计入 FLOPs，但必须：

- 在 Notes 列出；
- 在时延中实际包含；
- 如果某个实现把这些算子做成大矩阵乘（例如用 matmul 实现 distance），要按实际算子计入。

### 3.4 计数流程与验收

1. **固定计算图**：同一代码 commit、同一输入 `H×W×T`、同一权重状态、同一分支（teacher 的 tap 数、是否 scoring、是否 DA 适配都要一致）。
2. **算子计数器**：用模块 forward hook 覆盖 `Conv2d/ConvTranspose2d/Conv3d/ConvTranspose3d/Linear`，并包装 `torch.matmul/bmm/einsum/scaled_dot_product_attention`；逐模块记录 `op_type, module_name, input_shape, output_shape, MACs`。
3. **Profiler 只比共同支持的算子**：同一图跑 `torch.profiler(with_flops=True)`；**只对计数器与 profiler 都支持、且能一一对齐的算子**（通常是 `Conv2d`、`Linear`、未被融合的 `MatMul/BMM`）按算子类别/模块比较，共同子集差异 **≤1%**。不能拿“完整计数器总数”与“不完整 profiler 总数”相比，也不要求两者总 FLOPs 相等。
4. **独立公式核验 profiler 盲区**：对 `ConvTranspose2d/3d`、`Conv3d`、融合 `scaled_dot_product_attention`、自定义 kernel 等 profiler 漏计/无法拆分的算子，用计数器按实际输入输出形状计算，并用上表公式独立核对；每个盲区算子的 MACs 单独列出。MemAE 这类 3D 反卷积模型，完整计数器与 profiler 总数相差很大是正常的，不能因此判失败。
5. **完整 FLOPs 的定义**：`Full FLOPs = Σ(所有已验证 MAC 算子的 MACs) × 2`。它**不要求**等于 profiler 总数；profiler 只用于共同子集核验。某算子无法核验时，`flops_verified=false`、`flops_ops_missing` 列出该算子、`Status=partial/unverified`；**严禁把不完整值写成 Full FLOPs 或标 measured**。
6. **防重复计数**：同一次矩阵乘只能走一条计数路径。Module hook 已经计过的 `Linear/Conv`，Attention 包装不能再把同一 QK/AV 计一次；`scaled_dot_product_attention` 若内部由 matmul 组成，不能同时由 matmul 包装和 SDPA 包装累计。用 `op_id`（module + 调用序号 + 输入/输出 shape）或调用栈去重；共同子集与 profiler 的偏差若表现为整数倍，优先查重复计数。
7. **保存明细**：新结果 JSON 必须含 `flops_breakdown`（逐模块/逐算子 MACs）、`flops_ops_covered` / `flops_ops_missing`、`flops_common_subset_check`（共同子集差异）和 `flops_duplicate_check`（去重说明）；主表只引用通过检查的 `flops_g_per_output_frame`。

> 已知坑：`torch.profiler(with_flops=True)` 在 `torch.inference_mode()` 下可能拿不到 FLOPs（本仓库 v1 曾出现 0.0）；计数阶段改用 `torch.no_grad()`，计时阶段仍用 `inference_mode()`，并记录这一差异。Profiler 对 `ConvTranspose*`/`Conv3d`/融合 Attention 的覆盖不完整，不能把它的总数当作 Full FLOPs 的验收基准。

## 4. 时序方法：T、K、stride 必须分开

| 符号 | 定义 | 例子 |
|---|---|---|
| `T` | 一个窗口输入的帧数 | MemAE `T=16`；ANDT 论文 6 帧 `T=6` |
| `K` | 一个窗口产生的**有效输出异常分数**数，**不能默认等于 T** | ANDT 6 帧预测下一帧：`K=1`；6 帧重建并逐帧打分：可能 `K=6`；MemAE 原测试若只产 clip 级误差：`K=1` |
| `stride` | 窗口每次前进的原始视频帧数；`stride < T` 为重叠窗口 | 非重叠 `stride=T`；逐帧滑动 `stride=1` |
| `window_flops_g` | 一个窗口的 FLOPs | 由 §3 测得 |
| `window_latency_ms` | 一个窗口的 BS1 **Mean** 时延 | 与 `latency_bs1_mean_ms_per_output_frame × K` 对齐 |
| `flops_g_per_output_frame` | `window_flops_g / K` | 主表列 |
| `latency_bs1_*_ms_per_output_frame` | `window_latency_ms / K` | 主表列 |
| `fps_independent_window` | `K · 1000 / window_latency_mean_ms`（BS1） | 把每个窗口当独立样本、消费全部 K 个分数 |
| `score_density` | `K / stride` | 每个输入帧平均对应多少个分数；可 >1（重叠/多分数），**不是帧覆盖率** |
| `frame_coverage` | 实际收到至少一个分数的原始帧数 / 总帧数 | 0..1；必须按原方法的分数映射、窗口重叠和融合方式计算，不能由 `K/stride` 推出 |
| `fps_stream_estimated` | `stride · 1000 / window_latency_mean_ms`（BS1） | **按窗口步长推算**的流吞吐估计，不是实测真实流吞吐 |
| `fps_stream_measured` | 连续视频流测试的实际原始帧/秒 | 只有真正跑了连续流测试才填；否则 `null` |

规则：

- **不能默认 `K=T`**。必须从原方法测试/打分代码确认：窗口输出的是逐帧分数、clip 级分数，还是只预测下一帧。
- **`score_density` 与 `frame_coverage` 是两件事**：例如 `T=16, stride=1, K=16` 时 `score_density=16`（平均每输入帧 16 个分数），但 `frame_coverage` 最多是 `1.0`（100%），不可能是 1600%。`K<stride` 也不必然导致 `frame_coverage<1`，因为分数映射/融合可能把分数分配给多帧；必须按原方法计算 `frame_coverage`。
- 重叠窗口必须同时报告 **独立窗口吞吐** 和 **流吞吐估计**；两者不能混为一个 FPS。若没有实际连续流测试，只能写 `fps_stream_estimated`，**不能声称已实测 `fps_stream_measured`**。
- BS16 同理：`FPS@BS16 = 16 · K · 500 / median(round_time)`（输出分数/秒）；另报 `stream_FPS@BS16_estimated = 16 · stride · 500 / median(round_time)`（原始帧/秒）；只有实际连续流测试才另报 measured。
- 时序方法的主表 `Latency@BS1 P50` 按 output score 摊销；同时保留 `window_latency_ms`，避免把窗口时延伪装成单帧时延。
- **MemAE**：代码当前返回 `B×T` 的逐帧残差，但原论文测试可能生成 clip 级误差。现有 v1 的 `÷16` 结果**先标 `partial/unverified`**；重测时先核对原测试脚本的 `K`，再决定 `K=1` 还是 `K=T`，不得沿用旧除法。
- **ANDT**：论文 6 帧预测模型 `T=6, K=1, stride=1`；如果测的是 1 帧重建代码变体，必须标 `code-variant-1frame`，不能和论文 6 帧模型混表。
- **ASTT / DAD-FSM / Future Frame / ST Dissoc. / MLEP / HSTforU**：逐个确认 `T/K/stride`，有光流则按 §3.1 处理。

## 5. 完整推理范围与分模块记录

正式测量必须从 **GPU 预处理后输入张量** 到 **最终异常分数**，并记录分模块时延/FLOPs，便于复核：

| 模块 | 是否计入主表 | 说明 |
|---|---|---|
| 预处理（JPEG 解码/resize/归一化/灰度） | 否 | 单独报 `preprocess_ms_per_frame`；若做端到端部署，另报 |
| Teacher / Backbone | 是 | 冻结教师也计入推理参数和 FLOPs |
| Temporal encoder / predictor | 是 | 时序模型必含 |
| Decoder / reconstruction head | 是 | 含 2D/3D 反卷积 |
| Scoring head（top-k、距离、memory、检测后处理） | 是 | 产生乘加的部分计入；纯排序/阈值不计 MAC 但计入时延 |
| 在线光流 | 是 | 如果推理时实时计算；预计算则单独报离线成本 |
| 离线光流/缓存 | 否 | 主表不计；另报 `offline_*` 与端到端时延 |
| Training-only 模块（如 GANomaly D） | 否 | 不计入推理；在参数量/Notes 单列 |
| 测试时预处理/后处理 | 否 | 单独报；不能悄悄塞进模型时延 |

**MKD 特别要求**：按原方法的多层特征（source 与 cloner 的对应层）做评分；当前 v1 脚本只取最终层距离，不能称为“完整 MKD 推理”，必须重测后再标 `measured`。

## 6. Latency 与吞吐

### 6.1 Latency@BS1（P50 主指标）

- `batch_size=1`；每次调用完整推理链路，输出一个或多个分数。
- 预热 50 次；正式 500 次；重复 3 轮。
- 每次迭代前后 `torch.cuda.synchronize()`；记录每次耗时，报告 **P50（主指标）**、Mean、P95。
- 时序方法：同时记录 `window_latency_ms` 与 `latency_bs1_*_ms_per_output_frame = window_latency_ms / K`。
- 保存每轮原始耗时（至少 `bs1_round_times_s` 或逐次数组），不能只留一个 P50。

### 6.2 FPS@BS1

- **`FPS@BS1 = 1000 / mean_latency_ms`**（输出分数/秒）。
- 注意：**P50 和 Mean 是两个统计量**。主表 Latency 用 P50，FPS@BS1 用 Mean；不能拿 P50 反算 FPS 再声称一致。
- 时序方法：`FPS@BS1 = K · 1000 / mean_window_latency_ms`；另报 `fps_stream_estimated = stride · 1000 / mean_window_latency_ms`（未做连续流测试时只能叫 estimated）。

### 6.3 FPS@BS16

- `batch=16`，连续执行多个 batch；**只在整轮结束时同步**，不要每个 batch 强制同步。
- 预热 50 个 batch；正式 500 个 batch；重复 3 轮。
- `FPS@BS16 = 16 · K · 500 / median(round_time_s)`（输出分数/秒）。
- 时序方法另报 `stream_FPS@BS16_estimated = 16 · stride · 500 / median(round_time_s)`（原始帧/秒）；只有实际连续流测试才另报 measured。
- 保存 `bs16_round_times_s` 原始三轮时间。

### 6.4 环境记录

每个结果 JSON 记录：GPU 名称/UUID、驱动、torch/CUDA/cuDNN、Python、conda 环境、精度、是否 AMP/TF32、warmup/iterations/rounds、同卡是否有其他任务。不同环境的数字不能直接混排。

## 7. 输入规格与模型配置

- **保持各方法原论文正式配置**，不强制 resize；主表必须有 `Input (H×W×T)` 列（通道数写在 Notes，如 `C=3`）。
- 单帧方法：`H×W×1`；时序方法：`H×W×T`，并另表记录 `K/stride`。
- 如果为了跑通 DA 做了适配（resize、灰度、帧数、归一化、batch 组织），`da_adapted=true`，并在 Notes 写清改动；不能把适配版数字冒充原论文配置。
- 权重状态必须写：`scratch` / `pretrained` / `DA-trained` / `architecture-init`（仅测结构，权重随机）。效率数字对权重不敏感，但配置必须可追溯。
- `code_commit` 必须是实际运行的完整 commit SHA；本地 clone 路径写入 `source.local_path`。
- `profiling_config` 必须写成一行可复核的字符串，例如：
  `RTX 5880 Ada GPU-3ae8a423...; FP32; AMP off; TF32 off; eval; inference_mode; BS1 50/500/3; BS16 50/500/3`

## 8. AUC 来源与模型配置必须分开标注

AUC 和效率是两条独立证据，**不能暗示来自同一配置**：

- AUC 可能来自 ANDT 论文的 DA 适配版；效率可能来自原仓库默认模型（不同输入、不同层数、不同 head）。每行必须写 `auc_source` 和 `efficiency_config`。
- 外部五种方法（adVAE/GANomaly/MemAE/MKD/ANDT）的 AUC 来自用户提供的对比表 / ANDT 论文；ANDT 论文明确 baseline 是在其统一网络与评测协议下跑的，**不一定等于各方法原论文配置**。
- Ours 的 AUC 来自 `experiments/da/appearance/20261103_variant_matrix_*`（seed42，30 epoch）。
- 其余方法（MADViT/LightViT-AD/ASTT/DAD-FSM/HSTforU/CVAE 系列等）的 AUC 以用户表格/原论文为准，必须标注来源；`paper-only` 行不得与本机实测速度直接混排。
- **DwCVAE / CVAE / DCVAE / SCVAE / CLSTMVAE**：用户表里的速度来自不同论文/硬件/batch，标 `paper-only`；不能与 5880 Ada 本机实测放在同一排序里比较。

## 9. Status 定义

| Status | 含义 | 是否可进正式主表 |
|---|---|---|
| `measured` | v2 口径全部通过：同一图、参数量三列、FLOPs 计数覆盖完整且 `flops_verified=true`、时序 K/stride 已核对、BS1/BS16 原始耗时已保存 | 是 |
| `partial/unverified` | 有部分数字，但 FLOPs 覆盖不完整、K/stride 未核实、或配置/来源未对齐 | 否；只能作参考 |
| `paper-only` | 数字来自论文/用户表，未在本机同协议实测；硬件/batch 可能不同 | 否；单独表 |
| `blocked` | 代码/配置/decoder 缺失，无法测；不填猜测值 | 否 |

`measured` 的硬性条件：

- `flops_verified=true`、`flops_ops_missing` 为空，且 `flops_common_subset_check`/`flops_duplicate_check` 已记录；
- `code_commit` 为完整 SHA；
- `input_hwt`、`weights_state`、`da_adapted`、`auc_source`、`profiling_config` 均非空；
- `input_frames_t`、`effective_output_scores_k`、`window_stride` 均非空且已核实；
- `score_density`、`frame_coverage`、`fps_stream_estimated` 均非空且已核实（单帧方法为 1.0/1.0/同 FPS）；`fps_stream_measured` 只有实际连续流测试才填；
- `T>1` 时还必须有 `window_latency_ms`、`flops_window_g`、`fps_independent_window`；
- 原始 BS1/BS16 耗时已保存到结果 JSON；
- 参数量三列均为非空数值。

### 9.1 TF1 runtime-adapted 的特别口径

Future Frame / MLEP 的官方 TF1.15 运行时在 RTX 5880（sm_89）上无法执行 cuBLAS/cuDNN，不能按原 runtime 计时。允许用 TF2.20 + contrib shim 跑**同一官方模型图**，但必须：

- 写入单独的 `benchmark_v2_tf2compat.json`，`runtime_adapted=true`、`model_code_modified=false`、`adapted_reason` 非空；
- 给出 TF2/TF1 参数量逐参数核对（脚本内 `tf1_params_check`），不一致即失败；
- FLOPs 仍只能引用 TF1 profiler，`flops_verified=false`，因此 **Status 只能是 `partial/unverified`，不能升 `measured`**；
- 主表引用时用 `P50@BS1 + runtime-adapted` 脚注，不能覆盖官方 TF1.15 行，也不能把 TF2 数字写成论文 TF1 数字；
- 注明 TF `Session.run` 是同步的，BS16 排队方式与 PyTorch 异步 launch 不完全相同。

### 9.2 measured 与 paper-matched 分开

`measured` 只说明 v2 测速协议完成；**不等于**被测图/输入/评分与论文 AUC 所用配置一致。后者单独称 `paper-matched`，必须能从官方代码、配置或论文证明；无法证明时在 `notes`/论文表中标 `original-repo config`、`adapted`、`reconstructed` 或 `unconfirmed`。禁止仅因“都用了同一个网络名字”就把 `measured` 写成 `paper-matched`。逐方法判定见 [da/fairness_audit.md](da/fairness_audit.md)。

## 10. 表头与记录字段

### 10.1 主表表头

```
Method | Input (H×W×T) | Inference Params (M) ↓ | Training Trainable (M) ↓ |
Training-only (M) ↓ | FLOPs (G/output score) ↓ | FLOPs verified |
Latency@BS1 P50 (ms/output score) ↓ | FPS@BS1 ↑ | FPS@BS16 ↑ |
AUC mean ↑ | Status
```

箭头：**↓ 越低越好，↑ 越高越好**；FPS 越高越好。

### 10.2 时序附表

```
Method | T | K | stride | window FLOPs (G) | window Latency (ms) |
score density (K/stride) | frame coverage | FPS independent-window ↑ |
fps_stream_estimated (frames/s) ↑ | fps_stream_measured (frames/s) | Status
```

### 10.3 配置/来源附表

```
Method | code commit | Input (H×W×T) | weights state | DA-adapted |
AUC source | profiling config | benchmark source
```

### 10.4 机器字段

`record.json` 的可选 `efficiency` 对象字段定义见 [SCHEMA.md](SCHEMA.md)；原始 benchmark JSON 的字段命名必须与主表/附表一致，并含 `protocol_version: 2`、`measurement_status`、`flops_verified`、`flops_ops_covered/missing`、`flops_common_subset_check`、`flops_duplicate_check`、`flops_breakdown`。

## 11. 旧数据处置

- 现有 v1 文件已归到对应方法的 `legacy_v1/`（Ours 在 `comparison/da/ours/legacy_v1/`，外部方法在 `<Method>/default/legacy_v1/`）；**原样保留**，不覆盖、不静默改数。
- 旧审计 `experiments/da/audit/20261026_lightweight_profile_fp32_s42/` 原样保留。
- 新结果写在对应方法/变体目录：`comparison/<dataset>/<Method>/<variant>/benchmark_v2.json`（Ours 另在 `ours/` 下保留聚合文件）；JSON 内写 `protocol_version: 2`、`supersedes`、`measurement_status`、`code_commit`、`flops_counter` 和覆盖清单。
- v1 旧数字在 [da/comparison_summary.md](da/comparison_summary.md) 只放在“v1 归档”区，不得进入 v2 正式表。

### 11.1 GLoRD 新旧差异（v2 已定位）

| 项 | v1 profiler 结果 | 旧审计（2026-10-26） | v2 结论 |
|---|---:|---:|---:|
| V0 `rd_s12_t4_h512` FLOPs | 90.205G | 95.107G | **95.107G**（v1 漏 student native MHA） |
| V5 `rd_s4_t12_gl192_h128_qr2` FLOPs | 46.572G | 49.106G | **49.106G** |
| V0 BS1 latency | 17.800ms | 10.184ms | **14.521ms**（v2 重新计时；旧值不再用） |

v2 用同一计算图（teacher 12 taps、同一 scoring）、手写 92 个模块公式和主计数器逐模块对齐（max rel diff = 0）；profiler 总数 90.205G 只作参考，Full FLOPs 95.107G 与旧审计一致。**tap 是输出选择，不是计算裁剪**：V0 取 `[5,7,9,11]` 仍必须前向 `0..11`，FLOPs 按实际前向层数计，不能按 4 taps × 单层。旧 v1 latency（17.800ms/10.184ms）来自不同计时条件，不能再与 v2 混用。

## 12. 已知阻塞与逐方法重测清单

- **adVAE**：官方仓库 `WangXuhongCN/adVAE` 是表格/向量版（`(B,400)`，0.329M），不是 DA 视频模型；视频实现缺失，标 `blocked`。
- **ANDT**：`ANDT/ANDT/model.py` 有 `pdb.set_trace()`、硬编码 `.cuda()`，decoder `x.view(B,256,16,16)` 只支持 1 帧。v2 用最小 adapted patch `de_dense(feat[:,0])` 测 6 帧：`T=6, K=1, stride=1`，70.347M / 74.490G / 5.015ms / 199.4 FPS@BS1，结果标 **adapted**，不是官方 6 帧代码；官方 decoder 仍 `blocked`。若测 1 帧代码变体，另标 `code-variant-1frame`。
- **MemAE**：已核实原测试每个 16 帧窗口只输出一个 clip 级误差（对 T,H,W 取均值），`K=1, stride=1`；v2 全窗口 FLOPs=70.808G，旧 `÷16` 废弃。
- **MKD**：v2 已按原方法 taps 6/9/12 做多层特征评分；FLOPs=10.477G，P50=2.495ms。
- **GANomaly**：v2 已按 G 的完整推理图（含 ConvTranspose2d）重测；FLOPs=16.428G，D 仍为 training-only。
- **LightViT-AD / MADViT Full / ST Dissoc. / ASTT / DAD-FSM**：已按 v2 完成测量（结果在各自目录的 `benchmark_v2.json`）；MADViT Multi-layer/Last-layer 已在 `bench_v2_madvit_variants.py` 中按论文 Sec. 5.4 实现 adapted scoring 并完成测量。
- **HSTforU**：已用 mmcv-lite 2.x + 兼容 shim（runner/parrots/UPSAMPLE_LAYERS）完成测量：168.209G / 64.258ms / 15.5 FPS@BS1。
- **Future Frame / MLEP**：TF1 代码；已在 `uav_tf1` 环境构建推理图并得到 params/FLOPs（Future Frame 7.704M/74.476G；MLEP 12.561M/319.513G）。官方权重链接失效、官方 TF1.15 CUDA10 在 5880 的 GPU latency 无法按协议完成，官方记录标 `partial/unverified`。另用 TF2.20（CUDA12.5/cuDNN9）+ contrib shim 跑同一官方图，得到 runtime-adapted latency（Future Frame 5.467ms/180.8 FPS@BS1；MLEP 11.538ms/82.0 FPS@BS1）；TF2 图参数量与 TF1 逐参数一致，但 FLOPs 仍来自 TF1 profiler、`flops_verified=false`，所以状态仍 `partial/unverified`，不能冒充官方 TF1.15。
- **DwCVAE/CVAE/DCVAE/SCVAE/CLSTMVAE**：只有用户表数字，`paper-only`，不参与本机速度排序。

逐方法的“要改什么、要确认什么、当前状态”见 [RETEST_CHECKLIST.md](RETEST_CHECKLIST.md)。

## 13. 重测准入条件（DoD）

一个方法只有同时满足以下条件，才能从 `partial/unverified` / `blocked` 改为 `measured`：

- [ ] 固定完整 `code_commit`，记录本地路径和启动命令；
- [ ] 固定 `Input (H×W×T)`、通道顺序、归一化、权重状态、是否 DA 适配；
- [ ] 参数量三列（Inference / Training Trainable / Training-only）均为实测/结构审计值；
- [ ] FLOPs 用算子计数器核验所有 MAC 算子；profiler 只对共同支持子集比较（≤1%），盲区用独立公式；覆盖/漏计算子写入 JSON；完整 FLOPs 不要求等于 profiler 总数；防重复计数；
- [ ] 时序方法已核实 `T/K/stride`，并同时给出独立窗口与真实视频流吞吐；
- [ ] BS1 50/500/3、BS16 50/500/3，原始耗时保存，P50 与 Mean 分开；
- [ ] `flops_verified=true`、`flops_ops_missing=[]`；
- [ ] AUC 来源与效率配置分行注明，不暗示同一版本；
- [ ] 结果写入 `comparison/<dataset>/<Method>/<variant>/benchmark_v2.json`，未覆盖任何 v1 文件。

## 14. 复现入口（v2 改造后）

```bash
# 示例；v2 脚本改造完成前不要用 v1 脚本产出正式数字
CUDA_VISIBLE_DEVICES=GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6 \
python comparison/benchmarks/bench_ours.py
```

v2 脚本必须输出 `protocol_version: 2`、`measurement_status`、`flops_verified`、`flops_ops_covered`、`flops_ops_missing`、`flops_common_subset_check`、`flops_duplicate_check`、`flops_breakdown`、`input_hwt`、`T/K/stride`、`score_density`、`frame_coverage`、`fps_stream_estimated` 和原始耗时；否则只能算 v1 legacy。
