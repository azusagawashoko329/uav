# DA 方法对比总表

> **v2 协议（2026-11-04）已执行。** 轻量化协议见 [BENCHMARK_PROTOCOL.md](../BENCHMARK_PROTOCOL.md) v2：
> 参数量拆成 Inference / Training Trainable / Training-only；FLOPs = 所有已验证 MAC 算子之和，profiler 只比共同子集；
> 时序方法分开记录 `T/K/stride`；`Status` 只允许 `measured` / `partial/unverified` / `paper-only` / `blocked`。
> **§3 为 v2 实测结果；§5 的 v1 数字只作历史归档，不得作为正式对比结果。**
> 旧文件哈希与归档说明：[LEGACY_v1.md](LEGACY_v1.md)；验证记录：[v2_validation.md](v2_validation.md)；逐条改动：[protocol_change_log.md](../protocol_change_log.md)；逐方法重测：[RETEST_CHECKLIST.md](../RETEST_CHECKLIST.md)。

**AUC 来源必须分开读：**

- 外部方法（adVAE / GANomaly / MemAE / MKD / ANDT / Future Frame / ST Dissoc. / MLEP / ASTT / DAD-FSM / HSTforU / MADViT / LightViT-AD / CVAE 系列）的 AUC 取自用户提供的对比表 / ANDT 论文；ANDT 论文明确 baseline 是在其统一网络与评测协议下跑的，**不一定等于各方法原论文配置**。
- Ours 的 AUC 取自 `experiments/da/appearance/20261103_variant_matrix_*`（seed42，30 epoch，七场景 macro）。
- AUC 与轻量化是两条独立证据：AUC 可能来自 DA 适配版，效率可能来自原仓库默认模型；每行配置见 §3.2，不能暗示同一版本。

## 1. AUC（用户表格顺序，七场景）

| Method | highway | farmland | bike | solar | vehicle | crossroads | railway | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adVAE, 2020* | 61.1 | 73.8 | 72.8 | 66.1 | 55.1 | 56.1 | 62.1 | 63.87 |
| GANomaly, 2018* | 62.7 | 77.1 | 71.7 | 64.6 | 55.1 | 58.9 | 61.7 | 64.54 |
| MemAE, 2019* | 67.2 | 74.1 | 79.5 | 65.8 | 64.1 | 64.1 | 58.9 | 67.67 |
| MKD, 2021* | 64.3 | 75.2 | 74.8 | 63.5 | 62.7 | 63.5 | 62.4 | 66.63 |
| ANDT, 2022 | 68.7 | 79.5 | 82.2 | 64.2 | 61.3 | 65.2 | 59.4 | 68.64 |
| Future Frame, 2018 | 62.8 | — | 64 | — | 61.4 | 44.8 | 60.4 | — |
| ST Dissoc., 2022 | 76 | — | 79.5 | — | 38.6 | 38.1 | 33.5 | — |
| MLEP, 2019 | 55.6 | — | 45.1 | — | 56.2 | 49.8 | 68.4 | — |
| ASTT, 2024 | 83.5 | — | 78.4 | — | 74.1 | 38.6 | 64.5 | — |
| DAD-FSM, 2025 | 87.1 | — | 74 | — | 80.7 | 49.9 | 76 | — |
| HSTforU, 2025 | 70.6 | 60.8 | 82.7 | 67.7 | 65.6 | 53.5 | 67.8 | 66.96 |
| MADViT (Full) | 88.3 | 79.7 | 84.2 | 80.5 | 58.7 | 58.9 | 55.1 | 72.2 |
| MADViT (Multi-layer) | 74.2 | 77 | 82 | 32.8 | 59.9 | 56.9 | 57.5 | 62.9 |
| MADViT (Last-layer) | 93.3 | 89.9 | 80.5 | 84.7 | 59.7 | 60.3 | 50.9 | 74.19 |
| LightViT-AD, 2026 | 89 | 89.5 | 73.7 | 92.1 | — | — | 71.3 | *83.12 |
| DwCVAE, 2026 | 81.66 | — | 80.45 | — | 74.02 | 58.56 | 80.05 | *74.95 |
| CVAE, 2026 | 78.71 | — | 80.38 | — | 55.54 | 46.74 | 81.77 | *68.62 |
| DCVAE, 2026 | 76.56 | — | 79.9 | — | 58.92 | 44.88 | 60.31 | *64.11 |
| SCVAE, 2026 | 74.92 | — | 81.63 | — | 56.89 | 38.54 | 59.28 | *62.25 |
| CLSTMVAE, 2026 | 73.28 | — | 79.47 | — | 53.9 | 47.12 | 66.17 | *63.98 |
| Ours-外观(RD) (V0) | 92.31 | 99.4 | 98.08 | 87.29 | 60.54 | 59.7 | 94.93 | 84.6071 |
| 4block (V1) | 95.8 | 99.4 | 97.73 | 88.25 | 54.34 | 59.87 | 92.21 | 83.94 |
| Global-Local768 (V2) | 95.2336 | 99.2528 | 97.7067 | 89.5692 | 58.4427 | 61.195 | 92.0147 | 84.7735 |
| Global-Local192 (V3) | 95.02 | 99.32 | 97.6 | 82.05 | 57.48 | 61.25 | 85.01 | 82.53 |
| GL192-Dense128 (V4) | 94.5726 | 99.0567 | 97.5597 | 82.2232 | 58.2624 | 62.4147 | 84.2065 | 82.6137 |
| GL192-D128-Rank2 (V5) | 94.4461 | 99.2609 | 97.6317 | 83.1663 | 58.752 | 61.7818 | 85.6807 | 82.9599 |
| (30)GL192-D128-Rank2 (V5) | 94.8307 | 99.3254 | 97.6593 | 87.1792 | 56.2522 | 62.3047 | 90.5044 | 84.008 |

> `*` 表示只有部分场景或来自用户表，不能与七场景 macro 直接比较；空格 `—` 表示原表未给。

## 2. Ours 20261103 30ep 矩阵（新 AUC，与上表旧 Ours 行区分）

| ID | V | highway | farmland | bike | solar | vehicle | crossroads | railway | macro AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| rd_s12_t4_h512 | V0 | 93.16 | 99.42 | 98.05 | 87.14 | 60.25 | 58.81 | 96.71 | 84.7916 |
| rd_s4_t12_h512 | V1 | 95.72 | 99.25 | 97.68 | 89.42 | 57.24 | 60.85 | 94.44 | 84.9435 |
| rd_s4_t12_gl768_h512 | V2 | 95.86 | 99.22 | 97.65 | 90.09 | 57.82 | 63.05 | 93.91 | 85.3707 |
| rd_s4_t12_gl192_h512 | V3 | 94.86 | 99.31 | 97.66 | 86.42 | 58.59 | 60.88 | 90.89 | 84.0864 |
| rd_s4_t12_gl192_h128 | V4 | 94.48 | 99.13 | 97.61 | 86.18 | 58.30 | 62.20 | 89.53 | 83.9186 |
| rd_s4_t12_gl192_h128_qr2 | V5 | 94.89 | 99.32 | 97.68 | 87.18 | 56.25 | 62.25 | 90.53 | 84.0137 |

## 3. 轻量化主表 v2（2026-11-04 实测）

> 本表数字来自各方法/变体目录的 benchmark JSON：`FlopCounterMode + native-MHA 公式 + 手写模块公式 + profiler 共同子集`，50 warmup + 3×500 iterations。ST Dissoc.、DAD-FSM、Future Frame、MLEP 使用 `benchmark_v2_scored.json`（见 §3.4）；其余使用 `benchmark_v2.json`。旧 v1 数字在 §5 归档，不再引用。
> `FLOPs verified=yes` 表示所有 MAC 算子均已核验、`flops_ops_missing=[]`；profiler 只比共同支持算子，不用其总数做验收。
> **†** Future Frame / MLEP：latency/FPS 是 **TF2.20 runtime-adapted**（官方 TF1.15 CUDA10 在 5880 sm_89 上不可执行）；模型图未改、参数量与 TF1 逐参数一致，但 FLOPs 仍来自 TF1 profiler、`flops_verified=false`，所以状态仍是 `partial/unverified`，不是 `measured`。主表/§3.4 使用 score-inclusive PSNR 结果；§3.3 表保留 prediction-only 对照。MLEP 的 scored BS16 未测（见 §3.4），不能用 prediction-only 75.5 冒充。官方 TF1.15 的 latency 留空；详见 §3.3。
>
> **‡**（历史标记）只到 prediction / cluster-map，未含官方 PSNR、最终 loss_map 或异常分数；当前主表已无 ‡ 行；**§** 输入分割图是离线预计算，未含分割生成；**¶** 测的是原仓库配置（MKD MVTec-128、HSTforU Avenue/PVT2-b2），不是 DA AUC 配置；只作参考；**?** 模型变体未确认（ASTT 当前是 STE-TTE，DA `model.py` 缺失；仅结构参考，不进入主表排序）。`*` 表示 adapted/reconstructed（见下方 ANDT/MADViT 脚注）。输出为 batch 标量的方法（ANDT/ASTT/HSTforU/MADViT）其 BS16 FPS 不能按 16 个分数计，主表记 `—`，需逐样本重测；逐方法证据和“可用于主表/需脚注/需重测/暂不可用”判断见 [fairness_audit.md](fairness_audit.md)。

| Method | Input (H×W×T) | Inference Params (M) ↓ | Training Trainable (M) ↓ | Training-only (M) ↓ | FLOPs (G/output score) ↓ | FLOPs verified | Latency@BS1 P50 (ms/output score) ↓ | FPS@BS1 ↑ | FPS@BS16 ↑ | AUC mean ↑ | Status（measured / paper-matched） |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| adVAE | unknown | — | — | — | — | — | — | — | — | 63.87 | blocked |
| GANomaly | 256×256×1 | 143.950 | 188.690 | 44.739 | 16.428 | yes | 8.088 | 123.6 | 848.5 | 64.54 | measured / no（config 未证实） |
| MemAE | 256×256×16 | 6.503 | 6.503 | 0 | 70.808 | yes | 7.321 | 136.5 | 130.3 | 67.67 | measured / no（3 通道未证实） |
| MKD ¶ | 128×128×1 | 15.059 | 0.344 | 0 | 10.477 | yes | 2.495 | 394.5 | 2340.8 | 66.63 | measured / no（MVTec-128） |
| ANDT (adapted 6-frame)* | 256×256×6 | 70.347 | 70.347 | 0 | 74.490 | yes | 5.015 | 199.4 | — | 68.64 | partial/adapted / no（BS16 标量无效） |
| Future Frame † | 256×256×4 | 7.704 | 14.536 | 6.831 | 74.476 | no | 6.820 | 146.8 | 177.7 | — | partial / no（TF2 adapted） |
| ST Dissoc. | 256×256×5 | 6.120 | 6.120 | 0 | 60.612 | yes | 4.280 | 231.4 | 452.3 | — | measured / yes |
| MLEP † | 224×224×4 | 12.561 | 12.561 | 0 | 319.513 | no | 12.525 | 79.6 | — | — | partial / no（BS16 未测） |
| ASTT ? | 384×384×4 | 148.220 | 148.220 | 0 | 109.416 | yes | 19.048 | 52.2 | — | — | partial / no（variant 未确认） |
| DAD-FSM § | 224×224×4 | 123.979 | 123.979 | 0 | 71.194 | yes | 17.032 | 58.3 | 293.7 | — | measured / no（precomputed seg） |
| HSTforU ¶ | 256×256×4 | 148.970 | 148.970 | 0 | 168.209 | yes | 64.258 | 15.5 | — | 66.96 | partial / no（Avenue config） |
| MADViT Full * | 224×224×1 | 94.354 | 5.718 | 0 | 38.533 | yes | 11.525 | 83.9 | — | 72.2 | partial / no（token-only code） |
| MADViT Multi-layer * | 224×224×1 | 94.354 | 5.718 | 0 | 38.533 | yes | 11.130 | 88.9 | — | 62.9 | partial / no（reconstructed） |
| MADViT Last-layer * | 224×224×1 | 94.354 | 5.718 | 0 | 38.533 | yes | 9.467 | 103.5 | — | 74.19 | partial / no（reconstructed） |
| LightViT-AD | 224×224×1 | 8.840 | 3.049 | 0 | 3.753 | yes | 6.519 | 148.8 | 2438.6 | *83.12 | measured / yes（random weights） |
| DwCVAE | — | — | — | — | — | — | — | — | — | *74.95 | paper-only |
| CVAE | — | — | — | — | — | — | — | — | — | *68.62 | paper-only |
| DCVAE | — | — | — | — | — | — | — | — | — | *64.11 | paper-only |
| SCVAE | — | — | — | — | — | — | — | — | — | *62.25 | paper-only |
| CLSTMVAE | — | — | — | — | — | — | — | — | — | *63.98 | paper-only |
| Ours V0 `rd_s12_t4_h512` | 256×256×1 | 174.720 | 89.079 | 0 | 95.107 | yes | 14.521 | 68.4 | 187.4 | 84.7916 | measured / yes（architecture-init） |
| Ours V1 `rd_s4_t12_h512` | 256×256×1 | 125.900 | 40.259 | 0 | 68.532 | yes | 19.903 | 50.0 | 244.3 | 84.9435 | measured / yes（architecture-init） |
| Ours V2 `rd_s4_t12_gl768_h512` | 256×256×1 | 115.889 | 30.248 | 0 | 62.804 | yes | 14.787 | 67.3 | 263.3 | 85.3707 | measured / yes（architecture-init） |
| Ours V3 `rd_s4_t12_gl192_h512` | 256×256×1 | 93.360 | 7.718 | 0 | 51.132 | yes | 19.289 | 51.5 | 314.8 | 84.0864 | measured / yes（architecture-init） |
| Ours V4 `rd_s4_t12_gl192_h128` | 256×256×1 | 89.796 | 4.154 | 0 | 49.308 | yes | 19.417 | 51.3 | 324.7 | 83.9186 | measured / yes（architecture-init） |
| Ours V5 `rd_s4_t12_gl192_h128_qr2` | 256×256×1 | 89.402 | 3.761 | 0 | 49.106 | yes | 20.513 | 48.5 | 324.4 | 84.0137 | measured / yes（architecture-init） |

> **Status 列**：格式为 `measured / paper-matched`。`measured` 只表示 v2 测速协议完成；`paper-matched` 表示效率图/输入/评分与产生该 AUC 的模型配置一致。两者独立，`measured / no` 仍可展示但只能作参考，不能声称复现了论文模型。
>
> **\\*** ANDT 行为 **adapted 6-frame**：发布代码 decoder 只支持 1 帧，这里用 `de_dense(feat[:,0])` 预测下一帧；输入 `T=6, K=1, stride=1`。不是作者官方代码，AUC 仍引用论文/用户表的官方 6 帧模型，不能把这里的效率数字当成官方实现的效率；该行**不进入严格官方模型排名**，只作 adapted 参考。
>
> **MADViT 变体**：代码只有一条评分路径（全部 12 层 skip + token loss），没有 Full/Multi-layer/Last-layer 开关。按论文 Sec. 5.4 的定义，Full=全层 feature+token（代码默认）；Multi-layer=全层 feature only；Last-layer=final CLS/DIST token only。后两者用同一前向图、只替换 scoring，标 `reconstructed`；FLOPs/params 与 code default 相同，P50 因 scoring 不同而略有差异。**注意**：代码的 `multi_layer_distillation_loss` 实际 `return token_loss`，feature loss 计算后被丢弃，所以当前 “Full” 不是论文 Full；三变体都需重测 per-sample 评分，当前 BS16 FPS 因输出是 batch 标量而置 `—`。详见 [MADViT/README.md](MADViT/README.md)。
> **ST Dissoc.** 对 `padding=False` 做了现代 PyTorch 兼容修正（按 0 处理）；**DAD-FSM** 是 RGB+分割双流、ResNet50 随机初始化；**ASTT** 是 STE-TTE 变体；**LightViT-AD** 教师冻结、学生可训练。这些都在各自 record 的 `efficiency.notes` 里写明。
> **Future Frame 训练参数**：官方默认 `LAM_ADV=0.05`，训练同时更新 G+D；D 有 6.831M 可训练参数，因此 Training Trainable=14.536M、Training-only=6.831M，推理仍只走 G=7.704M。DA 专用 `LAM_ADV` 未在仓库中，是否用 D 训练未确认。
>
> **公平性复核**：`measured` 只表示 v2 测速完成，不等于 `paper-matched`（效率图与论文 AUC 所用配置一致）。逐方法的官方/适配/未确认判断、评分完整性和 `需脚注/需重测` 结论见 [fairness_audit.md](fairness_audit.md)。
>
> 复现与验证证据见 [v2_validation.md](v2_validation.md)。所有方法的结果都在各自目录的 `benchmark_v2.json`，未覆盖 v1 文件。

### 3.1 时序附表（T / K / stride）

| Method | T | K | stride | window FLOPs (G) | window Latency (ms) | score density (K/stride) | frame coverage | FPS independent-window ↑ | fps_stream_estimated (frames/s) ↑ | fps_stream_measured | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MemAE | 16 | 1 | 1 | 70.808 | 7.321 | 1.0 | 1.0（≈；边界帧除外） | 136.5 | 136.5 | null | measured |
| ANDT（adapted 6 帧） | 6 | 1 | 1 | 74.490 | 5.015 | 1.0 | ≈1.0（除开头 5 帧） | 199.4 | 199.4 | null | measured |
| Future Frame † | 4 | 1 | 1 | 74.476 | 6.820 | 1.0 | 1.0（≈） | 146.8 | 146.8 | null | partial/unverified † |
| ST Dissoc. | 5 | 1 | 1 | 60.612 | 4.280 | 1.0 | 1.0（≈） | 231.4 | 231.4 | null | measured |
| MLEP † | 4 | 1 | 1 | 319.513 | 12.525 | 1.0 | 1.0（≈） | 79.6 | 79.6 | null | partial/unverified † |
| ASTT ? | 4 | 1 | 1 | 109.416 | 19.048 | 1.0 | 1.0（≈） | 52.2 | 52.2 | null | partial/unverified ? |
| DAD-FSM § | 4 | 1 | 1 | 71.194 | 17.032 | 1.0 | 1.0（≈） | 58.3 | 58.3 | null | measured |
| HSTforU ¶ | 4 | 1 | 1 | 168.209 | 64.258 | 1.0 | 1.0（≈） | 15.5 | 15.5 | null | measured |

> `K` 不能默认等于 `T`；`score_density=K/stride` 可以 >1，**不是帧覆盖率**。`frame_coverage` 必须按原方法的分数映射、窗口重叠和融合方式单独计算（例如 T=16/stride=1/K=16 时 score_density=16，但 frame_coverage 最多 1.0）。`fps_stream_estimated` 只是按 stride 推算的吞吐估计；没有实际连续流测试时 `fps_stream_measured=null`，不能声称已实测。
> **MemAE 已核实**：`script_testing.py` 每个 16 帧窗口只输出一个 clip 级误差（对 T,H,W 取均值），并映射到中心帧；因此 `K=1, stride=1`，不是 `K=T=16`。旧 v1 的 `÷16` 把窗口时延/FLOPs 错误摊成 1/16，已废弃。
> **† Future Frame / MLEP**：§3.1 的 latency/FPS 来自 TF2.20 runtime-adapted；Future Frame 是 score-inclusive PSNR，MLEP 是 score-inclusive PSNR 的 BS1（BS16 未测，留 —）。`fps_stream_measured=null`，详见 §3.3/§3.4。官方 TF1.15 的 latency 仍未测，不能把这里的数字写成官方 TF1.15 数字。

### 3.2 每行配置与来源

| Method | code commit | Input (H×W×T) | weights state | DA-adapted | AUC source | profiling config |
|---|---|---|---|---|---|---|
| adVAE | `440d8af22b88b6b7b24e02f2f0599c9ef542b34c` | unknown（表格版 `(B,400)`） | — | no | ANDT Table VI / 用户表 | — |
| GANomaly | `78da4ea9a99f5b02ab60dd651a18def929176d77` | 256×256×1 | random init（未载入训练权重） | no | ANDT Table VI / 用户表 | v2: 5880 Ada, FP32, BS1 50/500/3, BS16 50/500/3 |
| MemAE | `ceece7714fb241e82ef3f3785d3d1ed86c28113e` | 256×256×16 | random init | no | ANDT Table VI / 用户表 | v2: 同上 |
| MKD | `0c574ae9ad12666982a1ccc2c7148acd6bf442fe` | 128×128×1（MVTec 配置） | random init（VGG16 source 未载入预训练） | no | ANDT Table VI / 用户表 | v2: 同上 |
| ANDT（adapted 6-frame） | `f2e8c4f701238a99faa5841b3e2884cb1c507d1e` | 256×256×6 | random init（无官方 checkpoint） | no（code adapted, not DA） | ANDT 论文 / 用户表（官方 6 帧模型；效率为适配版） | v2: 5880 Ada, FP32, BS1 50/500/3, BS16 50/500/3；decoder 为 `de_dense` 适配，论文 2FC→8×8×512→5conv 未复现，`paper_matched=false` |
| Future Frame | `9e0f161b8d5f1f46556e116577a5596405798343` | 256×256×4 | random init（无官方 checkpoint） | no | 用户表 | 官方 TF1.15：CPU profiler for FLOPs，GPU latency blocked；TF2.20 runtime-adapted 见 §3.3 |
| ST Dissoc. | `1206c2a7c4c4e0b47c21c498fadb8e1e3973dd56` | 256×256×5（15 通道） | random init | no | 用户表 | v2: 5880 Ada, FP32, BS1 50/500/3, BS16 50/500/3；score-inclusive 见 §3.4 |
| MLEP | `22e96634fe43a5b8394413a1aae267b673b3c4ce` | 224×224×4 | random init（无官方 checkpoint） | no | 用户表 | 官方 TF1.15：CPU profiler for FLOPs，GPU latency blocked；TF2.20 runtime-adapted 见 §3.3 |
| ASTT ? | `1d05305a94fb47b33897c7aa085c5351bcbbc8f9` | 384×384×4（STE-TTE） | random init | no | 用户表 | v2: 同上；**当前是 `model_ste_tte` 148.22M，DA `model.py` 缺失，变体未确认** |
| DAD-FSM | `602fed9ac93e6a3f2a291e9b1f23bcca775df961` | 224×224×4（RGB+分割双流） | random init（ResNet50 预训练权重未带） | no | 用户表 | v2: 同上；分割图离线预计算，score-inclusive 见 §3.4 |
| HSTforU ¶ | `9548473306bbb89895b077275e8503de79653029` | 256×256×4 | random init | no | 用户表 | v2: 5880 Ada, FP32, BS1 50/500/3, BS16 50/500/3；mmcv-lite 2.x + 兼容 shim；**Avenue/PVT2-b2 是假设，repo 无 DA config，paper_matched=false** |
| MADViT Full（code default） | `a1d1732c78f9f0a21a1a162b87b158436dad79f3` | 224×224×1 | random init | no | 用户表 | v2: 同上 |
| MADViT Multi-layer / Last-layer | 同上 | 224×224×1 | random init | no | 用户表 | v2: 论文 Sec. 5.4 的 adapted scoring；同一前向图 |
| LightViT-AD | `c0f6ca9a6817d126e15e09f83612710ee6f954d1` | 224×224×1 | random init（teacher 冻结） | no | 用户表（缺 vehicle/crossroads） | v2: 同上 |
| DwCVAE / CVAE / DCVAE / SCVAE / CLSTMVAE | — | — | — | — | 用户表（不同论文/硬件/batch） | paper-only，不可与本机混排 |
| Ours V0–V5 | `cef6f9690c0bcd3b5f05d4ffab6ff93640a134b7`（模型代码基线；v2 benchmark 脚本基于该提交运行，脚本见 `comparison/benchmarks/`） | 256×256×1 | architecture-init（效率测结构；AUC 用 DA-trained 权重） | yes（DA 输入/配置） | `experiments/da/appearance/20261103_variant_matrix_*`（seed42, 30ep） | v2: 5880 Ada, FP32, BS1 50/500/3, BS16 50/500/3 |

### 3.3 TF1 方法的 runtime-adapted 计时（TF2.20）

> 官方 TF1.15 wheel 绑定 CUDA 10.0/cuDNN 7.6.5，RTX 5880（sm_89）上 cuBLAS/cuDNN 会卡死，无法按 v2 协议计时；主表用 `†` 标出同图 TF2.20 runtime-adapted 数字，官方 TF1.15 的 latency 仍留空。下面是把**同一官方模型图**放到 TF2.20（CUDA 12.5.1/cuDNN 9）上、同一张 5880、同一 v2 计时协议得到的结果。模型文件未改，只补 TF1 contrib API；两方法的 TF2 图参数量与 TF1 图逐参数完全一致。运行时不同，不能冒充官方 TF1.15，也不能和论文里的 TF1 数字直接混比。

| Method | Runtime | Input (H×W×T) | Params (M) | FLOPs (G) | P50 (ms) | Mean (ms) | P95 (ms) | FPS@BS1 | FPS@BS16 | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Future Frame | TF2.20 / CUDA 12.5.1 / cuDNN 9 | 256×256×4 | 7.704 | 74.476 | 5.467 | 5.531 | 6.217 | 180.8 | 185.1 | `partial/unverified`（runtime-adapted） |
| MLEP | TF2.20 / CUDA 12.5.1 / cuDNN 9 | 224×224×4 | 12.561 | 319.513 | 11.538 | 12.188 | 13.822 | 82.0 | 75.5 | `partial/unverified`（runtime-adapted） |

说明：

- 两方法都是 `K=1, stride=1, frame_coverage=1.0`；FPS@BS16 按 16×500/median(round_time) 计算，未做连续流实测，`fps_stream_measured=null`。
- TF `Session.run` 是同步的；BS16 的排队方式与 PyTorch 的异步 launch 不完全相同，BS1 P50 是更干净的跨方法口径。
- MLEP 的 BS16 FPS 低于 BS1，来自 ConvLSTM `dynamic_rnn` 的 while-loop 与 batch 16 调度开销；这是实测值，不做外推修正。
- FLOPs 仍来自 TF1.15 profiler（同一图），没有做 v2 手写计数器交叉验证，所以 `flops_verified=false`；状态只能是 `partial/unverified`，不能升为 `measured`。
- 无官方 checkpoint，random init；latency/FLOPs 与权重数值无关，复现 AUC 才需要权重。
- 详细说明：[Future Frame/default/README.md](Future%20Frame/default/README.md)、[MLEP/default/README.md](MLEP/default/README.md)；原始数据：`Future Frame/default/benchmark_v2_tf2compat.json`、`MLEP/default/benchmark_v2_tf2compat.json`。
- 主表已用 **P50@BS1 + `runtime-adapted †`** 放入这两个方法；官方 TF1.15 的 latency 仍留空，不要用 TF2 数字覆盖官方 `benchmark_v2.json`，也不要把 `Status` 写成 `measured`。

### 3.4 score-inclusive 补测（2026-11-04）

旧 `benchmark_v2.json` 里 ST Dissoc. 是 stage-E forward-only、DAD-FSM 是 MSE-to-zero proxy、Future Frame/MLEP 是 prediction-only。已用 `bench_v2_scored.py` / `bench_tf2_scored.py` 补上官方评分路径，写独立 `benchmark_v2_scored.json`，不覆盖旧 JSON。计时用的 target 是**同形状随机张量，不是真实数据标签**，只衡量相同形状评分图的计算量：

| Method | 旧口径 | 旧 P50 / FPS@BS1 / FPS@BS16 | **score-inclusive** | 新 P50 / FPS@BS1 / FPS@BS16 |
|---|---|---|---:|---:|
| Future Frame † | prediction-only | 5.467ms / 180.8 / 185.1 | 官方 PSNR 图 vs 同形状随机 target，per-sample [B] | **6.820ms / 146.8 / 177.7** |
| ST Dissoc. | stage-E forward-only | 3.668ms / 269.7 / 288.2 | 官方 `loss_map` + reciprocal | **4.280ms / 231.4 / 452.3** |
| MLEP † | prediction-only | 11.538ms / 82.0 / 75.5 | 官方 PSNR 图 vs 同形状随机 target，per-sample [B] | **12.525ms / 79.6 / —（BS16 未测）** |
| DAD-FSM § | MSE-to-zero proxy | 12.684ms / 78.5 / 195.6 | 未来分割 target MSE（per-sample [B]） | **17.032ms / 58.3 / 293.7** |

- ST/DAD 的评分 FLOPs（elementwise）不计入各自的 60.612G / 71.194G MAC 计数；FF/MLEP 的 PSNR 也不计入 TF-profiler FLOPs。
- DAD-FSM 的分割图仍是离线预计算（`§`），不是完整在线成本。
- 旧 JSON 保留为对照；主表 ST/DAD/FF 已切到 score-inclusive 数字；MLEP 已切到 scored BS1，scored BS16 未测（留 —）。
- **BS16 A/B 核验**（`fairness_bs16_ab.json`，同卡 old/scored 交替 3 轮，50 warmup + 3×500）：ST 旧版 426.6 vs scored 422.2 FPS（时间比 1.010）；DAD 280.8 vs 280.7 FPS（时间比 1.000）。结论：旧 JSON 的 288.2/195.6 是低 GPU 时钟/其它负载下的 run，不能与 scored 直接比较，也不能据此说“评分更快”；scored 的 452.3/293.7 与本轮 A/B 的差为 run-to-run 时钟差异。

### 3.5 Measured vs Paper-matched

| Method | Measured | Paper-matched | 说明 |
|---|---|---|---|
| Ours V0–V5 | 是 | 是（结构） | 效率 architecture-init；AUC 用 DA-trained 权重 |
| GANomaly | 是 | 未确认 | 输入 256 未在 repo 证实；评分完整 |
| MemAE | 是 | 未确认 | 3 通道未在 repo 证实；T/K/评分正确 |
| MKD | 是 | **否** | 当前是 MVTec-128，不是 DA 配置 |
| ANDT | partial（adapted） | **否** | decoder 是 de_dense 适配；评分是 batch 标量代理；不进入严格官方排名 |
| Future Frame | partial | 推理 G 匹配；D/评分见注 | TF2 runtime-adapted + PSNR；DA LAM_ADV 未确认 |
| ST Dissoc. | 是（scored） | 是 | loss_map+reciprocal 已补；padding 兼容修正 |
| MLEP | partial | 推理结构匹配；BS16 未测 | TF2 runtime-adapted + PSNR BS1 |
| ASTT | partial | **否/未确认** | DA `model.py` 缺失；当前 STE-TTE 148.22M |
| DAD-FSM | 是（scored） | 部分 | 评分已补；分割图是离线预计算 |
| HSTforU | partial | **未确认** | repo 无 DA config；当前 Avenue/PVT2-b2；评分是 batch 标量代理 |
| MADViT Full | partial | **否** | code default 实际 token-only，不是 paper Full |
| MADViT Multi/Last | partial | reconstructed | 只替换 scoring；batch 标量需重测 |
| LightViT-AD | 是 | 是 | drone_anomaly.yaml 224 |
| adVAE | blocked | — | 无 DA 视频实现 |
| CVAE 系列 | paper-only | — | 无本机实现 |

> 该表与 [fairness_audit.md](fairness_audit.md) 同步；`Measured` 只表示 v2 测速完成，`Paper-matched` 表示图/输入/评分与论文 AUC 配置一致。

## 4. 参数审计细节（v2 三列口径）

> 以下是 v2 实测/结构审计值，三列定义与 §3 一致；Ours 的 Inference 含冻结教师 85.641M，Training Trainable 不含教师。

| Method/Variant | Inference Params (M) | Training Trainable (M) | Training-only (M) | Frozen Inference (M) | 备注 |
|---|---:|---:|---:|---:|---|
| GANomaly | 143.950（G） | 188.690（G+D） | 44.739（D） | 0 | G 推理，D 只训练；`Inference < Training Trainable` 正常 |
| MemAE | 6.503 | 6.503 | 0 | 0 | `mem_dim=2000`、`T=16`；参数与 T 无关 |
| MKD | 15.059 | 0.344 | 0 | 14.715 | source VGG16 冻结 + cloner B 可训练 |
| ANDT（论文 6 帧, mlp4096） | 70.347 | 70.347 | 0 | 0 | 代码结构审计；decoder 未修，FLOPs blocked |
| ANDT（论文 6 帧, mlp1536） | 62.478 | 62.478 | 0 | 0 | config 默认 |
| ANDT（1 帧代码变体, mlp4096） | 69.364 | 69.364 | 0 | 0 | 不能当论文 6 帧模型 |
| adVAE（官方表格版） | 0.329 | 0.329 | 0 | 0 | 输入 `(B,400)`，不是 DA 视频模型；主表 blocked |
| Ours V0 | 174.720 | 89.079 | 0 | 85.641 | teacher + OCBE + student |
| Ours V1 | 125.900 | 40.259 | 0 | 85.641 | 同上 |
| Ours V2 | 115.889 | 30.248 | 0 | 85.641 | 同上 |
| Ours V3 | 93.360 | 7.718 | 0 | 85.641 | 同上 |
| Ours V4 | 89.796 | 4.154 | 0 | 85.641 | 同上 |
| Ours V5 | 89.402 | 3.761 | 0 | 85.641 | 同上 |

- **Inference Params** 含冻结教师；**Training Trainable** 不含教师，含训练专用模块；**Training-only** 是训练专用子集。
- 旧字段 `total_params_m` / `trainable_params_m` 不再作为主表列名；它们只出现在 v1 归档 JSON 里。
- GANomaly 的 D 不能悄悄丢掉：推理表用 G，训练列写 G+D，并单列 D。

## 5. v1 轻量化归档（只读，不得作为正式对比）

> 以下数字来自 v1 口径（profiler-only FLOPs、未拆参数列、MemAE 未核实 K、MKD 最终层距离）。保留原值只为追溯；v2 重测前不得引用为正式结果。原始文件与哈希见 [LEGACY_v1.md](LEGACY_v1.md)。

| Method | Input (H×W×T) | Total Params (M) | Trainable Params (M) | FLOPs (G/output frame) | Latency@BS1 P50 (ms/output frame) | FPS@BS1 | FPS@BS16 | v1 Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| adVAE | unknown | blocked | blocked | blocked | blocked | — | — | blocked |
| GANomaly | 256×256×1 | 143.950 (G, inference) | 188.690 (G+D, train) | 10.952 | 13.357 | 74.9 | 622.7 | measured |
| MemAE | 256×256×16 | 6.503 | 6.503 | 0.066 | 0.437 | 2287.0 | 2193.0 | measured |
| MKD ¶ | 128×128×1 | 15.059 | 0.344 | 10.477 | 2.273 | 442.9 | 3554.4 | measured |
| ANDT | 256×256×6 | 70.347 (mlp4096) | 70.347 | blocked | blocked | — | — | blocked |
| Future Frame | — | pending | pending | pending | pending | — | — | pending/no code run |
| ST Dissoc. | — | pending | pending | pending | pending | — | — | pending/no code run |
| MLEP | — | pending | pending | pending | pending | — | — | pending/no code run |
| ASTT | — | pending | pending | pending | pending | — | — | pending/no code run |
| DAD-FSM | — | pending | pending | pending | pending | — | — | pending/no code run |
| HSTforU | — | pending | pending | pending | pending | — | — | pending/no code run |
| MADViT Full | — | pending | pending | pending | pending | — | — | pending/no code run |
| MADViT Multi-layer | — | pending | pending | pending | pending | — | — | pending/no code run |
| MADViT Last-layer | — | pending | pending | pending | pending | — | — | pending/no code run |
| LightViT-AD | — | pending | pending | pending | pending | — | — | pending/no code run |
| DwCVAE | — | user table 21.29 | user table 21.29 | user table 0.79 | user table 0.65 | user table 1532.9 | — | user-table, not re-measured |
| CVAE | — | user table 8.78 | user table 8.78 | user table 1.388 | user table 0.7 | user table 1426.15 | — | user-table, not re-measured |
| DCVAE | — | user table 8.78 | user table 8.78 | user table 1.388 | user table 0.96 | user table 1039.76 | — | user-table, not re-measured |
| SCVAE | — | user table 8.71 | user table 8.71 | user table 0.827 | user table 0.87 | user table 1149.87 | — | user-table, not re-measured |
| CLSTMVAE | — | user table 38.32 | user table 38.32 | user table 3.295 | user table 2.04 | user table 490.87 | — | user-table, not re-measured |
| Ours V0 | 256×256×1 | 174.720 | 89.079 | 90.205 | 17.800 | 58.7 | 193.9 | measured |
| Ours V1 | 256×256×1 | 125.900 | 40.259 | 65.240 | 18.672 | 53.3 | 249.2 | measured |
| Ours V2 | 256×256×1 | 115.889 | 30.248 | 60.119 | 18.648 | 53.4 | 264.1 | measured |
| Ours V3 | 256×256×1 | 93.360 | 7.718 | 48.595 | 18.620 | 53.4 | 318.2 | measured |
| Ours V4 | 256×256×1 | 89.796 | 4.154 | 46.771 | 18.656 | 53.3 | 325.0 | measured |
| Ours V5 | 256×256×1 | 89.402 | 3.761 | 46.572 | 19.638 | 50.7 | 323.7 | measured |

> v1 的 `Status=measured` 只是旧标签，按 v2 口径应视为 `partial/unverified`（FLOPs 未做算子级覆盖核验）或 `blocked`（adVAE/ANDT）。

## 6. 复核结论

1. **GLoRD 新旧 FLOPs 差异已定位（v2 解决）**：
   - v2 计数器：V0 = **95.107G**，V5 = **49.106G**，与 2026-10-26 审计一致；v1 的 90.205G/46.572G 是 profiler 漏计了 student 的 native MHA。
   - v2 重新计时：V0 BS1 P50 = **14.521ms**，V5 = **20.513ms**；旧 17.800/10.184ms 不再作为正式值。
   - **tap 不是计算裁剪**：V0 取 `[5,7,9,11]` 仍必须前向 `0..11`，FLOPs 按实际前向层数计，不能按 4 taps × 单层。
2. **MemAE 的 K 已核实**：原 `script_testing.py` 每个 16 帧窗口只输出一个 clip 级误差（对 T,H,W 取均值），并映射到中心帧；`K=1, stride=1`。v2 全窗口 FLOPs = **70.808G/output score**，旧 `÷16` 的 0.066G/0.437ms/2287FPS 错误，已废弃。
3. **MKD 多层特征评分已实现**：按原 `test_functions.py` 的 taps 6/9/12 做 cosine + 0.01·MSE；FLOPs = **10.477G**，P50 = **2.495ms**。v1 只用最终层距离，已废弃。
4. **ANDT 已测 adapted 6-frame**：官方代码 decoder 只支持 1 帧，仍不能冒充 official；v2 用 `de_dense(feat[:,0])` 适配为 `T=6,K=1,stride=1`，得到 70.347M / 74.490G / 5.015ms / 199.4 FPS@BS1；BS16 输出 batch 标量，per-sample FPS 无效。该行标 adapted，AUC 仍引用官方 6 帧论文值，不进入严格官方模型排名。
5. **本轮新增测量**：
   - LightViT-AD：3.753G / 6.519ms / 148.8 FPS@BS1；
   - MADViT：Full 38.533G / 11.525ms / 83.9 FPS@BS1；Multi-layer 11.130ms / 88.9 FPS；Last-layer 9.467ms / 103.5 FPS；后两者是按论文 Sec. 5.4 重建的 adapted scoring；三者输出 batch 标量，BS16 per-sample FPS 无效；
   - ST Dissoc.：60.612G / 4.280ms / 231.4 FPS@BS1 / 452.3 FPS@BS16（score-inclusive loss_map+reciprocal；`padding=False→0` 兼容修正）；旧 forward-only 3.668ms 保留为对照；
   - ASTT（STE-TTE）：109.416G / 19.048ms / 52.2 FPS@BS1；
   - DAD-FSM：71.194G / 17.032ms / 58.3 FPS@BS1 / 293.7 FPS@BS16（score-inclusive target MSE，per-sample [B]；RGB+分割双流，分割图离线预计算）；旧 zero-proxy 12.684ms 保留为对照；
   - HSTforU：168.209G / 64.258ms / 15.5 FPS@BS1（Avenue/PVT2-b2，mmcv-lite 2.x 兼容 shim）；
   - Future Frame：官方 TF1.15 记录 7.704M / 74.476G，latency 留空；TF2.20 runtime-adapted 为 score-inclusive PSNR 6.820ms / 146.8 FPS@BS1 / 177.7 FPS@BS16（prediction-only 5.467ms 保留为对照）；
   - MLEP：官方 TF1.15 记录 12.561M / 319.513G，latency 留空；TF2.20 runtime-adapted 为 score-inclusive PSNR 12.525ms / 79.6 FPS@BS1（scored BS16 未测；prediction-only 11.538ms / 75.5 保留为对照）；
   - 两方法的 TF2 图参数量与 TF1 图逐参数一致；但 FLOPs 仍来自 TF1 profiler、`flops_verified=false`，所以状态保持 `partial/unverified`（runtime-adapted），详见 §3.3。
6. **仍 blocked**：adVAE 只有表格版；Future Frame 官方 FlowNet2/checkpoint 链接失效；两方法的**官方 TF1.15 runtime** 仍受 CUDA10 在 sm_89 上不可执行的限制（TF2 runtime-adapted 已补，但不能冒充官方 TF1.15）。
7. **DwCVAE/CVAE/DCVAE/SCVAE/CLSTMVAE**：用户表速度的硬件/batch 不同，`paper-only`，不能与本机实测混排。

## 7. 证据与下一步

- 计数器小规模验证（ConvTranspose/Conv3d/Attention）与各方法 FLOPs 明细见 [v2_validation.md](v2_validation.md)。
- 逐方法待办与验收条件见 [RETEST_CHECKLIST.md](../RETEST_CHECKLIST.md)；Ours V0–V5、GANomaly、MemAE、MKD、ANDT（adapted 6-frame）、LightViT-AD、MADViT Full/Multi-layer/Last-layer、ST Dissoc.、ASTT、DAD-FSM、HSTforU 已完成；Future Frame/MLEP 的官方 TF1.15 记录仍为 partial（FLOPs 有、latency 未测），TF2.20 runtime-adapted 计时见 §3.3；官方 ANDT decoder、adVAE 仍 blocked。
- 改动原因与哪些旧数字不再正式见 [protocol_change_log.md](../protocol_change_log.md)。
- 旧 JSON/日志/审计文件哈希见 [LEGACY_v1.md](LEGACY_v1.md)。
- v2 结果文件：各方法/变体目录下的 `benchmark_v2.json`；Ours 聚合在 `comparison/da/ours/benchmark_v2_ours.json`（未覆盖任何 v1 文件）。
