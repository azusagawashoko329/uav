# 逐方法待重测清单（v2）

> **公平性复核**：`measured` 只表示测速完成，不等同于 `paper-matched`；逐方法证据和需脚注/需重测清单见 [da/fairness_audit.md](da/fairness_audit.md)。
>
> **2026-11-04 执行结果**：Ours V0–V5、GANomaly、MemAE、MKD、ANDT(adapted 6-frame)、LightViT-AD、
> MADViT Full/Multi-layer/Last-layer、ST Dissoc.、ASTT、DAD-FSM、HSTforU 已完成 v2 正式测量（50 warmup + 3×500），
> 见 [da/comparison_summary.md](da/comparison_summary.md) §3 与 [da/v2_validation.md](da/v2_validation.md)。
> Future Frame / MLEP 的官方 TF1.15 记录为 `partial/unverified`（params/FLOPs 有，GPU latency 未按协议完成）；已补 TF2.20 runtime-adapted 计时（Future Frame 5.467ms / 180.8 FPS@BS1；MLEP 11.538ms / 82.0 FPS@BS1），仍标 `partial/unverified`，不能冒充官方 TF1.15；adVAE 仍 `blocked`。

## 0. 全局准入（所有方法共用）

- [x] 协议 v2 文档：[BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)
- [x] record 字段定义：[SCHEMA.md](SCHEMA.md) + [schema.json](schema.json) + `check_comparisons.py`
- [x] v1 归档清单与哈希：[da/LEGACY_v1.md](da/LEGACY_v1.md)
- [x] v1 改动记录：[protocol_change_log.md](protocol_change_log.md)
- [x] v2 脚本：`bench_v2_ours.py` / `bench_v2_external.py` / `bench_v2_common.py`（规范见 [benchmarks/README.md](benchmarks/README.md)）
- [x] 算子计数器：Conv2d/ConvTranspose2d/Conv3d/ConvTranspose3d/Linear/MatMul/Attention QK/AV + native MHA
- [x] 只对 profiler **共同支持**的算子比较；ConvTranspose/Conv3d/融合 Attention 用独立公式核验；**完整 FLOPs 不要求等于 profiler 总数**
- [x] 防重复计数：module hook / matmul 包装 / SDPA 包装分开，手写公式与主计数器逐模块对齐
- [x] 时序方法 `score_density=K/stride`；`frame_coverage` 按分数映射/重叠/融合单独计算，不用 K/stride 冒充；没有连续流测试只能写 `fps_stream_estimated`
- [x] 结果写 `<Method>/<variant>/benchmark_v2.json`，不覆盖任何 v1 文件

## 1. Ours / GLoRD 六个变体（已完成 v2 测量）

当前状态：`measured`（V0–V5 全部通过 `flops_verified=true`）。

| 变体 | 当前输入 | 待确认 | 重测动作 |
|---|---|---|---|
| V0 `rd_s12_t4_h512` | 256×256×1 | teacher taps `[5,7,9,11]`；FLOPs 90.205 vs 95.107；latency 17.800 vs 10.184 | 记录 4 taps，但仍前向 `0..11`；分模块计 teacher/OCBE/student/readout；共同子集对 profiler、盲区用公式；BS1/BS16 重跑 |
| V1 `rd_s4_t12_h512` | 256×256×1 | 12 taps `0..11` | 同 V0 流程 |
| V2 `rd_s4_t12_gl768_h512` | 256×256×1 | GL768 分支是否全部计入 | 同 V0 流程 |
| V3 `rd_s4_t12_gl192_h512` | 256×256×1 | GL192 分支 | 同 V0 流程 |
| V4 `rd_s4_t12_gl192_h128` | 256×256×1 | OCBE h128 | 同 V0 流程 |
| V5 `rd_s4_t12_gl192_h128_qr2` | 256×256×1 | FLOPs 46.572 vs 49.106；latency 19.638 vs 旧审计 ~9.7 | 同 V0 流程；单独记录 Q rank-2 |

- [ ] 所有变体 `inference_params_m = teacher + OCBE + student`，`training_trainable_params_m = OCBE + student`，`training_only_params_m = 0`，`frozen_inference_params_m = teacher`
- [ ] 记录 `teacher_taps`，并确认 **tap 不是计算裁剪**：取 `[5,7,9,11]` 仍需前向 `0..11` 的所有 block，FLOPs 按实际前向层数计，不能按“4 taps × 单层”算；能省的是未用 tap 的特征保存/后处理，不是得到第 11 层所必需的前向
- [ ] 确认 `anomaly_readouts` 的 top-k/距离是否计入 FLOPs；纯排序不计 MAC 但计入时延
- [ ] 解决 V0/V5 新旧 FLOPs 与 V0 latency 差异；在解决前两组都只能 `partial/unverified`
- [ ] AUC 仍引用 `experiments/da/appearance/20261103_variant_matrix_*`，效率配置单独写 `architecture-init`

## 2. GANomaly（已完成 v2 测量）

当前状态：`measured`（含 ConvTranspose2d；16.428G）。

- [ ] 推理图只走 G：`inference_params_m=143.950`、`frozen_inference_params_m=0`
- [ ] 训练列 `training_trainable_params_m=188.690`（G+D），`training_only_params_m=44.739`（D）
- [ ] FLOPs 覆盖 G 的 `encoder1 → decoder → encoder2`，含 Conv2d/ConvTranspose2d/Linear；ConvTranspose2d 用独立公式核验（profiler 可能漏计）
- [ ] 明确输入 256×256×1、`T=1/K=1/stride=1`
- [ ] 结果写 `benchmark_v2_ganomaly.json`；AUC 来源仍标 ANDT Table VI / 用户表，不暗示同配置

## 3. MemAE（已完成 v2 测量）

当前状态：`measured`；`K=1, stride=1`，全窗口 70.808G。

- [ ] 先读原测试代码，确认输出是 clip 级误差还是 `T` 个逐帧分数
- [ ] 若 clip 级：`K=1`，`stride=16`（或原测试步长），不能沿用 `÷16`
- [ ] 若逐帧：`K=16`，`stride=16`，并说明每帧分数如何产生
- [ ] FLOPs 覆盖 Conv3d/ConvTranspose3d + memory 读写；`flops_ops_covered` 不得漏 3D 反卷积；这两个算子是 profiler 盲区，必须用独立公式核验
- [ ] 记录 `window_flops_g`、`window_latency_ms`、`fps_independent_window`、`score_density`、`frame_coverage`、`fps_stream_estimated`；未做连续流测试时 `fps_stream_measured=null`
- [ ] 结果写 `benchmark_v2_memae.json`

## 4. MKD（已完成 v2 测量）

当前状态：`measured`；已用 taps 6/9/12 多层评分。

- [ ] 按原方法实现 source/cloner **多层特征**评分，不能只取最终层
- [ ] 明确输入 128×128×1（原论文 MVTec 配置）；若改 256 跑 DA，`da_adapted=true` 并另列
- [ ] FLOPs 覆盖 source VGG16 + cloner 的 Conv2d/Linear；source 冻结但计入推理参数
- [ ] 参数三列：inference 15.059、training trainable 0.344、training-only 0、frozen 14.715
- [ ] 结果写 `benchmark_v2_mkd.json`

## 5. ANDT（adapted 6-frame 已测；official 仍 blocked）

当前状态：`adapted 6-frame measured`；官方 6 帧 decoder 仍不可用。

- [x] 采用最小 patch（去掉 `pdb`/`.cuda()`，`output = self.decoder(self.de_dense(feat[:,0]))`），明确标 `adapted`
- [x] 论文配置：`T=6, K=1, stride=1`，输入 256×256×6，预测下一帧
- [x] FLOPs 覆盖 Transformer encoder/decoder、patch embedding、de_dense、decoder conv；`K=1`
- [x] 结果：70.347M / 74.490G / 4.470ms / 223.6 FPS@BS1；写 `ANDT/default/benchmark_v2.json`
- [x] README 记录代码改动与 adapted 限制
- [ ] 官方 6 帧代码/decoder 仍待确认；确认前不能把 adapted 数字当 official
- [ ] 1 帧代码变体（若需要）单独标 `code-variant-1frame`，不能与论文 6 帧 AUC 混表

## 6. adVAE

当前状态：`blocked`（官方仓库是表格版）。

- [ ] 找到 ANDT 作者实际用的 adVAE 视频/图像实现或论文结构说明
- [ ] 在此之前不填参数量/FLOPs/时延；表格版 0.329M 只作说明，不进主表
- [ ] 若拿到实现：输入 H×W×T、K/stride、编码器/解码器/高斯模块全部计入
- [ ] 结果写 `benchmark_v2_advae.json`；否则保持 `blocked`

## 7. 其余外部方法（部分已完成 v2）

| 方法 | 当前状态 | 结果/限制 |
|---|---|---|
| Future Frame | partial/unverified | 官方 TF1.15：7.704M / 74.476G，GPU latency 受 CUDA10/sm_89 限制；TF2.20 runtime-adapted：5.467ms / 180.8 FPS@BS1 / 185.1 FPS@BS16（`benchmark_v2_tf2compat.json`） |
| ST Dissoc. | measured | 60.612G / 3.668ms / 269.7 FPS@BS1；`padding=False→0` 兼容修正 |
| MLEP | partial/unverified | 官方 TF1.15：12.561M / 319.513G，GPU latency 受 CUDA10/sm_89 限制；TF2.20 runtime-adapted：11.538ms / 82.0 FPS@BS1 / 75.5 FPS@BS16（`benchmark_v2_tf2compat.json`） |
| ASTT | measured | STE-TTE：109.416G / 19.048ms / 52.2 FPS@BS1；无 checkpoint |
| DAD-FSM | measured | 71.194G / 12.684ms / 78.5 FPS@BS1；RGB+分割双流，ResNet50 随机初始化 |
| HSTforU | measured | 168.209G / 64.258ms / 15.5 FPS@BS1；mmcv-lite 2.x + 兼容 shim |
| MADViT Full | measured | code default：38.533G / 11.525ms / 83.9 FPS@BS1 |
| MADViT Multi-layer / Last-layer | measured (adapted scoring) | 按论文 Sec. 5.4：Multi=全层 feature only；Last=final token only；同一前向图 |
| LightViT-AD | measured | 3.753G / 6.519ms / 148.8 FPS@BS1；teacher 冻结、student 可训练 |

- [x] 已测方法都有输入 H×W×T、T/K/stride、权重状态、AUC 来源和 benchmark JSON
- [x] MADViT 三变体已核对：代码只实现 Full-like 路径；Multi/Last 已按论文 Sec. 5.4 实现 adapted scoring 并完成测量
- [x] Future Frame/MLEP 已补 TF2.20 runtime-adapted 计时（官方 TF1.15 latency 仍受 CUDA10/sm_89 限制）；adVAE 仍待视频实现

## 8. Paper-only 方法（不重测，单独表）

- DwCVAE / CVAE / DCVAE / SCVAE / CLSTMVAE：用户表数字来自不同论文/硬件/batch，标 `paper-only`。
- [ ] 不放进 5880 Ada 本机实测排序；只放在“论文报告值”附表
- [ ] 若后续拿到代码，再按 v2 重测并升级状态

## 9. 完成顺序建议

1. 先做 **Ours + GANomaly + MemAE + MKD**（代码在本地，能最快暴露计数器/时序口径问题）；
2. 再处理 **ANDT**（先决定 blocked / adapted / 1-frame 变体）；
3. 其余外部方法按代码适配难度逐个补；
4. 全部完成后，才把 `comparison_summary.md` §3 的数字填上并把状态改为 `measured`。
