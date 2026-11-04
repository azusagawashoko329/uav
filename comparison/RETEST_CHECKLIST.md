# 逐方法待重测清单（v2）

> 本轮只做口径与清单，**不启动 GPU 重测**。所有条目在当前 `comparison/da/comparison_summary.md` §3 中保持
> `partial/unverified` / `blocked` / `paper-only`，直到本清单的验收项全部满足。

## 0. 全局准入（所有方法共用）

- [x] 协议 v2 文档：[BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)
- [x] record 字段定义：[SCHEMA.md](SCHEMA.md) + [schema.json](schema.json) + `check_comparisons.py`
- [x] v1 归档清单与哈希：[da/LEGACY_v1.md](da/LEGACY_v1.md)
- [x] v1 改动记录：[protocol_change_log.md](protocol_change_log.md)
- [ ] v2 脚本改造：`bench_ours.py` / `bench_external.py` 或新增 `bench_v2_*.py`（规范见 [benchmarks/README.md](benchmarks/README.md)）
- [ ] 算子计数器：Conv2d/ConvTranspose2d/Conv3d/ConvTranspose3d/Linear/MatMul/Attention QK/AV
- [ ] 只对 profiler **共同支持**的算子比较且差异 ≤1%；ConvTranspose/Conv3d/融合 Attention 用独立公式核验；**完整 FLOPs 不要求等于 profiler 总数**
- [ ] 防重复计数：module hook / matmul 包装 / SDPA 包装按 `op_id` 去重，同一矩阵乘不能累计两次
- [ ] 时序方法 `score_density=K/stride`；`frame_coverage` 按分数映射/重叠/融合单独计算，不用 K/stride 冒充；没有连续流测试只能写 `fps_stream_estimated`
- [ ] 结果写 `benchmark_v2_<method>.json`，不覆盖任何 v1 文件

## 1. Ours / GLoRD 六个变体

当前状态：`partial/unverified`（v1 profiler-only；旧/新 GLoRD 数字不一致）。

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

## 2. GANomaly

当前状态：`partial/unverified`（参数列已可拆；FLOPs 未做算子覆盖核验）。

- [ ] 推理图只走 G：`inference_params_m=143.950`、`frozen_inference_params_m=0`
- [ ] 训练列 `training_trainable_params_m=188.690`（G+D），`training_only_params_m=44.739`（D）
- [ ] FLOPs 覆盖 G 的 `encoder1 → decoder → encoder2`，含 Conv2d/ConvTranspose2d/Linear；ConvTranspose2d 用独立公式核验（profiler 可能漏计）
- [ ] 明确输入 256×256×1、`T=1/K=1/stride=1`
- [ ] 结果写 `benchmark_v2_ganomaly.json`；AUC 来源仍标 ANDT Table VI / 用户表，不暗示同配置

## 3. MemAE

当前状态：`partial/unverified`（`÷16` 未核实）。

- [ ] 先读原测试代码，确认输出是 clip 级误差还是 `T` 个逐帧分数
- [ ] 若 clip 级：`K=1`，`stride=16`（或原测试步长），不能沿用 `÷16`
- [ ] 若逐帧：`K=16`，`stride=16`，并说明每帧分数如何产生
- [ ] FLOPs 覆盖 Conv3d/ConvTranspose3d + memory 读写；`flops_ops_covered` 不得漏 3D 反卷积；这两个算子是 profiler 盲区，必须用独立公式核验
- [ ] 记录 `window_flops_g`、`window_latency_ms`、`fps_independent_window`、`score_density`、`frame_coverage`、`fps_stream_estimated`；未做连续流测试时 `fps_stream_measured=null`
- [ ] 结果写 `benchmark_v2_memae.json`

## 4. MKD

当前状态：`partial/unverified`（仅最终层距离）。

- [ ] 按原方法实现 source/cloner **多层特征**评分，不能只取最终层
- [ ] 明确输入 128×128×1（原论文 MVTec 配置）；若改 256 跑 DA，`da_adapted=true` 并另列
- [ ] FLOPs 覆盖 source VGG16 + cloner 的 Conv2d/Linear；source 冻结但计入推理参数
- [ ] 参数三列：inference 15.059、training trainable 0.344、training-only 0、frozen 14.715
- [ ] 结果写 `benchmark_v2_mkd.json`

## 5. ANDT

当前状态：`blocked`（发布代码 decoder 只支持 1 帧；论文 6 帧模型跑不通）。

- [ ] 先确认是否有作者完整 6 帧预测代码/decoder；没有则保持 `blocked`
- [ ] 论文配置：`T=6, K=1, stride=1`，输入 256×256×6，预测下一帧
- [ ] 若采用最小 patch（去掉 `pdb`，`output = self.decoder(self.de_dense(feat[:,0]))`），必须标 `adapted`，不能冒充 official
- [ ] 若测 1 帧代码变体，标 `code-variant-1frame`，`T=1`，不能与论文 6 帧 AUC 混表
- [ ] FLOPs 覆盖 Transformer encoder/decoder、patch embedding、时空位置编码；`K=1` 摊销
- [ ] 结果写 `benchmark_v2_andt.json` 前，先在 README 记录代码改动

## 6. adVAE

当前状态：`blocked`（官方仓库是表格版）。

- [ ] 找到 ANDT 作者实际用的 adVAE 视频/图像实现或论文结构说明
- [ ] 在此之前不填参数量/FLOPs/时延；表格版 0.329M 只作说明，不进主表
- [ ] 若拿到实现：输入 H×W×T、K/stride、编码器/解码器/高斯模块全部计入
- [ ] 结果写 `benchmark_v2_advae.json`；否则保持 `blocked`

## 7. 其余外部方法（代码已 clone，未适配）

| 方法 | 当前状态 | 先确认 | 重测重点 |
|---|---|---|---|
| Future Frame | blocked | 输入帧数/预测目标 | `T/K/stride`；光流是否在线；光流单独报 |
| ST Dissoc. | blocked | 输入窗口/静态/动态分支 | 双分支是否都计入；`K/stride` |
| MLEP | blocked | 输入分辨率/时序单元 | 时序编码器；`K/stride` |
| ASTT | blocked | 输入窗口/打分方式 | 时序编码器/预测器/打分；有光流则计入 |
| DAD-FSM | blocked | 输入窗口/打分方式 | 同上；双流/光流单列 |
| HSTforU | blocked | 输入窗口/层级 | 时序/层级打分；`K/stride` |
| MADViT Full / Multi-layer / Last-layer | blocked | 三个变体的层选择 | 同一 ViT 主干的 tap 集合；AUC 与效率配置对应 |
| LightViT-AD | blocked | 输入分辨率/轻量主干 | 主干+AD 头；缺 vehicle/crossroads 的 AUC 要标注 |

- [ ] 每个方法先写适配 README：输入 H×W×T、T/K/stride、是否光流、是否 DA 适配、AUC 来源
- [ ] 再按 §0 全局准入重测；未适配前不填估计值

## 8. Paper-only 方法（不重测，单独表）

- DwCVAE / CVAE / DCVAE / SCVAE / CLSTMVAE：用户表数字来自不同论文/硬件/batch，标 `paper-only`。
- [ ] 不放进 5880 Ada 本机实测排序；只放在“论文报告值”附表
- [ ] 若后续拿到代码，再按 v2 重测并升级状态

## 9. 完成顺序建议

1. 先做 **Ours + GANomaly + MemAE + MKD**（代码在本地，能最快暴露计数器/时序口径问题）；
2. 再处理 **ANDT**（先决定 blocked / adapted / 1-frame 变体）；
3. 其余外部方法按代码适配难度逐个补；
4. 全部完成后，才把 `comparison_summary.md` §3 的数字填上并把状态改为 `measured`。
