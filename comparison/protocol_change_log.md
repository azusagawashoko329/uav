# 轻量化协议修改记录：v1 → v2

- **日期**：2026-11-04
- **范围**：`comparison/BENCHMARK_PROTOCOL.md`、`comparison/SCHEMA.md`、`comparison/schema.json`、`comparison/check_comparisons.py`、`comparison/build_catalog.py`、`comparison/da/comparison_summary.md`、新增归档/重测清单。
- **本轮不重测**：没有运行任何 GPU benchmark；旧 `benchmark_*.json`、`*.log`、旧审计文件均未改动（哈希见 [da/LEGACY_v1.md](da/LEGACY_v1.md)）。
- **结论**：v1 的轻量化数字全部降级为历史归档；v2 主表暂时留空，逐方法重测见 [RETEST_CHECKLIST.md](RETEST_CHECKLIST.md)。

## 1. 逐条改动

| # | 改动 | v1 | v2 | 为什么改 | 受影响的旧数字 |
|---|---|---|---|---|---|
| 1 | 参数列拆分 | `total_params_m` / `trainable_params_m` | `inference_params_m` / `training_trainable_params_m` / `training_only_params_m`（可选 `frozen_inference_params_m`） | `Total < Trainable` 的歧义：GANomaly 推理只用 G，训练却更新 G+D | GANomaly 143.950/188.690；所有方法的 Total/Trainable 列 |
| 2 | 冻结教师位置 | 有的表把教师算进 Total，有的只算学生 | 教师计入 Inference，不计入 Training Trainable；单列 `frozen_inference_params_m` | 避免“Total”既当推理又当训练 | Ours V0–V5 的 174.720/89.079 等旧列 |
| 3 | FLOPs 唯一依据 | `torch.profiler(with_flops=True)` | 算子计数器 + 独立公式；profiler 只核验共同支持子集；`1 MAC=2 FLOPs`；记录 covered/missing | profiler 会漏 ConvTranspose/Conv3d/融合 Attention/自定义算子，不能单独当 Full FLOPs，也不能用它的总数做总验收 | 所有 v1 `flops_g*` 数字 |
| 4 | FLOPs 不完整处置 | 不完整值也写成 `FLOPs` | `flops_verified=false` + `flops_ops_missing` + `partial/unverified` | 严禁把不完整值冒充 Full FLOPs | MKD 10.477G；Ours 90.205/46.572G 等 |
| 5 | 时序计量 | 多数直接 `÷T`，默认 `K=T` | 分开 `T`（输入帧）、`K`（有效输出分数）、`stride`；窗口值 + 摊销值；独立窗口吞吐 + 真实流吞吐 | MemAE/ANDT 等 `K≠T`，`÷16` 会虚增/虚减 | MemAE 0.066G/0.437ms/2287FPS；ANDT 的窗口摊法 |
| 6 | 完整推理范围 | 有的只算学生/最终层 | 统一从 GPU 张量到最终分数，含 Teacher/Backbone/Decoder/Scoring；离线光流/缓存/预处理单列 | 避免“只算子网”与“完整模型”混表 | Ours 旧 3.319ms/47.958G（仅学生）vs 17.800ms/90.205G（含教师）不能混用 |
| 7 | MKD 评分 | 最终层特征距离 | 原方法多层特征评分 | 最终层距离不是完整 MKD | MKD 10.477G/2.273ms 降级 `partial/unverified` |
| 8 | 来源与配置 | 只写 AUC 来源 | 每行写 `code_commit`、`input_hwt`、`weights_state`、`da_adapted`、`auc_source`、`profiling_config` | AUC 可能是 ANDT DA 适配版，效率可能是原仓库默认模型，不能暗示同一配置 | 所有跨表引用 |
| 9 | Latency/FPS 统计量 | P50 与 FPS 关系未写清 | Latency 主指标 P50；`FPS@BS1 = 1000/Mean`；保存 Mean/P50/P95 + 原始耗时 | P50 和 Mean 不能互相反算后声称一致 | 所有 FPS@BS1 行 |
| 10 | Status | `measured` 混用 | `measured` / `partial/unverified` / `paper-only` / `blocked`，并给出硬性条件 | 让“有数字”不等于“可正式比较” | 所有 v1 `measured` 行 |
| 11 | 旧数据处置 | 新结果覆盖同文件 | v1 原样冻结在 `<Method>/<variant>/legacy_v1/`；v2 写 `<Method>/<variant>/benchmark_v2.json`，带 `supersedes`/`measurement_status` | 保留审计链，不静默改历史 | 现有 `benchmark_*.json` 全部 |
| 12 | 表头 | 单一 Total/FLOPs/Latency 表 | 主表 + 时序附表 + 配置/来源附表 | 让 T/K/stride、commit、权重状态有位置 | `comparison_summary.md` 轻量化表 |

## 2. 参数量：为什么必须拆三列

v1 的 `total_params_m` 实际有两种含义：

- GANomaly：`143.950M` 是**推理图 G**；`trainable_params_m=188.690M` 是**训练 G+D**。两个数不在同一口径，旧表写成 `Total=143.950`、`Trainable=188.690` 会让人误以为训练参数比模型总参数还大是错误。
- Ours：`total_params_m=174.720M` 是 `teacher + OCBE + student`；`trainable_params_m=89.079M` 是 `OCBE + student`。教师必须计入推理，但不能计入训练可更新。

v2 定义：

- `inference_params_m`：推理图实际执行的全部参数，**含冻结教师**，不含 training-only D。
- `training_trainable_params_m`：训练时 `requires_grad=True` 的全部参数，**含 D**，不含冻结教师。
- `training_only_params_m`：只训练使用、推理不执行的参数（D），是 Training Trainable 的子集。
- `frozen_inference_params_m`：Inference 中的冻结部分，仅作透明拆分。
- 规则：`training_only <= training_trainable`；`frozen <= inference`；不要求 `inference` 与 `training_trainable` 有大小关系。

**旧数字处置**：GANomaly 的 143.950/188.690 不再叫 Total/Trainable；改写成 143.950（G 推理）/188.690（G+D 训练）/44.739（D）。Ours 的 174.720/89.079 保留为结构审计，但列名改为 Inference/Training Trainable，并单列教师 85.641。

## 3. FLOPs：为什么 profiler 不能单独作数

v1 的唯一依据是：

```python
with profile(..., with_flops=True) as prof:
    full(x)
flops = sum(e.flops or 0 for e in prof.events())
```

问题：

- profiler 对 `ConvTranspose2d/ConvTranspose3d`、融合 `scaled_dot_product_attention`、自定义 kernel、部分 functional 调用的覆盖不完整；
- `torch.inference_mode()` 下可能拿不到 FLOPs（本仓库 v1 曾出现 0.0）；
- 只给一个总数，无法知道漏了哪些算子，也无法和手工公式对账。

v2 要求：

1. `1 MAC = 2 FLOPs`；
2. 模块 hook 覆盖 `Conv2d/ConvTranspose2d/Conv3d/ConvTranspose3d/Linear`，并包装 `matmul/bmm/einsum/scaled_dot_product_attention`；
3. profiler 只与计数器**共同支持的算子**比较，共同子集差异 ≤1%；不能要求完整计数器总数等于 profiler 总数；
4. `ConvTranspose2d/3d`、`Conv3d`、融合 Attention、自定义 kernel 等 profiler 盲区，用计数器 + 独立公式核验；
5. 完整 FLOPs = `Σ(所有已验证 MAC 算子 MACs) × 2`；不要求等于 profiler 总数；
6. 防重复计数：module hook / matmul 包装 / SDPA 包装按 `op_id` 去重，同一矩阵乘不能累计两次；
7. 有算子无法核验时 `flops_verified=false`、`flops_ops_missing` 列出、`Status=partial/unverified`；
8. 保存 `flops_breakdown`（逐模块/算子 MACs）和 `flops_common_subset_check` / `flops_duplicate_check`。

**旧数字处置**：所有 v1 `flops_g*` 不再直接作为正式 Full FLOPs；MKD 的 10.477G 因只算最终层距离，直接降级；Ours 的 90.205/46.572G 因未做算子覆盖核验且与旧审计不一致，降级 `partial/unverified`。

## 4. 时序：为什么不能默认 `K=T`

v1 对外部时序方法常用“窗口 FLOPs / T”或“窗口时延 / T”。这只在“每个输入帧都产出一个分数”时成立；实际：

- ANDT 论文 6 帧模型输入 6 帧，只预测**下一帧**，`K=1`，不是 6。
- MemAE 代码返回 `B×T` 残差，但原论文测试可能聚合成 clip 级误差，`K` 可能是 1；旧 `÷16` 未核实。
- 重叠窗口 `stride<T` 时，独立窗口吞吐（输出分数/秒）和真实视频流吞吐（原始帧/秒）不是同一个数；`K<stride` 时还有帧没有分数。

v2 定义：

- `T`：窗口输入帧数；
- `K`：每窗口有效输出分数数；
- `stride`：窗口前进的原始帧数；
- `window_flops_g` / `window_latency_ms`：窗口值；
- `flops_g_per_output_frame = window_flops_g / K`；
- `latency_*_ms_per_output_frame = window_latency_ms / K`；
- `fps_independent_window = K · 1000 / mean_window_latency_ms`；
- `fps_stream_estimated = stride · 1000 / mean_window_latency_ms`（只是按窗口步长推算的估计；没有连续流测试不能写 measured）；
- `score_density = K / stride`（每输入帧平均分数个数，可 >1，不是帧覆盖率）；
- `frame_coverage` = 实际收到至少一个分数的原始帧比例（0..1），按分数映射/重叠/融合单独计算，不能由 K/stride 推出。

**旧数字处置**：MemAE 的 0.066G/0.437ms/2287FPS 全部标 `partial/unverified`，先核对原测试 K；ANDT 标 `T=6,K=1,stride=1`，但代码 blocked；其余时序方法在适配前不填 K。

## 5. 完整推理范围

v1 有两类不可比数字：

- Ours 旧表 3.319ms/47.958G 只算学生子网；新表 17.800ms/90.205G 含 DINOv3 教师。两者不能混用。
- MKD 只取最终层距离，不是原方法多层评分。

v2 统一：

- 起点：GPU 上预处理后的输入张量；
- 终点：最终异常分数；
- 包含 Teacher/Backbone、时序编码器/预测器、Decoder、Scoring/检测头；在线光流计入；
- 不包含 JPEG 解码、CPU→GPU、磁盘 I/O、离线光流/缓存、测试时预处理；这些单独报；
- 分模块记录，便于核对“到底算了什么”。

**旧数字处置**：Ours 旧 3.319ms/47.958G 只作历史；MKD 10.477G 降级；所有“只算子网”的数字不得进入 v2 主表。

## 6. AUC 与效率配置分离

v1 表里 AUC 和效率并排，容易让人以为效率数字就是 AUC 那个模型。实际：

- AUC 可能是 ANDT 论文统一协议下的 DA 适配版；
- 效率可能来自原仓库默认输入/层数/head；
- 例如 MKD AUC 来自 DA 适配，效率按 MVTec 128×128 原配置；
- GANomaly/MemAE 的 AUC 来自 ANDT 表，效率是本机原仓库配置。

v2 每行必须写：

- `code_commit`（完整 SHA）
- `input_hwt`
- `weights_state`（scratch/pretrained/DA-trained/architecture-init）
- `da_adapted`
- `auc_source`（具体表/实验 ID）
- `profiling_config`（GPU/精度/BS/warmup/iters/rounds/同步）
- `benchmark_source`（v2 JSON 路径）

**旧数字处置**：v1 表没有这些字段，不能证明 AUC 与效率同配置，全部不能作为 v2 正式行。

## 7. Latency/FPS 统计量

v1 表头写 `Latency@BS1 P50` 和 `FPS@BS1`，但没有说明 FPS 是从 Mean 还是 P50 反算。v2 明确：

- Latency 主指标：`P50`；
- 同时保存 `Mean`、`P95`、每轮原始耗时；
- `FPS@BS1 = 1000 / Mean_latency_ms`；
- BS16：`16·K·500 / median(round_time_s)`，只在整轮结束同步；
- 时序方法另报 `window_latency_ms`、`fps_independent_window`、`score_density`、`frame_coverage`、`fps_stream_estimated`；只有实际连续流测试才填 `fps_stream_measured`。

**旧数字处置**：v1 的 FPS@BS1 虽然代码里用的是 Mean，但表头/文档没写清；保留数字，但在 v2 中重新生成并同时保存 Mean/P50/P95。

## 8. Status 细分

v1 的 `measured` 只表示“跑过脚本”，不表示“口径完整”。v2：

- `measured`：所有 MAC 算子都有计数器 + 独立公式核验（不要求与 profiler 总数相等）、参数三列完整、T/K/stride 与 `score_density`/`frame_coverage`/`fps_stream_estimated` 已核实、原始耗时保存、配置/来源齐全；
- `partial/unverified`：有数字但上述任一未完成；
- `paper-only`：论文/用户表数字，未本机同协议实测；
- `blocked`：代码/decoder/配置缺失，不填猜测值。

**旧数字处置**：v1 所有 `measured` 行按 v2 应视为 `partial/unverified`（Ours/GANomaly/MemAE/MKD）或 `blocked`（adVAE/ANDT）。

## 9. 哪些旧数字不再可作为正式对比结果

以下数字**只保留在 [da/comparison_summary.md](da/comparison_summary.md) §5 与旧 JSON/日志中**，不得用于论文/报告主表，直到 v2 重测：

1. GANomaly `10.952G / 13.357ms / 74.9 FPS / 622.7 FPS@BS16`：FLOPs 未覆盖核验；参数列已重定义。
2. MemAE `0.066G / 0.437ms / 2287 FPS / 2193 FPS@BS16`：`K=T=16` 未核实。
3. MKD `10.477G / 2.273ms / 442.9 FPS / 3554.4 FPS@BS16`：只算最终层距离，不是完整 MKD。
4. Ours V0–V5 的 `90.205/65.240/60.119/48.595/46.771/46.572G` 与对应 latency/FPS：profiler-only；V0/V5 还与旧审计不一致。
5. ANDT 参数量 70.347M：结构可数，但 decoder 只支持 1 帧，FLOPs/latency 仍 `blocked`。
6. adVAE 0.329M：表格版，不是 DA 视频模型，主表 `blocked`。
7. DwCVAE/CVAE/DCVAE/SCVAE/CLSTMVAE 的用户表速度：不同硬件/batch，`paper-only`，不能与本机 5880 Ada 混排。
8. Ours 旧表 `3.319ms / 301.3 FPS / 47.958G`（仅学生）：与新表含教师的口径不同，不能混用。

## 10. GLoRD 新旧差异（待复核）

| 项 | 新测 2026-11-04 | 旧审计 2026-10-26 | 差异 |
|---|---:|---:|---:|
| V0 FLOPs | 90.205G | 95.107G | 4.902G（约 5.2%） |
| V5 FLOPs | 46.572G | 49.106G | 2.534G（约 5.2%） |
| V0 BS1 latency | 17.800ms | 10.184ms | 7.616ms |

- 两个变体的 FLOPs 差异比例几乎相同（约 5.2%），更像是某个系统性算子/范围差异，而不是单变体噪声；但不能据此预设哪边正确。
- 复核要求：同一 teacher tap 数、同一 scoring、同一输入，分别做算子计数、共同子集 profiler 核验、分模块计时；记录 `teacher_taps`、是否含 `anomaly_readouts`、是否含教师全部层。
- **tap 不是计算裁剪**：V0 取 `[5,7,9,11]` 仍必须前向 `0..11` 才能得到第 11 层特征；FLOPs 按实际前向层数计，不能按 4 taps × 单层。能省的是未用 tap 的保存/后处理，不是必需的前向。
- 复核前两组数字都标 `partial/unverified`，不选一个当正式值。

## 11. 文件变更与未变更

**变更（本轮）**：

- `comparison/BENCHMARK_PROTOCOL.md`：重写为 v2。
- `comparison/SCHEMA.md`、`comparison/schema.json`、`comparison/check_comparisons.py`：新增可选 `efficiency` 对象及校验。
- `comparison/build_catalog.py`：CATALOG/INDEX 增加轻量化列。
- `comparison/da/comparison_summary.md`：v2 空表 + 时序附表 + 配置/来源表 + v1 归档区。
- 新增 `comparison/da/LEGACY_v1.md`、`comparison/da/LEGACY_v1.sha256`。
- 新增 `comparison/RETEST_CHECKLIST.md`、`comparison/benchmarks/README.md`、本文件。

**未变更（按用户要求原样归档）**：

- `comparison/da/ours/legacy_v1/benchmark_ours.json`、`benchmark_ours.log`
- `comparison/da/GANomaly/default/legacy_v1/benchmark_ganomaly.json`、`.log`
- `comparison/da/MemAE/default/legacy_v1/benchmark_memae.json`、`.log`
- `comparison/da/MKD/default/legacy_v1/benchmark_mkd.json`、`.log`
- `experiments/da/audit/20261026_lightweight_profile_fp32_s42/results/summary.{json,csv,md}`

验证：

```bash
sha256sum -c comparison/da/LEGACY_v1.sha256
```

## 12. 下一步

1. 先按 [RETEST_CHECKLIST.md](RETEST_CHECKLIST.md) 改造 v2 脚本/算子计数器；
2. 先测 Ours + GANomaly + MemAE + MKD，验证口径；
3. 再处理 ANDT/adVAE 的 blocked 状态；
4. 所有数字通过 §0 验收后，才把 `comparison_summary.md` §3 填上并改 `measured`。

## 13. 2026-11-04 追加修订（复核反馈）

### 13.1 `score_coverage` → `score_density`，新增 `frame_coverage`

- **问题**：`score_coverage = K/stride` 只能描述输出分数密度，不能代表视频帧覆盖率。例：`T=16, stride=1, K=16` 时 `K/stride=16`，但真实帧覆盖率最多 100%，不可能是 1600%。
- **修正**：
  - 字段改名 `score_density`，定义仍为 `K/stride`，允许 >1；
  - 新增 `frame_coverage`：实际收到至少一个分数的原始帧比例（0..1），必须按原方法的分数映射、窗口重叠和融合方式计算，不能由 `K/stride` 推出；
  - `check_comparisons.py` 不再检查“覆盖率 == K/stride”，改为检查 `score_density == K/stride`、`frame_coverage ∈ [0,1]`。
- **涉及文件**：`BENCHMARK_PROTOCOL.md`、`SCHEMA.md`、`schema.json`、`check_comparisons.py`、`da/comparison_summary.md`、`RETEST_CHECKLIST.md`、`benchmarks/README.md`。

### 13.2 `fps_stream` → `fps_stream_estimated`，新增 `fps_stream_measured`

- **问题**：`stride × 1000 / window_latency` 只是按窗口步长推算的吞吐能力；没有实际执行连续视频流测试，不能声称是实测真实流吞吐。
- **修正**：
  - 字段改名 `fps_stream_estimated`，公式不变，明确是估计值；
  - 新增 `fps_stream_measured`：只有真正跑了连续流测试才填，否则 `null`；
  - 主表/时序附表/脚本规范/检查器同步改名。
- **涉及文件**：同上。

### 13.3 FLOPs 验收：不再要求完整总数与 profiler 总数差 ≤1%

- **问题**：profiler 会漏 `ConvTranspose2d/3d`、`Conv3d`、融合 Attention、自定义 kernel；MemAE 这类模型即使计数器完全正确，两个工具的总数也可能相差几十倍。强制总数 ≤1% 会让正确结果无法通过。
- **修正**：
  - 只对计数器与 profiler **共同支持**的算子比较，共同子集差异 ≤1%；
  - profiler 盲区算子用计数器 + 独立公式核验；
  - 完整 FLOPs = `Σ(所有已验证 MAC 算子 MACs) × 2`，不要求等于 profiler 总数；
  - 新增 `flops_common_subset_check`、`flops_duplicate_check`；按 `op_id` 去重，防止 module hook / matmul / SDPA 包装重复累计同一矩阵乘；
  - `measured` 要求 `flops_verified=true`、`flops_ops_missing=[]`，但不再要求 profiler 总数一致。
- **涉及文件**：`BENCHMARK_PROTOCOL.md`、`SCHEMA.md`、`schema.json`、`check_comparisons.py`、`RETEST_CHECKLIST.md`、`benchmarks/README.md`。

### 13.4 Teacher taps 不是计算裁剪

- **问题**：V0 取 `[5,7,9,11]` 不等于只运行 4 个 Transformer Block；要得到第 11 层特征，仍必须前向 `0..11`。
- **修正**：在协议、重测清单、脚本规范、总表复核项中明确：
  - tap 是输出选择，不是计算裁剪；
  - FLOPs 按实际前向层数计，不能按“4 taps × 单层”；
  - 能优化的是未用 tap 的特征保存/后处理，不是得到第 11 层所必需的前向。
- **涉及文件**：`BENCHMARK_PROTOCOL.md`、`RETEST_CHECKLIST.md`、`benchmarks/README.md`、`da/comparison_summary.md`、本文件。

## 14. 2026-11-04 v2 实测完成

新增：

- `comparison/benchmarks/flops_counter.py`：FlopCounterMode + native MHA 公式 + 手写模块公式 + profiler 共同子集 + 去重检查。
- `comparison/benchmarks/bench_v2_common.py`：50 warmup、BS1 3×500、BS16 3×500、JSON 输出。
- `comparison/benchmarks/bench_v2_ours.py`、`bench_v2_external.py`、`bench_v2_andt.py`：只写对应方法目录下的 `benchmark_v2.json`。
- `comparison/da/ANDT/default/scripts/andt_adapted_model.py`：ANDT 6 帧最小适配（`de_dense(feat[:,0])`）。
- `comparison/da/v2_validation.md`：玩具算子 + GANomaly/MemAE/GLoRD V0 的验证记录。

验证结论：

- GANomaly：16.427778048G；手写 21 模块误差 0；profiler 只报 10.951852032G（漏 ConvTranspose2d）。
- MemAE：70.808240128G/窗口；手写 8 模块误差 0；profiler 只报 1.050624G（漏 Conv3d/ConvTranspose3d）。
- GLoRD V0：95.106764800G；手写 92 模块误差 0；profiler 90.204910640G（漏 student native MHA）；与旧审计 95.107G 一致。
- 全部 `flops_ops_missing=[]`、`flops_verified=true`；没有把 profiler 总数当 Full FLOPs。

正式结果（各方法/变体目录的 `benchmark_v2.json`）：

| Method | FLOPs G/output | P50 ms | FPS BS1 | FPS BS16 |
|---|---:|---:|---:|---:|
| GANomaly | 16.428 | 8.088 | 123.6 | 848.5 |
| MemAE | 70.808 | 7.321 | 136.5 | 130.3 |
| MKD | 10.477 | 2.495 | 394.5 | 2340.8 |
| ANDT adapted 6-frame | 74.490 | 4.470 | 223.6 | 210.4 |
| Ours V0 | 95.107 | 14.521 | 68.4 | 187.4 |
| Ours V1 | 68.532 | 19.903 | 50.0 | 244.3 |
| Ours V2 | 62.804 | 14.787 | 67.3 | 263.3 |
| Ours V3 | 51.132 | 19.289 | 51.5 | 314.8 |
| Ours V4 | 49.308 | 19.417 | 51.3 | 324.7 |
| Ours V5 | 49.106 | 20.513 | 48.5 | 324.4 |

已解决/关闭：

- GLoRD V0/V5 新旧 FLOPs 差异：v2 为 95.107/49.106，与旧审计一致；v1 的 90.205/46.572 是 profiler 漏计 native MHA。
- MemAE `K`：确认为 `K=1, stride=1`，旧 `÷16` 废弃。
- MKD：已改为 taps 6/9/12 多层评分，旧最终层距离废弃。
- ANDT：官方 6 帧 decoder 仍不可用；已用 adapted `de_dense(feat[:,0])` 测 `T=6,K=1,stride=1`，70.347M / 74.490G / 4.470ms / 223.6 FPS@BS1，标 adapted，不冒充 official。

仍未完成：

- 官方 ANDT 6 帧 decoder、adVAE（表格版）、Future Frame/ST Dissoc./MLEP/ASTT/DAD-FSM/HSTforU/MADViT/LightViT-AD（未适配）保持 `blocked`。
- DwCVAE/CVAE/DCVAE/SCVAE/CLSTMVAE 保持 `paper-only`。
