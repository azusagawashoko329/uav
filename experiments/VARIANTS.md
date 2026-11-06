# 反向蒸馏结构变体 ID

本文件是六个反向蒸馏（RD）外观分支结构变体的唯一命名入口。**变体 ID 只描述模型结构与超参，不包含数据集、种子、训练轮数或 checkpoint**；定位一次具体训练结果时，必须再给出 `experiment ID + dataset + seed + run_id`。这样同一个结构在 DA / UIT 上的不同运行不会被混成同一个模型。

**首批正式六变体矩阵**：`experiments/uit/appearance/20261101_uit_45ep_*_retry2_s42/`，UIT-ADrone，seed42，45 epoch，batch64，AdamW，256×256，冻结 DINOv3 ViT-B/16，`source_commit=532e0e024c1d697d5f45ff1b3aed95487f85fbc1`；六项 `record.json` 均为 `completed`。

**20261104 matrix 命名重跑**：`experiments/uit/appearance/20261104_variant_matrix_v{0..5}_*_s42/`，同样的六变体/45ep/seed42/双卡动态队列，使用当前源码与训练前 variant guard；六项均为 `completed`，matched-score AUC：

| ID | 编号 | matched AUC | matched EER |
|---|---|---:|---:|
| `rd_s12_t4_h512` | V0 | 85.2491 | 0.2158 |
| `rd_s4_t12_h512` | V1 | 82.6516 | 0.2464 |
| `rd_s4_t12_gl768_h512` | V2 | 81.7358 | 0.2586 |
| `rd_s4_t12_gl192_h512` | V3 | 76.2947 | 0.3166 |
| `rd_s4_t12_gl192_h128` | V4 | 75.7384 | 0.3208 |
| `rd_s4_t12_gl192_h128_qr2` | V5 | 75.5807 | 0.3234 |

对比归档在 `comparison/uit/ours/<variant>/`；旧 `20261101` 矩阵保留为历史运行。

所有变体的 `code_dim=64`；`h512/h128` 指 OCBE 的 `fuse_hidden`，**不是** `code_dim`。

## 六个变体

| ID | 编号 | 模型 | Student | Teacher taps | GL 维度 | OCBE h | Q | 参数量 |
|---|---|---|---|---|---|---|---:|---:|
| `rd_s12_t4_h512` | V0 | 原始 12block | 12 × 标准 Block | `[5,7,9,11]`（4 taps） | — | 512 | — | 89,079,104 |
| `rd_s4_t12_h512` | V1 | 4block | 4 × 标准 Block | `0..11`（12 taps） | — | 512 | — | 40,258,880 |
| `rd_s4_t12_gl768_h512` | V2 | 4block + Global-Local768 | 1 Global + 3 Local | `0..11` | 768 | 512 | — | 30,248,000 |
| `rd_s4_t12_gl192_h512` | V3 | 4block + Global-Local192 | 1 Global + 3 Local | `0..11` | 192 | 512 | — | 7,718,336 |
| `rd_s4_t12_gl192_h128` | V4 | GL192 + Dense128 | 1 Global + 3 Local | `0..11` | 192 | 128 | — | 4,154,432 |
| `rd_s4_t12_gl192_h128_qr2` | V5 | GL192 + Dense128 + Learnable Rank2 Q | 1 Global + 3 Local | `0..11` | 192 | 128 | 2×3 可学习 Q，随机正交初始化 | 3,761,222 |

结构细节：

- **V0**：`StudentDecoder(n_taps=4, depth=12)`；4 个重建头落在第 3/6/9/12 个 block 后（`recon_at=[2,5,8,11]`），反向对齐 4 个教师 tap。
- **V1–V5**：12 个教师层按相邻 3 层分 4 组；第 1 个 student stage 重建最深组 9–11，最后一个 stage 重建 0–2；每个 stage 3 个重建头。
- **V5 的 Q**：`LearnableDepthRank2OCBE`，每个 3 层组通过同一个 2×3 矩阵投影成 2 种深度模式；`basis_init='random_orthonormal'`，`basis_seed=seed`，前向 Gram-Schmidt 保持正交。

## 命名规则

| 字段 | 含义 |
|---|---|
| `rd` | Reverse Distillation，反向蒸馏 |
| `s12` / `s4` | 学生 Transformer Block 数量 |
| `t4` / `t12` | 教师特征提取层（taps）数量 |
| `gl768` / `gl192` | Global-Local 内部特征维度 |
| `h512` / `h128` | OCBE 隐藏层宽度 `fuse_hidden` |
| `qr2` | Rank-2 深度投影 Q（可学习，随机正交初始化） |

补充：

- `code_dim=64` 是六个变体固定的输出瓶颈，不写进 ID。
- `t4` 在 V0 中特指 `[5,7,9,11]`；`t12` 指 `0..11`。
- `h*` 是 OCBE `fuse_hidden`，不是 `code_dim`。

## 代码入口

- 统一变体构造：[scripts/uit_appearance_runner.py:98-121](../scripts/uit_appearance_runner.py#L98-L121) 的 `build_variant()`。
- V0 student：[model/student.py:40](../model/student.py#L40) `StudentDecoder`。
- V1 student：[scripts/uit_appearance_runner.py:31](../scripts/uit_appearance_runner.py#L31) `PlainFourBlockDecoder`（历史实现，位于 runner 内，不在 `model/`）。
- V2–V5 student：[model/student_structural.py:76](../model/student_structural.py#L76) `FourBlockGlobalLocalDecoder`。
- OCBE：[model/bottleneck.py:27](../model/bottleneck.py#L27) `OCBE`。
- V5 OCBE：[model/bottleneck_depthrank_learnable.py:15](../model/bottleneck_depthrank_learnable.py#L15) `LearnableDepthRank2OCBE`。
- 损失：[scripts/uit_appearance_runner.py:81-96](../scripts/uit_appearance_runner.py#L81-L96)。V0 的 4 taps 用 tap-mean top5 cosine；V1–V5 的 12 taps 用 group-balanced top5 cosine（alpha=0.5）。

## 单因素对照链

- **V0 → V1**：12block/4 taps → 4block/12 taps。同时改变了学生深度、教师 taps 和损失（tap-mean → group-balanced），**不是严格单因素对照**。
- **V1 → V2**：4 个标准 Block → 1 Global + 3 Local（宽度 768）。只改 block topology。
- **V2 → V3**：GL 宽度 768 → 192。只改宽度。
- **V3 → V4**：OCBE `fuse_hidden` 512 → 128。只改瓶颈宽度。
- **V4 → V5**：加入可学习 rank2 Q（随机正交初始化）。只改深度投影。

## 正式运行（UIT 45ep 六变体矩阵）

| ID | `build_variant` 分支 | 实验目录 | 状态 | 45ep UIT AUC |
|---|---|---|---|---:|
| V0 | `uit_12block_dense512` | [20261101_uit_45ep_12block_dense512_retry2_s42](uit/appearance/20261101_uit_45ep_12block_dense512_retry2_s42/README.md) | `completed` | 85.25 |
| V1 | `uit_4block_plain` | [20261101_uit_45ep_4block_dense512_retry2_s42](uit/appearance/20261101_uit_45ep_4block_dense512_retry2_s42/README.md) | `completed` | 82.23 |
| V2 | `uit_4block_gl768` | [20261101_uit_45ep_4block_gl768_dense512_retry2_s42](uit/appearance/20261101_uit_45ep_4block_gl768_dense512_retry2_s42/README.md) | `completed` | 81.38 |
| V3 | `uit_4block_gl192` | [20261101_uit_45ep_4block_gl192_dense512_retry2_s42](uit/appearance/20261101_uit_45ep_4block_gl192_dense512_retry2_s42/README.md) | `completed` | 75.99 |
| V4 | `uit_gl192_dense128` | [20261101_uit_45ep_gl192_dense128_retry2_s42](uit/appearance/20261101_uit_45ep_gl192_dense128_retry2_s42/README.md) | `completed` | 75.51 |
| V5 | `uit_gl192_dense128_rank2` | [20261101_uit_45ep_gl192_dense128_rank2_retry2_s42](uit/appearance/20261101_uit_45ep_gl192_dense128_rank2_retry2_s42/README.md) | `completed` | 75.40 |

AUC 是单 seed42、45 epoch 的 UIT 结果，只用于识别本次矩阵，不构成多种子结论；V0 与 V1–V5 的 taps/loss 不同，不能把这张表当作严格单因素排名。

## 容易混淆的模型（不要映射到这些 ID）

1. **DA Student12 GL768 ≠ V2**。`20261102_student12_gl768_*` / `20261103_student12_gl768_retry_failed3_s42` 是 **12-block 3×[1 Global + 3 Local]**，4 个教师 taps `[5,7,9,11]`，4 个重建头；V2 是 **4-block 1 Global + 3 Local**，12 个教师 taps。
2. **DA `20261013_seed43_glll192_ocbe128_rank2_s43` ≠ V5**。它用固定 mean/trend 的 `DepthRank2OCBE`，不是可学习 Q。
3. **DA `20261013_ocbe_depthrank2_learnableorth_h128_s42` ≠ V5**。它虽然可学习，但 Q 初始化为 mean/trend，不是 `qr2` 要求的随机正交初始化。
4. **DA `20261013_ocbe_depthrank2_hidden128_s42` ≠ V5**。它是固定 mean/trend rank2。
5. **DA `20261102_student12_plain_*` 的结构是 V0**，但它是 DA 30ep 运行，不是 UIT 45ep 的 V0 正式运行；其 `gl12` 对照也不是 V2。该系列 config 里 `teacher taps 0..11` 是旧描述，实际 `run_scene.py` 用 `[5,7,9,11]`，以脚本为准。

## 同架构的其它运行（非穷举）

同一 `variant_id` 可以在不同数据集/种子上复用；下表只列已核对过脚本或配置的例子。V1 在 DA 侧没有独立实验目录；`20261012_structural_mixer_glll_s42` 的 no-adapter baseline 使用 40,258,880 参数，与 V1 参数一致，结果嵌在该实验的 summary 内，引用时需注明来源。

| ID | 运行 | 数据集 / seed / epochs | 说明 |
|---|---|---|---|
| V0 | `20260611_rd_baseline` | DA / original / 15 | 原始 baseline |
| V0 | `20261102_student12_plain_retry4_s42` | DA / 42 / 30 | plain12，30ep |
| V2 | `20261012_structural_mixer_glll_s42` | DA / 42 / 15 | GL768 |
| V2 | `20261013_seed43_glll768_ocbe512_s43` | DA / 43 / 15 | GL768 |
| V3 | `20261012_student_width192_glll_s42` | DA / 42 / 15 | GL192 |
| V3 | `20261013_seed43_glll192_ocbe512_s43` | DA / 43 / 15 | GL192 |
| V4 | `20261013_ocbe_hidden128_dense_s42` | DA / 42 / 15 | GL192 + h128 |
| V4 | `20261013_seed43_glll192_ocbe128_s43` | DA / 43 / 15 | GL192 + h128 |
| V5 | `20261013_ocbe_depthrank2_randomorth_h128_s42` | DA / 42 / 15 | GL192 + h128 + learnable random-orth Q |
