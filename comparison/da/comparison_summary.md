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

> 本表全部来自各方法/变体目录的 `benchmark_v2.json`：`FlopCounterMode + native-MHA 公式 + 手写模块公式 + profiler 共同子集`，50 warmup + 3×500 iterations。旧 v1 数字在 §5 归档，不再引用。
> `FLOPs verified=yes` 表示所有 MAC 算子均已核验、`flops_ops_missing=[]`；profiler 只比共同支持算子，不用其总数做验收。

| Method | Input (H×W×T) | Inference Params (M) ↓ | Training Trainable (M) ↓ | Training-only (M) ↓ | FLOPs (G/output score) ↓ | FLOPs verified | Latency@BS1 P50 (ms/output score) ↓ | FPS@BS1 ↑ | FPS@BS16 ↑ | AUC mean ↑ | Status |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| adVAE | unknown | — | — | — | — | — | — | — | — | 63.87 | blocked |
| GANomaly | 256×256×1 | 143.950 | 188.690 | 44.739 | 16.428 | yes | 8.088 | 123.6 | 848.5 | 64.54 | measured |
| MemAE | 256×256×16 | 6.503 | 6.503 | 0 | 70.808 | yes | 7.321 | 136.5 | 130.3 | 67.67 | measured |
| MKD | 128×128×1 | 15.059 | 0.344 | 0 | 10.477 | yes | 2.495 | 394.5 | 2340.8 | 66.63 | measured |
| ANDT | 256×256×6 | 70.347 | 70.347 | 0 | — | no | — | — | — | 68.64 | blocked |
| Future Frame | TBD | — | — | — | — | — | — | — | — | — | blocked |
| ST Dissoc. | TBD | — | — | — | — | — | — | — | — | — | blocked |
| MLEP | TBD | — | — | — | — | — | — | — | — | — | blocked |
| ASTT | TBD | — | — | — | — | — | — | — | — | — | blocked |
| DAD-FSM | TBD | — | — | — | — | — | — | — | — | — | blocked |
| HSTforU | TBD | — | — | — | — | — | — | — | — | 66.96 | blocked |
| MADViT Full | TBD | — | — | — | — | — | — | — | — | 72.2 | blocked |
| MADViT Multi-layer | TBD | — | — | — | — | — | — | — | — | 62.9 | blocked |
| MADViT Last-layer | TBD | — | — | — | — | — | — | — | — | 74.19 | blocked |
| LightViT-AD | TBD | — | — | — | — | — | — | — | — | *83.12 | blocked |
| DwCVAE | — | — | — | — | — | — | — | — | — | *74.95 | paper-only |
| CVAE | — | — | — | — | — | — | — | — | — | *68.62 | paper-only |
| DCVAE | — | — | — | — | — | — | — | — | — | *64.11 | paper-only |
| SCVAE | — | — | — | — | — | — | — | — | — | *62.25 | paper-only |
| CLSTMVAE | — | — | — | — | — | — | — | — | — | *63.98 | paper-only |
| Ours V0 `rd_s12_t4_h512` | 256×256×1 | 174.720 | 89.079 | 0 | 95.107 | yes | 14.521 | 68.4 | 187.4 | 84.7916 | measured |
| Ours V1 `rd_s4_t12_h512` | 256×256×1 | 125.900 | 40.259 | 0 | 68.532 | yes | 19.903 | 50.0 | 244.3 | 84.9435 | measured |
| Ours V2 `rd_s4_t12_gl768_h512` | 256×256×1 | 115.889 | 30.248 | 0 | 62.804 | yes | 14.787 | 67.3 | 263.3 | 85.3707 | measured |
| Ours V3 `rd_s4_t12_gl192_h512` | 256×256×1 | 93.360 | 7.718 | 0 | 51.132 | yes | 19.289 | 51.5 | 314.8 | 84.0864 | measured |
| Ours V4 `rd_s4_t12_gl192_h128` | 256×256×1 | 89.796 | 4.154 | 0 | 49.308 | yes | 19.417 | 51.3 | 324.7 | 83.9186 | measured |
| Ours V5 `rd_s4_t12_gl192_h128_qr2` | 256×256×1 | 89.402 | 3.761 | 0 | 49.106 | yes | 20.513 | 48.5 | 324.4 | 84.0137 | measured |

> 复现与验证证据见 [v2_validation.md](v2_validation.md)。所有方法的结果都在各自目录的 `benchmark_v2.json`，未覆盖 v1 文件。

### 3.1 时序附表（T / K / stride）

| Method | T | K | stride | window FLOPs (G) | window Latency (ms) | score density (K/stride) | frame coverage | FPS independent-window ↑ | fps_stream_estimated (frames/s) ↑ | fps_stream_measured | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MemAE | 16 | 1 | 1 | 70.808 | 7.321 | 1.0 | 1.0（≈；边界帧除外） | 136.5 | 136.5 | null | measured |
| ANDT（论文 6 帧） | 6 | 1 | 1 | — | — | 1.0 | ≈1.0（除开头 5 帧） | — | — | null | blocked |
| Future Frame | TBD | TBD | TBD | — | — | — | — | — | — | null | blocked |
| ST Dissoc. | TBD | TBD | TBD | — | — | — | — | — | — | null | blocked |
| MLEP | TBD | TBD | TBD | — | — | — | — | — | — | null | blocked |
| ASTT | TBD | TBD | TBD | — | — | — | — | — | — | null | blocked |
| DAD-FSM | TBD | TBD | TBD | — | — | — | — | — | — | null | blocked |
| HSTforU | TBD | TBD | TBD | — | — | — | — | — | — | null | blocked |

> `K` 不能默认等于 `T`；`score_density=K/stride` 可以 >1，**不是帧覆盖率**。`frame_coverage` 必须按原方法的分数映射、窗口重叠和融合方式单独计算（例如 T=16/stride=1/K=16 时 score_density=16，但 frame_coverage 最多 1.0）。`fps_stream_estimated` 只是按 stride 推算的吞吐估计；没有实际连续流测试时 `fps_stream_measured=null`，不能声称已实测。
> **MemAE 已核实**：`script_testing.py` 每个 16 帧窗口只输出一个 clip 级误差（对 T,H,W 取均值），并映射到中心帧；因此 `K=1, stride=1`，不是 `K=T=16`。旧 v1 的 `÷16` 把窗口时延/FLOPs 错误摊成 1/16，已废弃。

### 3.2 每行配置与来源

| Method | code commit | Input (H×W×T) | weights state | DA-adapted | AUC source | profiling config |
|---|---|---|---|---|---|---|
| adVAE | `440d8af22b88b6b7b24e02f2f0599c9ef542b34c` | unknown（表格版 `(B,400)`） | — | no | ANDT Table VI / 用户表 | — |
| GANomaly | `78da4ea9a99f5b02ab60dd651a18def929176d77` | 256×256×1 | random init（未载入训练权重） | no | ANDT Table VI / 用户表 | v2: 5880 Ada, FP32, BS1 50/500/3, BS16 50/500/3 |
| MemAE | `ceece7714fb241e82ef3f3785d3d1ed86c28113e` | 256×256×16 | random init | no | ANDT Table VI / 用户表 | v2: 同上 |
| MKD | `0c574ae9ad12666982a1ccc2c7148acd6bf442fe` | 128×128×1（MVTec 配置） | random init（VGG16 source 未载入预训练） | no | ANDT Table VI / 用户表 | v2: 同上 |
| ANDT | `f2e8c4f701238a99faa5841b3e2884cb1c507d1e` | 256×256×6（论文 6 帧） | — | no | ANDT 论文 / 用户表 | — |
| Future Frame | `9e0f161b8d5f1f46556e116577a5596405798343` | TBD | — | no | 用户表 | — |
| ST Dissoc. | `1206c2a7c4c4e0b47c21c498fadb8e1e3973dd56` | TBD | — | no | 用户表 | — |
| MLEP | `22e96634fe43a5b8394413a1aae267b673b3c4ce` | TBD | — | no | 用户表 | — |
| ASTT | `1d05305a94fb47b33897c7aa085c5351bcbbc8f9` | TBD | — | no | 用户表 | — |
| DAD-FSM | `602fed9ac93e6a3f2a291e9b1f23bcca775df961` | TBD | — | no | 用户表 | — |
| HSTforU | `9548473306bbb89895b077275e8503de79653029` | TBD | — | no | 用户表 | — |
| MADViT（三个变体） | `a1d1732c78f9f0a21a1a162b87b158436dad79f3` | TBD | — | no | 用户表 | — |
| LightViT-AD | `c0f6ca9a6817d126e15e09f83612710ee6f954d1` | TBD | — | no | 用户表（缺 vehicle/crossroads） | — |
| DwCVAE / CVAE / DCVAE / SCVAE / CLSTMVAE | — | — | — | — | 用户表（不同论文/硬件/batch） | paper-only，不可与本机混排 |
| Ours V0–V5 | `cef6f9690c0bcd3b5f05d4ffab6ff93640a134b7`（模型代码基线；v2 benchmark 脚本基于该提交运行，脚本见 `comparison/benchmarks/`） | 256×256×1 | architecture-init（效率测结构；AUC 用 DA-trained 权重） | yes（DA 输入/配置） | `experiments/da/appearance/20261103_variant_matrix_*`（seed42, 30ep） | v2: 5880 Ada, FP32, BS1 50/500/3, BS16 50/500/3 |

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
| MKD | 128×128×1 | 15.059 | 0.344 | 10.477 | 2.273 | 442.9 | 3554.4 | measured |
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
4. **ANDT 6 帧 decoder 仍 blocked**：发布代码只支持 1 帧；论文 6 帧模型 `T=6,K=1,stride=1`，需修/确认 decoder 后才能测 FLOPs/latency。
5. **adVAE 仍 blocked**：官方仓库是表格版，视频实现缺失。
6. **DwCVAE/CVAE/DCVAE/SCVAE/CLSTMVAE**：用户表速度的硬件/batch 不同，`paper-only`，不能与本机实测混排。
7. **其余外部方法**（Future Frame/ST Dissoc./MLEP/ASTT/DAD-FSM/HSTforU/MADViT/LightViT-AD）：代码已 clone 但未按 v2 适配，保持 `blocked`，不填估计值。

## 7. 证据与下一步

- 计数器小规模验证（ConvTranspose/Conv3d/Attention）与各方法 FLOPs 明细见 [v2_validation.md](v2_validation.md)。
- 逐方法待办与验收条件见 [RETEST_CHECKLIST.md](../RETEST_CHECKLIST.md)；Ours V0–V5、GANomaly、MemAE、MKD 已完成，ANDT/adVAE/其余外部方法仍 blocked。
- 改动原因与哪些旧数字不再正式见 [protocol_change_log.md](../protocol_change_log.md)。
- 旧 JSON/日志/审计文件哈希见 [LEGACY_v1.md](LEGACY_v1.md)。
- v2 结果文件：各方法/变体目录下的 `benchmark_v2.json`；Ours 聚合在 `comparison/da/ours/benchmark_v2_ours.json`（未覆盖任何 v1 文件）。
