# DA 方法对比总表

公平口径：[comparison/BENCHMARK_PROTOCOL.md](../BENCHMARK_PROTOCOL.md)

**AUC 来源必须分开读：**

- 外部方法（adVAE / GANomaly / MemAE / MKD / ANDT / Future Frame / ST Dissoc. / MLEP / ASTT / DAD-FSM / HSTforU / MADViT / LightViT-AD / 以及 CVAE 系列）的 AUC 取自用户提供的对比表 / ANDT 论文；ANDT 论文明确 baseline 是在其统一网络与评测协议下跑的，**不一定等于各方法原论文配置**。
- Ours 的 AUC 取自 `experiments/da/appearance/20261103_variant_matrix_*`（seed42，30 epoch，七场景 macro）。
- 轻量化指标是新协议实测/审计结果，不能和旧表里“只算学生、不含教师”的 Latency/FPS 混用。旧表 Ours V0 是 3.319 ms/301.3 FPS；新协议含 DINOv3 教师，是 17.800 ms/58.7 FPS。两组数字不能混用。
- GANomaly / MemAE / MKD 的轻量化数字是**各自原始代码/配置**实测；它们的 AUC 数字来自 ANDT benchmark，二者不是同一适配版本，不能把“原论文模型参数”当成“ANDT 适配版参数”。

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

## 2. Ours 20261103 30ep 矩阵（新 AUC，与上表旧 Ours 行区分）

| ID | V | highway | farmland | bike | solar | vehicle | crossroads | railway | macro AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| rd_s12_t4_h512 | V0 | 93.16 | 99.42 | 98.05 | 87.14 | 60.25 | 58.81 | 96.71 | 84.7916 |
| rd_s4_t12_h512 | V1 | 95.72 | 99.25 | 97.68 | 89.42 | 57.24 | 60.85 | 94.44 | 84.9435 |
| rd_s4_t12_gl768_h512 | V2 | 95.86 | 99.22 | 97.65 | 90.09 | 57.82 | 63.05 | 93.91 | 85.3707 |
| rd_s4_t12_gl192_h512 | V3 | 94.86 | 99.31 | 97.66 | 86.42 | 58.59 | 60.88 | 90.89 | 84.0864 |
| rd_s4_t12_gl192_h128 | V4 | 94.48 | 99.13 | 97.61 | 86.18 | 58.30 | 62.20 | 89.53 | 83.9186 |
| rd_s4_t12_gl192_h128_qr2 | V5 | 94.89 | 99.32 | 97.68 | 87.18 | 56.25 | 62.25 | 90.53 | 84.0137 |

## 3. 轻量化主表（按新协议；实测/审计）

| Method | Input (H×W×T) | Total Params (M) | Trainable Params (M) | FLOPs (G/output frame) | Latency@BS1 P50 (ms/output frame) | FPS@BS1 | FPS@BS16 | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| adVAE | unknown | blocked | blocked | blocked | blocked | — | — | blocked |
| GANomaly | 256×256×1 | 143.950 (G) | 188.690 (G+D train) | 10.952 | 13.357 | 74.9 | 622.7 | measured |
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

## 4. 参数审计细节

- **Teacher**：DINOv3 ViT-B/16，frozen，85.641M；Ours 的 Total = teacher + OCBE + student；Trainable = OCBE + student。
- **GANomaly**：`NetG`（encoder1+decoder+encoder2）143.950M 是推理图；`NetD` 44.739M 只在训练使用；Trainable 列写 G+D 训练参数 188.690M，避免把 D 悄悄丢掉。
- **MemAE**：`chnum_in=3`、`mem_dim=2000`、`T=16`；参数与 T 无关，6.503M。
- **MKD**：source VGG16 features 14.715M（frozen）+ cloner cfg B 0.344M（trainable）= 15.059M；输入 128×128 按原论文 MVTec 配置；若 DA 适配到 256，需标 `adapted`。首次测量在 `inference_mode` 下被 `VGG.forward` 的 `register_hook` 拒绝；改用无 hook 的 source/cloner 前向重测，表中数字来自重测。计时/FLOPs 覆盖 source+cloner 全前向；打分用最终特征距离作示例，距离计算本身可忽略。
- **ANDT**：代码 `b16` + `num_layers=2`、`heads=6`、`num_frames=6`；`mlp=4096` 时 70.347M，`mlp=1536` 时 62.478M。论文 ablation 选 4096；但 forward 有 `pdb.set_trace()` 且 decoder 对 6 帧形状不匹配，FLOPs/latency 必须先确认代码版本。
- **adVAE**：官方仓库 `WangXuhongCN/adVAE` 是表格数据（ODDS）实现，不是 DA 视频模型；其 `Encoder(400,64)+Decoder(400,64)+Gauss_trans(64)` 合计 **0.329M**，但输入是 `(B,400)` 表格向量，不是 256×256 图像。这个 0.329M 不能填进 DA 主表；需要 ANDT 作者实际 adVAE 视频实现或论文图像配置，否则只能保留 `blocked`。详见 [adVAE/default/README.md](adVAE/default/README.md)。

## 5. 待办

1. ✅ GANomaly / MemAE / MKD 的 BS1/BS16/FLOPs 已完成（脚本 `comparison/benchmarks/bench_external.py`，结果 `benchmark_{ganomaly,memae,mkd}.json`）。
2. 确认 ANDT 的 6 帧论文模型/decoder 版本，再补 FLOPs/latency；当前参数按代码结构数，时延/FLOPS 保留 `blocked`。
3. 找到 ANDT 实际使用的 adVAE 视频实现，补参数/时延；否则主表保留 `blocked`。
4. 其余方法（ASTT/DAD-FSM/HSTforU/MADViT/LightViT-AD 等）按同一协议逐个补测；其中 MADViT/LightViT-AD/ASTT/DAD-FSM/HSTforU 的代码已 clone，但尚未做协议适配。
