# UIT-ADrone 对比汇总

UIT-ADrone 使用官方 51 个 test 视频拼接后的**单次全局 AUC/EER**；这与 DA 的七场景 macro 不同，不能把 DA 的 macro 数值直接搬过来。

- 外部方法的 AUC/EER 来自用户提供的对比表/原论文；不是本机复现。
- Ours 六行来自本机 `experiments/uit/appearance/20261104_variant_matrix_v*_s42/`，seed42、45 epoch、matched group-balanced top-5 评分；对应归档在 `comparison/uit/ours/<variant>/`。
- 截图/旧表中的 Ours 六行是 `20261101` 旧 top-k 结果：85.249 / 82.234 / 81.384 / 75.990 / 75.510 / 75.404；本表用 `20261104` matched 结果替换，旧实验仍保留。

## 主表

| Method | Venue | AUC | EER | Source / 状态 |
|---|---|---:|---:|---|
| Future Frame Prediction, 2018 | CVPR | 53.560 | 0.470 | 用户表 |
| MLEP, 2019 | IJCAI | 53.550 | 0.470 | 用户表 |
| MNAD, 2020 | CVPR | 55.880 | 0.460 | 用户表 |
| AMMCNet, 2021 | AAAI | 60.700 | 0.441 | 用户表 |
| RTFM, 2021 | ICCV | 62.880 | — | 用户表 |
| ST Dissoc., 2022 | PR | 57.050 | 0.450 | 用户表 |
| F2DN, 2022 | IEEE TPAMI | 59.600 | 0.448 | 用户表 |
| ANDT, 2022 | IEEE TGRS | 60.500 | 0.420 | 用户表 |
| MSL, 2022 | AAAI | 63.660 | — | 用户表 |
| MAAMNet, 2023 | PR | 60.100 | 0.462 | 用户表 |
| ASTNet, 2023 | APIN | 61.400 | 0.415 | 用户表 |
| TEVAD, 2023 | CVPRW | 63.460 | — | 用户表 |
| MGFN, 2023 | AAAI | 64.250 | — | 用户表 |
| ASTT, 2024 | IEEE TCSVT | 65.450 | 0.390 | 用户表 |
| VAD-Cluster, 2024 | PR | 58.400 | 0.494 | 用户表 |
| DAD-FSM, 2025 | IEEE Access | 68.130 | 0.340 | 用户表 |
| VAD-Mamba, 2025 | IEEE ICME | 68.600 | 0.403 | 用户表 |
| LGN-Net, 2025 | ESWA | 62.700 | 0.428 | 用户表 |
| MA-PDM, 2025 | AAAI | 65.700 | 0.384 | 用户表 |
| HSTforU, 2025 | APIN | 66.200 | 0.386 | 用户表 |
| MMVAD (ResNet50), 2025 | ESWA | 67.400 | — | 用户表 |
| MMVAD (CSP-DarkNet53), 2025 | ESWA | 69.560 | — | 用户表 |
| FTDMamba, 2026 | preprint | 70.700 | 0.368 | 用户表 |
| MADViT (Full model) | ESWA | 77.810 | 0.290 | 用户表 |
| MADViT (Multi-layer distillation) | ESWA | 75.590 | 0.313 | 用户表 |
| MADViT (Last-layer tokens) | ESWA | 83.650 | 0.240 | 用户表 |
| LightViT-AD, 2026 | IJMLC | 71.820 | 0.345 | 用户表 |
| DwCVAE, 2026 | ESWA | 79.770 | 0.270 | 用户表 |
| CVAE (Saleem et al.), 2026 | ESWA | 79.720 | 0.260 | 用户表 |
| DCVAE (Saleem et al.), 2026 | ESWA | 79.100 | 0.270 | 用户表 |
| SCVAE (Saleem et al.), 2026 | ESWA | 79.340 | 0.270 | 用户表 |
| CLSTMVAE (Saleem et al.), 2026 | ESWA | 79.060 | 0.270 | 用户表 |
| **12block Ours (RD) / V0** | — | **85.2491** | **0.2158** | 本机 20261104 matched |
| **4block / V1** | — | **82.6516** | **0.2464** | 本机 20261104 matched |
| **Global-Local768 / V2** | — | **81.7358** | **0.2586** | 本机 20261104 matched |
| **Global-Local192 / V3** | — | **76.2947** | **0.3166** | 本机 20261104 matched |
| **GL192-Dense128 / V4** | — | **75.7384** | **0.3208** | 本机 20261104 matched |
| **GL192-D128-Rank2 / V5** | — | **75.5807** | **0.3234** | 本机 20261104 matched |

## 说明

- `—` 表示用户表未给 EER，不是 0。
- 外部方法只登记表中 AUC/EER，未在本机重新训练/评测；不能把本机 `comparison/da/` 的 DA 数字当作 UIT 数字。
- Ours 六行的旧 top-k 数值与 matched 数值不同；以本表 matched 为准，旧值仅保留在 `experiments/uit/appearance/20261101_*` 和 `20261104` 的 `uit_legacy_topk.json` / `results_legacy_topk/` 中。
- Ours 六行的效率指标见 `comparison/uit/ours/<variant>/benchmark_v2.json` 和 `record.json` 的 `efficiency`；AUC/EER 来自本机 UIT 实验，不参与效率排名。
