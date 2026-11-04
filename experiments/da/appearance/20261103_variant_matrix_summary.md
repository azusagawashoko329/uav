# DA 六变体矩阵（seed42, 30 epoch）

同一 `scripts/da_variant_scheduler.py` 双卡动态队列；每个场景独立模型，七场景 macro 平均。

| ID | 编号 | status | macro AUC | macro EER | params | scenes |
|---|---|---|---:|---:|---:|---:|
| `rd_s12_t4_h512` | V0 | `completed` | 84.7916 | 0.2029 | 89,079,104 | 7/7 |
| `rd_s4_t12_h512` | V1 | `completed` | 84.9435 | 0.2011 | 40,258,880 | 7/7 |
| `rd_s4_t12_gl768_h512` | V2 | `completed` | 85.3707 | 0.1932 | 30,248,000 | 7/7 |
| `rd_s4_t12_gl192_h512` | V3 | `completed` | 84.0864 | 0.1992 | 7,718,336 | 7/7 |
| `rd_s4_t12_gl192_h128` | V4 | `completed` | 83.9186 | 0.2024 | 4,154,432 | 7/7 |
| `rd_s4_t12_gl192_h128_qr2` | V5 | `completed` | 84.0137 | 0.2021 | 3,761,222 | 7/7 |

## 分场景 AUC

| ID | bike | crossroads | farmland | highway | railway | solar | vehicle |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0 | 98.05 | 58.81 | 99.42 | 93.16 | 96.71 | 87.14 | 60.25 |
| V1 | 97.68 | 60.85 | 99.25 | 95.72 | 94.44 | 89.42 | 57.24 |
| V2 | 97.65 | 63.05 | 99.22 | 95.86 | 93.91 | 90.09 | 57.82 |
| V3 | 97.66 | 60.88 | 99.31 | 94.86 | 90.89 | 86.42 | 58.59 |
| V4 | 97.61 | 62.20 | 99.13 | 94.48 | 89.53 | 86.18 | 58.30 |
| V5 | 97.68 | 62.25 | 99.32 | 94.89 | 90.53 | 87.18 | 56.25 |

## 比较注意

- 矩阵内部同 runner、seed、30 epoch、数据划分；V0 与 V1–V5 的 taps/loss 不同，V0→V1 不是严格单因素对照。
- 单 seed42；分场景波动可大于 macro 差异，不能据此声称显著性。
- 与旧 DA 同名运行的绝对数值差异可能来自 cuDNN deterministic、动态双卡分配和运行方差；旧运行模型代码与当前一致（git diff 05443fa..HEAD 对 model/ 为空），不是加载错模型。
- 新旧 runner 的 `split_fingerprint` 配方不同：旧 DA runner 用绝对路径拼 manifest，新 runner 用场景相对路径；数据文件相同但哈希不同（Crossroads 旧 `8ded80ee…`，新 `22f9aa85…`），不要直接用 fingerprint 跨代判断数据划分。新矩阵六变体之间 fingerprint 一致。
- 数据来源与泄漏审计见 [20261103_variant_matrix_data_audit.md](20261103_variant_matrix_data_audit.md)。
