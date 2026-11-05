# DA 六变体矩阵（seed42, 30 epoch）

同一 `scripts/da_variant_scheduler.py` 双卡动态队列；每个场景独立模型，七场景 macro 平均。

| ID | 编号 | status | macro AUC | macro EER | params | scenes |
|---|---|---|---:|---:|---:|---:|
| `rd_s12_t4_h512` | V0 | `completed` | 84.7916 | 0.2028 | 89,079,104 | 7/7 |
| `rd_s4_t12_h512` | V1 | `completed` | 85.1314 | 0.2031 | 40,258,880 | 7/7 |
| `rd_s4_t12_gl768_h512` | V2 | `completed` | 85.2147 | 0.1965 | 30,248,000 | 7/7 |
| `rd_s4_t12_gl192_h512` | V3 | `completed` | 84.2137 | 0.2001 | 7,718,336 | 7/7 |
| `rd_s4_t12_gl192_h128` | V4 | `completed` | 83.9925 | 0.2029 | 4,154,432 | 7/7 |
| `rd_s4_t12_gl192_h128_qr2` | V5 | `completed` | 84.6233 | 0.1970 | 3,761,222 | 7/7 |

## 分场景 AUC

| ID | bike | crossroads | farmland | highway | railway | solar | vehicle |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0 | 98.05 | 58.81 | 99.42 | 93.16 | 96.71 | 87.14 | 60.25 |
| V1 | 97.74 | 60.18 | 99.49 | 94.82 | 95.61 | 89.42 | 58.67 |
| V2 | 97.70 | 61.55 | 99.46 | 94.94 | 95.48 | 89.72 | 57.65 |
| V3 | 97.80 | 58.40 | 99.57 | 94.63 | 92.01 | 86.55 | 60.53 |
| V4 | 97.76 | 59.78 | 99.49 | 93.58 | 91.20 | 86.58 | 59.56 |
| V5 | 97.82 | 60.43 | 99.57 | 94.42 | 92.20 | 87.55 | 60.38 |

## 比较注意

- 矩阵内部同 runner、seed、30 epoch、数据划分；V0 与 V1–V5 的 taps/loss 不同，V0→V1 不是严格单因素对照。
- 单 seed42；分场景波动可大于 macro 差异，不能据此声称显著性。
- 与旧 DA 同名运行的绝对数值差异可能来自 cuDNN deterministic、动态双卡分配和运行方差；旧运行模型代码与当前一致（git diff 05443fa..HEAD 对 model/ 为空），不是加载错模型。
- 新旧 runner 的 `split_fingerprint` 配方不同：旧 DA runner 用绝对路径拼 manifest，新 runner 用场景相对路径；数据文件相同但哈希不同（Crossroads 旧 `8ded80ee…`，新 `22f9aa85…`），不要直接用 fingerprint 跨代判断数据划分。新矩阵六变体之间 fingerprint 一致。
- 数据来源与泄漏审计见 [20261103_variant_matrix_data_audit.md](20261103_variant_matrix_data_audit.md)。
