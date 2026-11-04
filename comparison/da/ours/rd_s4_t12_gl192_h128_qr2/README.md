# da-ours-rd_s4_t12_gl192_h128_qr2

状态：`completed`　变体：`rd_s4_t12_gl192_h128_qr2`（V5）

- 原始实验：[20261103_variant_matrix_v5_rd_s4_t12_gl192_h128_qr2_s42](../../../../experiments/da/appearance/20261103_variant_matrix_v5_rd_s4_t12_gl192_h128_qr2_s42)（run `20261103T144204Z_da_variant_matrix_s42_20261103_variant_matrix_v5_rd_s4_t12_gl192_h128_qr2_s42`）
- 统一协议：one-class，官方 train 正常帧；每场景 test concat 后一次 global AUC/EER；seed42，30 epoch，256×256。
- 结构说明：GL192; OCBE fuse_hidden=128; 12 teacher taps; learnable rank-2 Q (random orthonormal init); group-balanced top5 loss (alpha=0.5).
- 矩阵总表：[20261103_variant_matrix_summary.md](../../../../experiments/da/appearance/20261103_variant_matrix_summary.md)
- 数据/泄漏审计：[20261103_variant_matrix_data_audit.md](../../../../experiments/da/appearance/20261103_variant_matrix_data_audit.md)

## 结果

macro AUC：**84.0137**　macro EER：**0.2021**

| scene | AUC | EER |
|---|---:|---:|
| bike | 97.6828 | 0.0454 |
| crossroads | 62.2490 | 0.4178 |
| farmland | 99.3211 | 0.0629 |
| highway | 94.8856 | 0.1322 |
| railway | 90.5261 | 0.1314 |
| solar | 87.1792 | 0.1972 |
| vehicle | 56.2522 | 0.4277 |

完整逐帧分数与 checkpoint 以原始实验目录为准；本条目只复制 `results/summary.{json,md}`。
