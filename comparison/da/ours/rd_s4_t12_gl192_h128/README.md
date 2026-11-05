# da-ours-rd_s4_t12_gl192_h128

状态：`completed`　变体：`rd_s4_t12_gl192_h128`（V4）

- 原始实验：[20261103_variant_matrix_v4_rd_s4_t12_gl192_h128_s42](../../../../experiments/da/appearance/20261103_variant_matrix_v4_rd_s4_t12_gl192_h128_s42)（run `20261103T144204Z_da_variant_matrix_s42_20261103_variant_matrix_v4_rd_s4_t12_gl192_h128_s42`）
- 统一协议：one-class，官方 train 正常帧；每场景 test concat 后一次 global AUC/EER；seed42，30 epoch，256×256。
- 结构说明：GL192; OCBE fuse_hidden=128; 12 teacher taps; group-balanced top5 loss (alpha=0.5).
- 矩阵总表：[20261103_variant_matrix_summary.md](../../../../experiments/da/appearance/20261103_variant_matrix_summary.md)
- 数据/泄漏审计：[20261103_variant_matrix_data_audit.md](../../../../experiments/da/appearance/20261103_variant_matrix_data_audit.md)

## 结果

macro AUC：**83.9925**　macro EER：**0.2029**

| scene | AUC | EER |
|---|---:|---:|
| bike | 97.7566 | 0.0459 |
| crossroads | 59.7789 | 0.4266 |
| farmland | 99.4880 | 0.0542 |
| highway | 93.5796 | 0.1748 |
| railway | 91.2036 | 0.1265 |
| solar | 86.5823 | 0.1942 |
| vehicle | 59.5584 | 0.3982 |

完整逐帧分数与 checkpoint 以原始实验目录为准；本条目只复制 `results/summary.{json,md}`。
