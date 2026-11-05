# da-ours-rd_s4_t12_gl192_h128_qr2

状态：`completed`　变体：`rd_s4_t12_gl192_h128_qr2`（V5）

- 原始实验：[20261103_variant_matrix_v5_rd_s4_t12_gl192_h128_qr2_s42](../../../../experiments/da/appearance/20261103_variant_matrix_v5_rd_s4_t12_gl192_h128_qr2_s42)（run `20261103T144204Z_da_variant_matrix_s42_20261103_variant_matrix_v5_rd_s4_t12_gl192_h128_qr2_s42`）
- 统一协议：one-class，官方 train 正常帧；每场景 test concat 后一次 global AUC/EER；seed42，30 epoch，256×256。
- 结构说明：GL192; OCBE fuse_hidden=128; 12 teacher taps; learnable rank-2 Q (random orthonormal init); group-balanced top5 loss (alpha=0.5).
- 矩阵总表：[20261103_variant_matrix_summary.md](../../../../experiments/da/appearance/20261103_variant_matrix_summary.md)
- 数据/泄漏审计：[20261103_variant_matrix_data_audit.md](../../../../experiments/da/appearance/20261103_variant_matrix_data_audit.md)

## 结果

macro AUC：**84.6233**　macro EER：**0.1970**

| scene | AUC | EER |
|---|---:|---:|
| bike | 97.8152 | 0.0432 |
| crossroads | 60.4277 | 0.4345 |
| farmland | 99.5703 | 0.0485 |
| highway | 94.4218 | 0.1528 |
| railway | 92.1990 | 0.1229 |
| solar | 87.5511 | 0.1936 |
| vehicle | 60.3781 | 0.3837 |

完整逐帧分数与 checkpoint 以原始实验目录为准；本条目只复制 `results/summary.{json,md}`。
