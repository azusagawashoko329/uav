# da-ours-rd_s4_t12_gl192_h512

状态：`completed`　变体：`rd_s4_t12_gl192_h512`（V3）

- 原始实验：[20261103_variant_matrix_v3_rd_s4_t12_gl192_h512_s42](../../../../experiments/da/appearance/20261103_variant_matrix_v3_rd_s4_t12_gl192_h512_s42)（run `20261103T144204Z_da_variant_matrix_s42_20261103_variant_matrix_v3_rd_s4_t12_gl192_h512_s42`）
- 统一协议：one-class，官方 train 正常帧；每场景 test concat 后一次 global AUC/EER；seed42，30 epoch，256×256。
- 结构说明：4-block Global-Local width 192; 12 teacher taps; group-balanced top5 loss (alpha=0.5).
- 矩阵总表：[20261103_variant_matrix_summary.md](../../../../experiments/da/appearance/20261103_variant_matrix_summary.md)
- 数据/泄漏审计：[20261103_variant_matrix_data_audit.md](../../../../experiments/da/appearance/20261103_variant_matrix_data_audit.md)

## 结果

macro AUC：**84.0864**　macro EER：**0.1992**

| scene | AUC | EER |
|---|---:|---:|
| bike | 97.6565 | 0.0453 |
| crossroads | 60.8752 | 0.4273 |
| farmland | 99.3140 | 0.0603 |
| highway | 94.8593 | 0.1352 |
| railway | 90.8856 | 0.1229 |
| solar | 86.4232 | 0.2033 |
| vehicle | 58.5910 | 0.4000 |

完整逐帧分数与 checkpoint 以原始实验目录为准；本条目只复制 `results/summary.{json,md}`。
