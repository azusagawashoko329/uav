# da-ours-rd_s4_t12_gl192_h512

状态：`completed`　变体：`rd_s4_t12_gl192_h512`（V3）

- 原始实验：[20261103_variant_matrix_v3_rd_s4_t12_gl192_h512_s42](../../../../experiments/da/appearance/20261103_variant_matrix_v3_rd_s4_t12_gl192_h512_s42)（run `20261103T144204Z_da_variant_matrix_s42_20261103_variant_matrix_v3_rd_s4_t12_gl192_h512_s42`）
- 统一协议：one-class，官方 train 正常帧；每场景 test concat 后一次 global AUC/EER；seed42，30 epoch，256×256。
- 结构说明：4-block Global-Local width 192; 12 teacher taps; group-balanced top5 loss (alpha=0.5).
- 矩阵总表：[20261103_variant_matrix_summary.md](../../../../experiments/da/appearance/20261103_variant_matrix_summary.md)
- 数据/泄漏审计：[20261103_variant_matrix_data_audit.md](../../../../experiments/da/appearance/20261103_variant_matrix_data_audit.md)

## 结果

macro AUC：**84.2137**　macro EER：**0.2001**

| scene | AUC | EER |
|---|---:|---:|
| bike | 97.7989 | 0.0440 |
| crossroads | 58.3984 | 0.4485 |
| farmland | 99.5709 | 0.0481 |
| highway | 94.6346 | 0.1554 |
| railway | 92.0129 | 0.1204 |
| solar | 86.5489 | 0.2021 |
| vehicle | 60.5315 | 0.3819 |

完整逐帧分数与 checkpoint 以原始实验目录为准；本条目只复制 `results/summary.{json,md}`。
