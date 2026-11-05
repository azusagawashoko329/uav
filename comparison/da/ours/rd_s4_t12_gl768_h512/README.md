# da-ours-rd_s4_t12_gl768_h512

状态：`completed`　变体：`rd_s4_t12_gl768_h512`（V2）

- 原始实验：[20261103_variant_matrix_v2_rd_s4_t12_gl768_h512_s42](../../../../experiments/da/appearance/20261103_variant_matrix_v2_rd_s4_t12_gl768_h512_s42)（run `20261103T144204Z_da_variant_matrix_s42_20261103_variant_matrix_v2_rd_s4_t12_gl768_h512_s42`）
- 统一协议：one-class，官方 train 正常帧；每场景 test concat 后一次 global AUC/EER；seed42，30 epoch，256×256。
- 结构说明：4-block Global-Local width 768; 12 teacher taps; group-balanced top5 loss (alpha=0.5).
- 矩阵总表：[20261103_variant_matrix_summary.md](../../../../experiments/da/appearance/20261103_variant_matrix_summary.md)
- 数据/泄漏审计：[20261103_variant_matrix_data_audit.md](../../../../experiments/da/appearance/20261103_variant_matrix_data_audit.md)

## 结果

macro AUC：**85.2147**　macro EER：**0.1965**

| scene | AUC | EER |
|---|---:|---:|
| bike | 97.7002 | 0.0459 |
| crossroads | 61.5545 | 0.4332 |
| farmland | 99.4576 | 0.0550 |
| highway | 94.9450 | 0.1593 |
| railway | 95.4762 | 0.1069 |
| solar | 89.7237 | 0.1683 |
| vehicle | 57.6453 | 0.4066 |

完整逐帧分数与 checkpoint 以原始实验目录为准；本条目只复制 `results/summary.{json,md}`。
