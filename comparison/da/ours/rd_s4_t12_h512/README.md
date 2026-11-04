# da-ours-rd_s4_t12_h512

状态：`completed`　变体：`rd_s4_t12_h512`（V1）

- 原始实验：[20261103_variant_matrix_v1_rd_s4_t12_h512_s42](../../../../experiments/da/appearance/20261103_variant_matrix_v1_rd_s4_t12_h512_s42)（run `20261103T144204Z_da_variant_matrix_s42_20261103_variant_matrix_v1_rd_s4_t12_h512_s42`）
- 统一协议：one-class，官方 train 正常帧；每场景 test concat 后一次 global AUC/EER；seed42，30 epoch，256×256。
- 结构说明：4-block plain; 12 teacher taps 0..11; group-balanced top5 loss (alpha=0.5).
- 矩阵总表：[20261103_variant_matrix_summary.md](../../../../experiments/da/appearance/20261103_variant_matrix_summary.md)
- 数据/泄漏审计：[20261103_variant_matrix_data_audit.md](../../../../experiments/da/appearance/20261103_variant_matrix_data_audit.md)

## 结果

macro AUC：**84.9435**　macro EER：**0.2011**

| scene | AUC | EER |
|---|---:|---:|
| bike | 97.6834 | 0.0454 |
| crossroads | 60.8492 | 0.4347 |
| farmland | 99.2496 | 0.0633 |
| highway | 95.7209 | 0.1283 |
| railway | 94.4428 | 0.1044 |
| solar | 89.4215 | 0.1840 |
| vehicle | 57.2374 | 0.4476 |

完整逐帧分数与 checkpoint 以原始实验目录为准；本条目只复制 `results/summary.{json,md}`。
