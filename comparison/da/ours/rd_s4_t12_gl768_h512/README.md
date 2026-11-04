# da-ours-rd_s4_t12_gl768_h512

状态：`completed`　变体：`rd_s4_t12_gl768_h512`（V2）

- 原始实验：[20261103_variant_matrix_v2_rd_s4_t12_gl768_h512_s42](../../../../experiments/da/appearance/20261103_variant_matrix_v2_rd_s4_t12_gl768_h512_s42)（run `20261103T144204Z_da_variant_matrix_s42_20261103_variant_matrix_v2_rd_s4_t12_gl768_h512_s42`）
- 统一协议：one-class，官方 train 正常帧；每场景 test concat 后一次 global AUC/EER；seed42，30 epoch，256×256。
- 结构说明：4-block Global-Local width 768; 12 teacher taps; group-balanced top5 loss (alpha=0.5).
- 矩阵总表：[20261103_variant_matrix_summary.md](../../../../experiments/da/appearance/20261103_variant_matrix_summary.md)
- 数据/泄漏审计：[20261103_variant_matrix_data_audit.md](../../../../experiments/da/appearance/20261103_variant_matrix_data_audit.md)

## 结果

macro AUC：**85.3707**　macro EER：**0.1932**

| scene | AUC | EER |
|---|---:|---:|
| bike | 97.6475 | 0.0467 |
| crossroads | 63.0526 | 0.4198 |
| farmland | 99.2187 | 0.0650 |
| highway | 95.8580 | 0.1365 |
| railway | 93.9081 | 0.1057 |
| solar | 90.0914 | 0.1665 |
| vehicle | 57.8189 | 0.4120 |

完整逐帧分数与 checkpoint 以原始实验目录为准；本条目只复制 `results/summary.{json,md}`。
