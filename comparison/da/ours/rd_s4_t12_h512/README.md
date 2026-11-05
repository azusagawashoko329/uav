# da-ours-rd_s4_t12_h512

状态：`completed`　变体：`rd_s4_t12_h512`（V1）

- 原始实验：[20261103_variant_matrix_v1_rd_s4_t12_h512_s42](../../../../experiments/da/appearance/20261103_variant_matrix_v1_rd_s4_t12_h512_s42)（run `20261103T144204Z_da_variant_matrix_s42_20261103_variant_matrix_v1_rd_s4_t12_h512_s42`）
- 统一协议：one-class，官方 train 正常帧；每场景 test concat 后一次 global AUC/EER；seed42，30 epoch，256×256。
- 结构说明：4-block plain; 12 teacher taps 0..11; group-balanced top5 loss (alpha=0.5).
- 矩阵总表：[20261103_variant_matrix_summary.md](../../../../experiments/da/appearance/20261103_variant_matrix_summary.md)
- 数据/泄漏审计：[20261103_variant_matrix_data_audit.md](../../../../experiments/da/appearance/20261103_variant_matrix_data_audit.md)

## 结果

macro AUC：**85.1314**　macro EER：**0.2031**

| scene | AUC | EER |
|---|---:|---:|
| bike | 97.7429 | 0.0448 |
| crossroads | 60.1778 | 0.4400 |
| farmland | 99.4896 | 0.0537 |
| highway | 94.8154 | 0.1567 |
| railway | 95.6081 | 0.1069 |
| solar | 89.4196 | 0.1803 |
| vehicle | 58.6661 | 0.4392 |

完整逐帧分数与 checkpoint 以原始实验目录为准；本条目只复制 `results/summary.{json,md}`。
