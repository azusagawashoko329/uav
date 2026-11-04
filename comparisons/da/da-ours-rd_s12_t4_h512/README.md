# da-ours-rd_s12_t4_h512

状态：`completed`　变体：`rd_s12_t4_h512`（V0）

- 原始实验：[20261103_variant_matrix_v0_rd_s12_t4_h512_s42](../../../experiments/da/appearance/20261103_variant_matrix_v0_rd_s12_t4_h512_s42)（run `20261103T144204Z_da_variant_matrix_s42_20261103_variant_matrix_v0_rd_s12_t4_h512_s42`）
- 统一协议：one-class，官方 train 正常帧；每场景 test concat 后一次 global AUC/EER；seed42，30 epoch，256×256。
- 结构说明：12-block plain; teacher taps [5,7,9,11]; tap-mean top5 loss.
- 矩阵总表：[20261103_variant_matrix_summary.md](../../../experiments/da/appearance/20261103_variant_matrix_summary.md)
- 数据/泄漏审计：[20261103_variant_matrix_data_audit.md](../../../experiments/da/appearance/20261103_variant_matrix_data_audit.md)

## 结果

macro AUC：**84.7916**　macro EER：**0.2029**

| scene | AUC | EER |
|---|---:|---:|
| bike | 98.0517 | 0.0423 |
| crossroads | 58.8085 | 0.4290 |
| farmland | 99.4230 | 0.0529 |
| highway | 93.1645 | 0.2040 |
| railway | 96.7101 | 0.0995 |
| solar | 87.1368 | 0.1779 |
| vehicle | 60.2465 | 0.4145 |

完整逐帧分数与 checkpoint 以原始实验目录为准；本条目只复制 `results/summary.{json,md}`。
