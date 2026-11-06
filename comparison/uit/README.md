# UIT-ADrone 对比空间

与 `comparison/da/` 使用同一 [record schema](../SCHEMA.md) 和统一 One-Class 协议。Ours 六变体由 `experiments/uit/appearance/20261104_variant_matrix_v*_s42/` 训练后，通过 `scripts/uit_variant_transfer.py` 归档为 `comparison/uit/ours/<variant>/`。六项均已归档，评分为训练匹配的 group-balanced top-k：

| Variant | AUC | EER |
|---|---:|---:|
| `rd_s12_t4_h512` (V0) | 85.2491 | 0.2158 |
| `rd_s4_t12_h512` (V1) | 82.6516 | 0.2464 |
| `rd_s4_t12_gl768_h512` (V2) | 81.7358 | 0.2586 |
| `rd_s4_t12_gl192_h512` (V3) | 76.2947 | 0.3166 |
| `rd_s4_t12_gl192_h128` (V4) | 75.7384 | 0.3208 |
| `rd_s4_t12_gl192_h128_qr2` (V5) | 75.5807 | 0.3234 |

UIT 是 51 个 test 视频拼接后的单 AUC/EER，不能直接照搬 DA 的七场景 macro。

旧版 `20261101_uit_45ep_*_retry2_s42` 已完成同配置 45 epoch 矩阵，保留为历史实验；20261104 matrix 重跑完成后按同样方式归档。手动归档时按 `comparison/da/ours/*` 的方式建 `comparison/uit/<Method>/<variant>/`（方法名保留官方大小写），填 `record.json` 并运行：

```bash
python comparison/build_catalog.py
python comparison/check_comparisons.py
```

外部方法同样等待上游 URL；UIT 的评测口径是官方 51 个 test 视频拼接后的单 AUC/EER，不能直接照搬 DA 的七场景 macro。

完整主表见 [comparison_summary.md](comparison_summary.md)。
