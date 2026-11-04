# UIT-ADrone 对比空间

与 `comparison/da/` 使用同一 [record schema](../SCHEMA.md) 和统一 One-Class 协议。Ours 六变体由 `experiments/uit/appearance/20261104_variant_matrix_v*_s42/` 训练后，通过 `scripts/uit_variant_transfer.py` 归档为 `comparison/uit/ours/<variant>/`；在此之前该目录没有正式条目。

旧版 `20261101_uit_45ep_*_retry2_s42` 已完成同配置 45 epoch 矩阵，保留为历史实验；20261104 matrix 重跑完成后按同样方式归档。手动归档时按 `comparison/da/ours/*` 的方式建 `comparison/uit/<Method>/<variant>/`（方法名保留官方大小写），填 `record.json` 并运行：

```bash
python comparison/build_catalog.py
python comparison/check_comparisons.py
```

外部方法同样等待上游 URL；UIT 的评测口径是官方 51 个 test 视频拼接后的单 AUC/EER，不能直接照搬 DA 的七场景 macro。
