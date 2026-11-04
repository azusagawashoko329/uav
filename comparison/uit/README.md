# UIT-ADrone 对比空间

与 `comparison/da/` 使用同一 [record schema](../SCHEMA.md) 和统一 One-Class 协议；目前没有已登记条目。

Ours 的 UIT 六变体 45 epoch 矩阵仍保存在 `experiments/uit/appearance/20261101_uit_45ep_*_retry2_s42/`。需要把 UIT 纳入对比表时，按 `comparison/da/ours/*` 的方式建 `comparison/uit/<Method>/<variant>/`（方法名保留官方大小写），填 `record.json` 并运行：

```bash
python comparison/build_catalog.py
python comparison/check_comparisons.py
```

外部方法同样等待上游 URL；UIT 的评测口径是官方 51 个 test 视频拼接后的单 AUC/EER，不能直接照搬 DA 的七场景 macro。
