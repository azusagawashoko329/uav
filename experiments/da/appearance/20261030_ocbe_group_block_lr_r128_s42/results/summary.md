# Group-Block Low-Rank Fusion 结果复盘

| 场景 | Rank2 30 epoch | Group-Block LR 30 epoch | Δ LR−Rank2 |
|---|---:|---:|---:|
| highway | 94.8307 | 95.6221 | +0.7914 |
| farmland | 99.3254 | 99.1553 | -0.1701 |
| bike | 97.6593 | 97.6510 | -0.0083 |
| solar | 87.1792 | 84.9473 | -2.2320 |
| vehicle | 56.2522 | 55.7888 | -0.4634 |
| crossroads | 62.3047 | 61.0128 | -1.2919 |
| railway | 90.5044 | 89.7221 | -0.7823 |
| **mean(7)** | **84.0080** | **83.4142** | **-0.5938** |

- 可训练参数：**3,236,934**；Rank2 30 epoch 为 3,761,222，减少约 **13.9%**。
- 七场景训练耗时合计：**120.18 分钟**。
- 融合形式为共享输入投影 V、四个独立 U_g，在 GELU 前求和；没有上下文 gate、hop、attention 或额外 loss。
- 相比 GCF，AUC 均值提高约 0.49 pp，说明保留 GELU 前跨组融合比先压缩每组更稳；但仍低于 Rank2 baseline。
- 推理测速见 [`speed_summary.md`](speed_summary.md)。

## 判断

该结构是目前更合理的轻量化候选，但单 seed 下还不能作为主线替代 Rank2；Solar 和 Crossroads 仍有明显下降。
