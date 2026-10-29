# seed43 Rank2 与 SVD-rank64 联合复盘

| 版本 | mean AUC | 参数 | batch1 core FPS | batch1 latency |
|---|---:|---:|---:|---:|
| Rank2 refresh | 82.2651 | 3,761,222 | 610.1 | 1.6393 ms |
| SVD-rank64 fine-tuned | 83.8732 | 3,376,198 | 605.2 | 1.6526 ms |

SVD 直接压缩 mean AUC 为 81.6535，微调后为 83.8732；相对 seed43 Rank2 refresh +1.6082 pp。参数减少约 10.2%，核心 FPS 基本持平。Solar 和 Railway 提升明显，Vehicle 下降，需要多种子验证。
