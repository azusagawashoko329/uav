# HAP-RD-Lite Q 定义与历史结果边界

## 当前定义

从当前论文定义开始，HAP-RD-Lite 的 PODP 使用四个独立的可学习 Q。

- 每个相邻三层 teacher group 使用一个独立的 `2×3` Q，四组总计 `4×2×3=24` 个原始可训练标量。
- 每个 Q 按 seed 独立随机正交初始化。
- 前向对每个 Q 单独执行可微 Gram-Schmidt，使两行保持正交。
- 训练和测试都使用更新后的四个 group-specific Q。
- 当前机器可识别入口为 `hap_rd_lite_per_q`，DA 已完成运行的变体 ID 为 `hap_rd_lite_q4`。
- 可训练参数量为 3,761,240，比共享-Q结构多 18 个标量。

对应实现为 [UIT per-Q 模型](../model/hap_lllg_per_q.py) 和 [DA per-Q 模型](../model/bottleneck_depthrank_group_learnable.py)。

## 历史共享-Q运行

下列完成结果严格保留原始共享-Q口径，不能改写成当前独立-Q模型的结果。

| 数据集 | 历史实验 | Q形状 | 参数量 | 状态 |
|---|---|---:|---:|---|
| DA | `ablation/da/publication/20261117_hap_rd_lite_s42/s43/s44` | `2×3`，四组共用 | 3,761,222 | 历史正式共享-Q |
| UIT | `experiments/uit/appearance/20261120_formal_lllg_hap_rd_lite_s42/s43/s44` | `2×3`，四组共用 | 3,761,222 | 历史正式共享-Q |

这些目录中的配置、source commit、checkpoint、日志和指标保持不变。引用它们时应注明“shared-Q historical run”。

## 当前独立-Q证据

| 数据集 | 实验 | Q形状 | 参数量 | 状态 |
|---|---|---:|---:|---|
| DA | `experiments/da/appearance/20261126_podp_q_per_group_s42/s43/s44` | `4×2×3` | 3,761,240 | 已完成 |
| UIT | `experiments/uit/appearance/20261127_formal_lllg_hap_rd_lite_per_q_s42/s43/s44` | `4×2×3` | 3,761,240 | 已完成 |

新旧结果不得按同一 variant 混合汇总。DA 和 UIT 的共享-Q与独立-Q配对分析分别保存在各自实验目录中。
