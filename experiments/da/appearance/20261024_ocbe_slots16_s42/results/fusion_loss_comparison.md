# 四组融合与 loss 对齐：seed42 汇总

所有版本均使用共享可学习随机正交 Rank-2 Q、192 维 Global-Local 学生、三对一对齐和 checkpoint 保存；只比较四组 code 组织方式与 score-aligned loss。

| 场景 | DGT | Slot16 | ResidualDepth-GB | ResidualDepth-ScoreAligned |
|---|---:|---:|---:|---:|
| bike | 97.5717 | 97.6343 | 97.5745 | 97.5810 |
| crossroads | 62.1188 | 63.6957 | 63.5480 | 63.7385 |
| farmland | 99.2431 | 99.2290 | 99.1596 | 99.1667 |
| highway | 95.6274 | 93.7867 | 95.1927 | 95.0752 |
| railway | 83.1009 | 81.6267 | 82.6312 | 82.7179 |
| solar | 82.7286 | 83.1987 | 81.8218 | 82.0431 |
| vehicle | 57.1538 | 54.5145 | 55.3803 | 54.4041 |

| 版本 | 宏平均 AUC | 相对 DGT | 可训练参数 |
|---|---:|---:|---:|
| DGT | 82.5063 | +0.0000 pp | 3,179,718 |
| Slot16 | 81.9551 | -0.5512 pp | 3,015,590 |
| ResidualDepth-GB | 82.1869 | -0.3194 pp | 3,184,806 |
| ResidualDepth-ScoreAligned | 82.1038 | -0.4025 pp | 3,184,806 |

结论：Slot16 将可训练参数减少 164,128（5.16%），但宏平均 AUC 下降 0.5512 pp；参数压缩造成的组内容量损失主要出现在 Highway、Railway 和 Vehicle。ResidualDepth-GB 增加 5,088 个参数，宏平均下降 0.3194 pp；ResidualDepth-ScoreAligned 使用 0.75 group-balanced + 0.25 frame mean loss，宏平均下降 0.4025 pp，未显示 score-aligned loss 能改善 AUC。

诊断：ResidualDepth 的 residual RMS 在 Crossroads/Farmland 较大，但对应的 AUC 提升不能覆盖 Highway、Railway、Solar 和 Vehicle 的下降；这说明正常样本重建梯度确实打开了残差分支，却没有形成稳定的异常排序收益。Score-aligned loss 只对齐正常帧平均分数，仍然没有异常/正常排序项，因此不能视为真正的 AUC 代理。

本汇总为 seed42 单种子结果，权重位于各实验目录的 `ckpt/`，哈希见各自 `artifacts.json`。
