# Seed43 七场景复核：192+dense128+rank2

训练源码 `07af639ce4514b238ab207afe04b604ae7cde477`；运行 `20261013T100419Z_seed43_four_variant_matrix`。28 项调度均零返回。逐帧分数重新计算 AUC，并与原始 `.npy` 标签、帧键、数据指纹及对应 seed42 实验配对核验。

| 场景 | seed42 AUC | seed43 AUC | Δ43−42 (pp) | 帧数 | GPU（PyTorch） |
|---|---:|---:|---:|---:|---|
| bike | 97.6400 | 97.5565 | -0.0835 | 18427 | NVIDIA RTX A6000 |
| crossroads | 61.1349 | 62.5551 | +1.4201 | 6244 | NVIDIA RTX 5880 Ada Generation |
| farmland | 99.1445 | 99.1797 | +0.0352 | 2387 | NVIDIA RTX A6000 |
| highway | 94.9078 | 95.1196 | +0.2118 | 2820 | NVIDIA RTX 5880 Ada Generation |
| railway | 86.3636 | 84.5787 | -1.7849 | 882 | NVIDIA RTX A6000 |
| solar | 82.7100 | 81.6683 | -1.0417 | 2450 | NVIDIA RTX A6000 |
| vehicle | 57.2450 | 57.6023 | +0.3573 | 2633 | NVIDIA RTX 5880 Ada Generation |
| **七场景宏平均** | **82.7351** | **82.6086** | **-0.1265** | | |

可训练 OCBE＋学生参数：**3,761,216**。

观察：两颗种子各自七场景宏平均的均值为 **82.6718**。场景差异和与其他结构的配对比较见 [四结构总表](../../20261013_seed43_glll192_ocbe512_s43/results/four_variant_review.md)。

限制：只有 seed42/43 两颗种子；宏平均差异不能直接解释为稳定性能提升。不同场景独立训练和测试，不能在测试集上事后选择最优结构作为无偏结果。此参数量不含冻结教师、YOLO 行为分支或 KNN 数据库；无人机端时延未测。
