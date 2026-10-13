# Seed43 七场景复核：192+原版OCBE

训练源码 `07af639ce4514b238ab207afe04b604ae7cde477`；运行 `20261013T100419Z_seed43_four_variant_matrix`。28 项调度均零返回。逐帧分数重新计算 AUC，并与原始 `.npy` 标签、帧键、数据指纹及对应 seed42 实验配对核验。

| 场景 | seed42 AUC | seed43 AUC | Δ43−42 (pp) | 帧数 | GPU（PyTorch） |
|---|---:|---:|---:|---:|---|
| bike | 97.6071 | 97.6009 | -0.0062 | 18427 | NVIDIA RTX 5880 Ada Generation |
| crossroads | 61.2535 | 62.0001 | +0.7466 | 6244 | NVIDIA RTX A6000 |
| farmland | 99.3270 | 99.2832 | -0.0439 | 2387 | NVIDIA RTX 5880 Ada Generation |
| highway | 95.0202 | 94.7249 | -0.2953 | 2820 | NVIDIA RTX 5880 Ada Generation |
| railway | 85.0159 | 84.8190 | -0.1969 | 882 | NVIDIA RTX 5880 Ada Generation |
| solar | 82.0578 | 81.8073 | -0.2506 | 2450 | NVIDIA RTX A6000 |
| vehicle | 57.4851 | 55.9284 | -1.5567 | 2633 | NVIDIA RTX 5880 Ada Generation |
| **七场景宏平均** | **82.5381** | **82.3091** | **-0.2290** | | |

可训练 OCBE＋学生参数：**7,718,336**。

观察：两颗种子各自七场景宏平均的均值为 **82.4236**。场景差异和与其他结构的配对比较见 [四结构总表](../../20261013_seed43_glll192_ocbe512_s43/results/four_variant_review.md)。

限制：只有 seed42/43 两颗种子；宏平均差异不能直接解释为稳定性能提升。不同场景独立训练和测试，不能在测试集上事后选择最优结构作为无偏结果。此参数量不含冻结教师、YOLO 行为分支或 KNN 数据库；无人机端时延未测。
