# LAGA 补充消融：LLLG+GATE

完整结果：[三seed分析、场景分解与模板比较](../../../experiments/da/aggregation/20261119_laga_lggd_a0_s42/results/laga_comparison.md)。

| LLLG+GATE配置 | AUC均值 ± 样本SD | EER均值 ± 样本SD |
|---|---:|---:|
| α=0：HAP-RD w/o LAGA | 85.8234 ± 0.4714 | 0.189799 ± 0.001636 |
| α=0.5：原正式HAP-RD | 85.6552 ± 0.2803 | 0.192889 ± 0.001900 |

新增21个场景任务完成，无失败。加入LAGA的平均AUC变化为−0.1682个百分点（1/3种子提升），EER变化为+0.003090（三个种子均变差）。此前GLLL+gate Loss矩阵的LAGA平均AUC增益为+0.5937，GGGG+GELU的C→D为+0.0482；当前LLLG结果未复现一致增益。该比较是描述性模板差异，不是显著性交互检验。

原八配置publication矩阵、逐帧评分、来源和正式模型定义均保留。此处是新增的过程消融证据，不覆盖旧D或HAP-RD，也不将α=0自动改为正式模型。

[逐种子矩阵CSV](../../../experiments/da/aggregation/20261119_laga_lggd_a0_s42/results/comparison_matrix.csv)，[核验记录](../../../experiments/da/aggregation/20261119_laga_lggd_a0_s42/results/validation.json)。
