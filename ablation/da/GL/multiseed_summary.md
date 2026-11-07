# DA / GL 三 seed 消融矩阵总结

本汇总合并 seed42、seed43、seed44 的八个 GL 变体。所有配置固定 V2 母模板：DINOv3 ViT-B/16、12 teacher taps、OCBE hidden 512、code_dim 64、student width 768、4 stages、12 heads、30 epoch、batch64、AdamW + OneCycleLR。训练与测试均使用 `matched_score(k=5, alpha=0.5)`。每个 seed 的 AUC/EER 先按七场景等权宏平均，再对三个 seed 计算均值和样本标准差。

## 三 seed 总体矩阵

| block order | FFN | params | seed42 AUC | seed43 AUC | seed44 AUC | mean AUC ± SD | mean EER ± SD |
|---|---|---:|---:|---:|---:|---:|---:|
| GGGG | gate | 35,540,288 | 85.3199 | 85.5271 | 86.1336 | 85.6602 ± 0.4229 | 0.1914 ± 0.0035 |
| LLLG | gate | 30,248,000 | 85.3461 | 85.8928 | 85.7268 | 85.6552 ± 0.2803 | 0.1929 ± 0.0019 |
| LLLL | gate | 28,483,904 | 85.5609 | 85.1606 | 86.1333 | 85.6183 ± 0.4889 | 0.1972 ± 0.0049 |
| GLLL | gelumlp | 34,966,592 | 85.5066 | 85.2601 | 85.4011 | 85.3893 ± 0.1237 | 0.1954 ± 0.0019 |
| GLLL | gate | 30,248,000 | 85.2147 | 85.6322 | 85.2080 | 85.3516 ± 0.2430 | 0.1965 ± 0.0021 |
| GGGG | gelumlp | 40,258,880 | 85.1314 | 85.0532 | 85.7929 | 85.3258 ± 0.4064 | 0.1970 ± 0.0053 |
| LLLL | gelumlp | 33,202,496 | 84.7780 | 85.0777 | 85.4373 | 85.0977 ± 0.3301 | 0.1990 ± 0.0047 |
| LLLG | gelumlp | 34,966,592 | 85.5263 | 84.0347 | 85.1234 | 84.8948 ± 0.7716 | 0.1973 ± 0.0073 |

## 三 seed 逐场景 AUC 均值

| scene | GGGG gate | GGGG gelumlp | LLLL gate | LLLL gelumlp | GLLL gate | GLLL gelumlp | LLLG gate | LLLG gelumlp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bike | 97.72 | 97.74 | 97.70 | 97.65 | 97.73 | 97.67 | 97.71 | 97.67 |
| crossroads | 59.71 | 60.07 | 60.73 | 60.98 | 61.86 | 61.44 | 59.75 | 58.84 |
| farmland | 99.46 | 99.51 | 99.43 | 99.48 | 99.42 | 99.52 | 99.42 | 99.47 |
| highway | 95.25 | 94.89 | 93.95 | 93.78 | 94.51 | 94.24 | 94.10 | 94.30 |
| railway | 95.81 | 95.17 | 95.55 | 94.99 | 95.55 | 94.46 | 95.63 | 94.79 |
| solar | 89.14 | 89.44 | 89.04 | 89.37 | 89.76 | 89.39 | 89.57 | 89.19 |
| vehicle | 62.52 | 60.46 | 62.93 | 59.44 | 58.63 | 61.02 | 63.41 | 59.99 |

## 矩阵分析

- 三 seed 宏平均 AUC 最高为 `GGGG + gate`：85.6602% ± 0.4229；宏平均 EER 最低为 `GGGG + gate`：0.1914 ± 0.0035。
- 三 seed 下的排序比单 seed 更稳定地反映结构差异；本次三 seed 宏平均中 AUC 与 EER 均由 `GGGG + gate` 最优。
- FFN 从 gate 换成 gelumlp，在 `GGGG` 上跨 seed 的 AUC 变化均值为 -0.3344 个百分点，EER 变化均值为 +0.0057；该效应随 block order 改变，不能单独解释为 FFN 的普遍收益。
- FFN 从 gate 换成 gelumlp，在 `LLLL` 上跨 seed 的 AUC 变化均值为 -0.5206 个百分点，EER 变化均值为 +0.0018；该效应随 block order 改变，不能单独解释为 FFN 的普遍收益。
- FFN 从 gate 换成 gelumlp，在 `GLLL` 上跨 seed 的 AUC 变化均值为 +0.0377 个百分点，EER 变化均值为 -0.0011；该效应随 block order 改变，不能单独解释为 FFN 的普遍收益。
- FFN 从 gate 换成 gelumlp，在 `LLLG` 上跨 seed 的 AUC 变化均值为 -0.7604 个百分点，EER 变化均值为 +0.0044；该效应随 block order 改变，不能单独解释为 FFN 的普遍收益。
- 相对 GLLL，`GGGG` 在 gate 下 AUC 变化均值 +0.3086、EER -0.0051；在 gelumlp 下 AUC -0.0634、EER +0.0016。
- 相对 GLLL，`LLLL` 在 gate 下 AUC 变化均值 +0.2667、EER +0.0007；在 gelumlp 下 AUC -0.2916、EER +0.0035。
- 相对 GLLL，`LLLG` 在 gate 下 AUC 变化均值 +0.3036、EER -0.0036；在 gelumlp 下 AUC -0.4945、EER +0.0018。
- 逐场景均值显示 Crossroads 和 Vehicle 仍是主要区分场景；Bike、Farmland、Railway 的 AUC 普遍较高，配置间差异相对小。
- 该结果覆盖三个 seed，但仍只有三种子、每种子七个独立场景模型；标准差用于描述稳定性，不能替代正式显著性检验。

## 运行证据

- seed43/44 共 112 个场景任务由混合 seed 双卡动态队列完成，scheduler 报告 `completed=112, skipped=0, failures=[]`。
- 16 个实验目录分别保留场景 JSON、逐帧 CSV、checkpoint 哈希和 scheduler 日志；大权重仍由 `.gitignore` 排除。
