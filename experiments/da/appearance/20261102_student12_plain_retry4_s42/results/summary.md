# DA Student12 Plain12 vs GL768 full seven-scene summary

将 GL12 原实验中已完成的四个场景（bike、crossroads、farmland、highway）与本次重跑的三个场景（railway、solar、vehicle）合并后，与 Plain12 的七场景结果逐场比较。

| 场景 | GL 数据来源 | Plain AUC | GL AUC | Δ AUC | Plain EER | GL EER | Δ EER |
|---|---|---:|---:|---:|---:|---:|---:|
| bike | 原实验 | 98.056 | 98.057 | +0.000 | 0.0414 | 0.0407 | -0.0006 |
| crossroads | 原实验 | 60.234 | 62.006 | +1.772 | 0.4202 | 0.4055 | -0.0147 |
| farmland | 原实验 | 99.428 | 99.414 | -0.014 | 0.0520 | 0.0537 | +0.0017 |
| highway | 原实验 | 92.968 | 93.395 | +0.426 | 0.2079 | 0.1997 | -0.0082 |
| railway | 重跑 | 96.389 | 97.969 | +1.581 | 0.1032 | 0.0737 | -0.0295 |
| solar | 重跑 | 86.999 | 87.281 | +0.282 | 0.1834 | 0.1797 | -0.0036 |
| vehicle | 重跑 | 64.783 | 60.912 | -3.871 | 0.3554 | 0.3922 | +0.0367 |

七场景宏平均：Plain12 AUC **85.551**、EER **0.1948**；GL12 AUC **85.576**、EER **0.1922**。GL12 相对 Plain12 的宏平均变化为 AUC **+0.025**、EER **-0.0026**。

按全部测试帧数加权的 AUC：Plain12 **87.918**，GL12 **88.033**，GL12 变化 **+0.115**。

观察：GL12 在 bike、crossroads、highway、railway、solar 上 AUC 高于 Plain12；farmland 基本持平；vehicle 低于 Plain12。EER 方向与 AUC 整体一致，但 vehicle 明显变差。以上是单 seed42 的结果。
