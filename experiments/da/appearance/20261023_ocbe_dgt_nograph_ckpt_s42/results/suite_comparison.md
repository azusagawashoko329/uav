# OCBE 融合结构消融：seed42 对照汇总

四个版本均使用共享可学习随机正交 Rank-2 Q、192 维 Global-Local 学生和原有 group-balanced top-k loss；只有右侧四组融合算子不同。

| 场景 | DGT固定图 | NoGraph | CrossAttnPool | Bilinear |
|---|---:|---:|---:|---:|
| bike | 97.5717 | 97.5484 | 97.6314 | 97.6348 |
| crossroads | 62.1188 | 62.0721 | 62.3546 | 60.8573 |
| farmland | 99.2431 | 99.2409 | 99.3292 | 99.1266 |
| highway | 95.6274 | 95.5810 | 95.2689 | 93.7282 |
| railway | 83.1009 | 83.0864 | 81.8941 | 81.7658 |
| solar | 82.7286 | 82.7592 | 82.7636 | 81.2760 |
| vehicle | 57.1538 | 56.8938 | 56.6537 | 58.1310 |

| 版本 | 宏平均 AUC | 相对固定图 | 可训练参数 |
|---|---:|---:|---:|
| DGT-fixed-graph | 82.5063 | +0.0000 pp | 3,179,718 |
| DGT-NoGraph | 82.4545 | -0.0518 pp | 3,179,590 |
| DGT-CrossAttnPool | 82.2708 | -0.2356 pp | 3,182,182 |
| DGT-Bilinear | 81.7885 | -0.7178 pp | 3,182,758 |

观察：NoGraph 与固定图几乎一致，固定 1-hop/2-hop 传播只带来很小收益；CrossAttnPool 在本 seed 下下降，Bilinear 下降更明显。CrossAttnPool 的 pool entropy 接近 log(4)，说明最终两个查询的组权重多数接近平均，动态池化尚未形成稳定的深度选择。Bilinear 的交互增量在 Crossroads/Highway 较大，但宏平均下降，说明二阶组间交互可能把正常重建目标和异常可分性推向不同方向。

本汇总只用于结构筛选，单个 seed 不能推断跨场景稳健性。每个 checkpoint 版本的七个场景权重位于各自目录的 `ckpt/`，哈希见对应 `artifacts.json`。
