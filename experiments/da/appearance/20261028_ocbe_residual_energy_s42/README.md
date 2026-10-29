# Rank2 + zero-initialized discarded-depth residual energy side channel，seed42

## 研究问题

本实验围绕 Rank2 Learnable Q 基线继续深化 OCBE。教师端仍按相邻三层组成四组，每组使用共享可学习随机正交 `Q∈R^(2×3)`，学生仍为 GL192，三层教师对应一个学生 block，损失仍为原 group-balanced top-k。Adds four log energies from the orthogonal complement of Q through a zero-initialized 4->128 side projection before the existing GELU.

## 唯一变化

- 场景独立训练：bike, crossroads, farmland, highway, railway, solar, vehicle。
- Q、patch-wise 输入、student、对齐和 loss 不变。
- 不引入辅助损失。
- 两张 RTX 5880 Ada 由动态调度器复用，15 epoch、batch64、seed42。

用 `q_perp=q_0×q_1` 得到 Q 丢弃的唯一深度方向，对每组计算 `log(mean((q_perp^T t)^2))`，得到四个标量。它们通过零初始化的 `4→128` 侧投影进入原始第一层隐藏表示，保留第三方向的能量而不恢复完整 768 维。

## 评测

记录每场景 AUC、参数量、checkpoint SHA 和训练耗时。SVD 版本额外记录直接压缩 AUC。完成后与 Rank2、LCDB、Cross-Depth 和 SGF 结果对照。

## 限制

单种子；AUC 没有直接 loss 代理；GPU 时钟不固定；SVD 版本是从训练后 baseline 做结构化压缩，和从随机初始化端到端训练的结果分开解释。

## 最终结果

七场景 mean AUC 为 **82.3511**，可训练参数为 **3,761,734**。结果见 [`summary.md`](results/summary.md)、[`speed_summary.md`](results/speed_summary.md)。

残差能量侧通道比 baseline 低 0.6101 pp，当前单种子下没有证明被 Q 丢弃的深度能量可以提升 AUC。
