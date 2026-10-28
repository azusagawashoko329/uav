# Rank2 + zero-initialized 10D cross-group Gram side channel，seed42

## 研究问题

本实验围绕 Rank2 Learnable Q 基线继续深化 OCBE。教师端仍按相邻三层组成四组，每组使用共享可学习随机正交 `Q∈R^(2×3)`，学生仍为 GL192，三层教师对应一个学生 block，损失仍为原 group-balanced top-k。Adds four group log-RMS values and six pairwise cosine values through a zero-initialized 10->128 side projection before the existing GELU.

## 唯一变化

- 场景独立训练：bike, crossroads, farmland, highway, railway, solar, vehicle。
- Q、patch-wise 输入、student、对齐和 loss 不变。
- 不引入辅助损失。
- 两张 RTX 5880 Ada 由动态调度器复用，15 epoch、batch64、seed42。

对每个 patch 的四组 Rank2 向量 `z_g∈R^1536` 计算归一化 Gram 关系。侧特征由四个 `log(RMS(z_g))` 和六个组间 cosine 组成，共 10 维。第一层隐藏表示为 `h=W_x concat(z_g)+W_s side+b`，其中 `W_s` 零初始化，保证初始函数等价于 Rank2 基线。

## 评测

记录每场景 AUC、参数量、checkpoint SHA 和训练耗时。SVD 版本额外记录直接压缩 AUC。完成后与 Rank2、LCDB、Cross-Depth 和 SGF 结果对照。

## 限制

单种子；AUC 没有直接 loss 代理；GPU 时钟不固定；SVD 版本是从训练后 baseline 做结构化压缩，和从随机初始化端到端训练的结果分开解释。
