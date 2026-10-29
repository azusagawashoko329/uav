# Rank2 Learnable Q random-orthonormal baseline checkpoint refresh，seed42

## 研究问题

本实验围绕 Rank2 Learnable Q 基线继续深化 OCBE。教师端仍按相邻三层组成四组，每组使用共享可学习随机正交 `Q∈R^(2×3)`，学生仍为 GL192，三层教师对应一个学生 block，损失仍为原 group-balanced top-k。Only reproduces the established Rank2 Learnable-Q baseline and writes scene checkpoints for the SVD compression stage.

## 唯一变化

- 场景独立训练：bike, crossroads, farmland, highway, railway, solar, vehicle。
- Q、patch-wise 输入、student、对齐和 loss 不变。
- 不引入辅助损失。
- 两张 RTX 5880 Ada 由动态调度器复用，15 epoch、batch64、seed42。

该实验仅用于产生与历史 Rank2 Learnable Q AUC 对应的完整场景 checkpoint，不是新的模型结论。

## 评测

记录每场景 AUC、参数量、checkpoint SHA 和训练耗时。SVD 版本额外记录直接压缩 AUC。完成后与 Rank2、LCDB、Cross-Depth 和 SGF 结果对照。

## 限制

单种子；AUC 没有直接 loss 代理；GPU 时钟不固定；SVD 版本是从训练后 baseline 做结构化压缩，和从随机初始化端到端训练的结果分开解释。

## 最终结果

七场景 mean AUC 为 **82.9612**，可训练参数为 **3,761,222**。结果见 [`summary.md`](results/summary.md)、[`speed_summary.md`](results/speed_summary.md)。

该轮复现 Rank2 Learnable Q，mean AUC 为 82.9612，七个场景 checkpoint 作为 SVD 压缩的初始化来源。
