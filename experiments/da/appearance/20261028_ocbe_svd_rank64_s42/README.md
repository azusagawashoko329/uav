# Rank2 Learnable Q with SVD initialized rank64 factorized 6144->128 head，seed42

## 研究问题

本实验围绕 Rank2 Learnable Q 基线继续深化 OCBE。教师端仍按相邻三层组成四组，每组使用共享可学习随机正交 `Q∈R^(2×3)`，学生仍为 GL192，三层教师对应一个学生 block，损失仍为原 group-balanced top-k。Loads the matched Rank2 checkpoint, replaces its 6144->128 matrix by an SVD rank64 factorization, records direct compression AUC, then fine-tunes under the unchanged loss.

## 唯一变化

- 场景独立训练：bike, crossroads, farmland, highway, railway, solar, vehicle。
- Q、patch-wise 输入、student、对齐和 loss 不变。
- 不引入辅助损失。
- 两张 RTX 5880 Ada 由动态调度器复用，15 epoch、batch64、seed42。

先由 `20261028_rank2_checkpoint_refresh_s42` 提供同配置的训练 checkpoint。对每个场景 Rank2 checkpoint 的 `W1∈R^(128×6144)` 做截断 SVD：`W1≈U_64Σ_64V_64^T`，替换为两个线性层。先记录直接压缩但未微调的 AUC，再在原损失下微调 15 epoch。

## 评测

记录每场景 AUC、参数量、checkpoint SHA 和训练耗时。SVD 版本额外记录直接压缩 AUC。完成后与 Rank2、LCDB、Cross-Depth 和 SGF 结果对照。

## 限制

单种子；AUC 没有直接 loss 代理；GPU 时钟不固定；SVD 版本是从训练后 baseline 做结构化压缩，和从随机初始化端到端训练的结果分开解释。
