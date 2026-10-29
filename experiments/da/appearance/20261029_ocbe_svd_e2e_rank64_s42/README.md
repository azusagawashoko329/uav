# Randomly initialized end-to-end factorized rank64 OCBE for 30 epochs，seed42

本实验从随机初始化的 rank64 factorized `6144→128` head 端到端训练，不加载任何 Rank2 baseline checkpoint，用于和同样 30 epoch 的 Rank2 baseline 公平比较。

教师端仍为 12 taps、相邻三层一组、共享可学习随机正交 Q；学生为 GL192，三对一对齐和 group-balanced top-k loss 不变。seed42、batch64、两张卡动态调度，**从头训练 30 epoch**。

## 公平性设置

Rank2 和 SVD-e2e 使用相同数据顺序、优化器、OneCycleLR 总步数和训练 epoch。SVD-e2e 的低秩融合层使用随机 factorized 初始化，不能从 baseline 继承权重；因此比较的是端到端结构，而不是训练后压缩恢复。

## 评测

记录七场景 AUC、参数量、训练耗时和训练完成权重 FPS。结果与 15 epoch Rank2、baseline→SVD 微调结果分开解释。

## 限制

单种子；30 epoch 可能改变与 MADViT 15 epoch 的外部对照口径；若 Rank2 在 30 epoch 仍持续提升，则此前 15 epoch 结果不能作为充分收敛结论。
