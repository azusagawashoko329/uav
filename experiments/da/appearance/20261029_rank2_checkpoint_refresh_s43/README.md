# Rank2 baseline checkpoint refresh seed43，seed43

本实验完全复现共享可学习、随机正交初始化的 Rank2 Q baseline，seed43，并保存七个场景 checkpoint，供同 seed 的 SVD-rank64 压缩使用。它不是新的架构结论。

教师端仍为 12 taps、相邻三层一组、共享可学习随机正交 `Q∈R^(2×3)`；学生为 GL192，三对一对齐和原损失不变。七场景独立训练，15 epoch、batch64、两张卡动态调度。

## 评测

记录逐场景 AUC、训练耗时、参数量、checkpoint SHA；SVD 版本额外记录未微调的直接压缩 AUC、微调后 AUC 和推理速度。

## 限制

单种子；SVD 是从训练后 baseline checkpoint 开始的压缩微调，不等同于随机初始化端到端训练；AUC 没有直接 loss 代理。
