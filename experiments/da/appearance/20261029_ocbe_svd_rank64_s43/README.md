# Corrected SVD initialized rank64 factorized head seed43，seed43

本实验加载同 seed43 的 Rank2 baseline checkpoint，保留训练后的 Q、12 个 tap LayerNorm 和学生权重，只将 `6144→128` 融合矩阵做 rank64 SVD 分解。先记录直接压缩 AUC，再在原 group-balanced top-k loss 下微调 15 epoch。与 seed42 版本不同，本轮明确载入 baseline 的全部非融合层状态，保证函数保持型初始化。

教师端仍为 12 taps、相邻三层一组、共享可学习随机正交 `Q∈R^(2×3)`；学生为 GL192，三对一对齐和原损失不变。七场景独立训练，15 epoch、batch64、两张卡动态调度。

## 评测

记录逐场景 AUC、训练耗时、参数量、checkpoint SHA；SVD 版本额外记录未微调的直接压缩 AUC、微调后 AUC 和推理速度。

## 限制

单种子；SVD 是从训练后 baseline checkpoint 开始的压缩微调，不等同于随机初始化端到端训练；AUC 没有直接 loss 代理。
