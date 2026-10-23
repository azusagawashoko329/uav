# DGT-NoGraph：组内门控消融，seed42

本实验（带最终权重保存）检查上一版 DGT 的 AUC 变化是否来自固定 `1-hop/2-hop` 深度图传播。共享可学习、随机正交初始化的 Rank-2 `Q`、四组三层教师压缩、组内 value/gate、192 维 Global-Local 学生、三对一对齐和 group-balanced top-k loss 全部不变；仅移除跨组 `P/P²` 传播，四组门控结果直接进入原有输出投影。

对每个 patch，组内仍为

\[
h_g=\operatorname{GELU}(W_vz_g)\odot(1+\tanh(W_gz_g)).
\]

输出为 `Linear(4×64,64)`。该版本不增加辅助损失，避免把结构变量与新的优化目标混在一起。记录 gate 均值、标准差和隐藏特征 RMS，结果与 [DGT seed42](../20261022_ocbe_dgt64_s42/results/summary.md) 对照。

七场景独立训练、seed42、15 epoch、两张卡动态复用。每个场景训练结束保存 `ckpt/<scene>_final.pth`，并在 `artifacts.json` 登记 SHA-256。参数预估为 3,179,590 个可训练参数。
