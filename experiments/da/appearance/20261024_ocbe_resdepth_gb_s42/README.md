# 20261024_ocbe_resdepth_gb_s42

当前 DGT 主路径加零初始化低秩深度变化残差，原始 loss。

共享可学习随机正交 Rank-2 Q、四组三层教师、192 维 Global-Local 学生、三对一对齐、seed42 和训练流程保持不变。每个场景结束保存 `ckpt/<scene>_final.pth`，并登记 SHA-256。

训练 loss：`group_balanced`。

七场景独立训练，15 epoch，两张 GPU 动态复用。参数预估：OCBE 236,838，OCBE+学生 3,184,806。
