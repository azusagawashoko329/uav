# DGT-Bilinear：相邻深度组低秩双线性交互，seed42

本实验用低秩二阶交互替换 DGT 的固定深度图传播。共享可学习随机正交 Rank-2 `Q`、组内 value/gate、学生、三对一对齐、top-k loss 和训练流程不变。

对于相邻深度组：

\[
r_g=\phi(Ah_g)\odot\phi(Bh_{g+1}),\qquad
\widetilde h_g=h_g+Ur_g,
\quad g=1,2,3.
\]

其中交互秩为 16，`U` 零初始化，保证初始仍有稳定的残差主路径。该结构明确建模相邻教师深度组的乘法关系，参数远少于完整四组二阶矩阵；最终仍使用 `Linear(4×64,64)` 输出 code。记录双线性交互增量 RMS，判断新路径是否被训练使用。

不增加辅助损失，避免结构和优化目标同时变化。七场景独立训练、seed42、15 epoch、两张卡动态复用。每个场景训练结束保存 `ckpt/<scene>_final.pth`，并在 `artifacts.json` 登记 SHA-256。参数预估为 3,182,758 个可训练参数。
