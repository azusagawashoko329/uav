# DGT-CrossAttnPool：深度组交叉注意力池化，seed42

本实验把 DGT 的固定 `1-hop/2-hop` 深度图传播和四组直接 concat 收口替换为输入相关的深度组交叉注意力及双查询池化。共享可学习随机正交 Rank-2 `Q`、组内 value/gate、学生、三对一对齐、top-k loss 和训练流程不变。

四个深度组形成 `H∈R^(4×64)`。先计算

\[
A=\operatorname{softmax}(QK^\top/\sqrt{16}),\qquad
\widetilde H=H+W_o(AV),
\]

再用两个可学习深度查询分别池化四组：

\[
a_{jg}=\operatorname{softmax}_g(q_j^\top\widetilde h_g),\qquad
c_j=\sum_ga_{jg}\widetilde h_g,
\]

最后把两个池化结果映射到 64 维。这样组间权重随 patch 和输入变化，不再使用固定深度邻接矩阵，也不把四组直接展平为一个 256 维组向量。记录 attention entropy、pool entropy 和 diagonal mass，检查注意力是否退化为平均或恒等映射。

不增加 attention entropy 等辅助损失；注意力输出投影零初始化，使训练初期保留稳定残差路径。七场景独立训练、seed42、15 epoch、两张卡动态复用。每个场景训练结束保存 `ckpt/<scene>_final.pth`，并在 `artifacts.json` 登记 SHA-256。参数预估为 3,182,182 个可训练参数。
