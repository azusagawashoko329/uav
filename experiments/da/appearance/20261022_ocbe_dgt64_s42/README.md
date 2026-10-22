# DGT-OCBE：深度组门控张量融合，seed42

## 研究问题

右侧 OCBE 的 `6144→128→64` 稠密 MLP 是否可以改为有明确轴结构的融合模块，在保持 Rank-2 教师压缩、学生结构和蒸馏损失不变的情况下，减少参数并保留七场景 AUC。

## 唯一变化

本实验沿用已汇报的共享可学习、随机正交初始化的 `2×3` 深度矩阵 `Q`。每组三个相邻教师 tap 仍压缩为两个深度方向，四组仍对应四个学生 block。唯一变化是右侧 OCBE 融合：

1. 每个深度组的 `2×768` 特征展平为 `1536`，用共享的 value projection 和 gate projection 分别映射到 64 维；
2. 使用 `value * (1 + tanh(gate))` 形成乘法门控；gate 投影零初始化，使初始门控为恒等映射；
3. 四组按教师深度顺序通过固定四节点路径图做一阶、二阶残差混合，两个通道系数零初始化；
4. 四组拼接后用 `256→64` 输出 code。

数学形式为：

```text
z_g ∈ R^(2×768) -> vec(z_g) ∈ R^1536
u_g = GELU(z_g W_v + b_v)
v_g = z_g W_g + b_g
h_g = u_g ⊙ (1 + tanh(v_g))
H' = H + α₁ P H + α₂ P² H
code = W_o vec(H') ∈ R^64
```

其中 `P` 是四个相邻教师深度组的路径图，`α₁/α₂` 初始为零。该模块仍逐 patch 计算，不引入 patch-to-patch 信息混合，也不改变输出 `[B,N,64]` 的接口。

## 对照与固定设置

- 对照：`20261013_ocbe_depthrank2_randomorth_h128_s42`，共享可学习随机正交 `Q`、Dense OCBE、192 维四 block Global-Local 学生。
- 冻结 DINOv3 ViT-B/16，12 个 teacher tap，embed dim 768；每组三层教师对应一个学生 block。
- 学生、top-k=5、group-balanced loss（alpha=0.5）、15 epoch、batch=64、AdamW、OneCycleLR、max_lr=5e-4、weight_decay=0.05 全部保持不变。
- Drone-Anomaly 七场景独立训练和独立测试，seed=42；两张 GPU 动态复用，空闲卡立即领取下一个场景。
- AUC 为逐场景逐帧 ROC AUC×100，七场景宏平均；结果同时记录可训练参数量、数据划分指纹和每场景日志。

## 参数预估

当前 Dense 右侧融合约 794,816 个参数；DGT-OCBE 右侧约 213,312 个参数。包含 12 个 tap LayerNorm 和 6 个 Q 参数时，OCBE 预计 231,750 个参数；加 192 维 Global-Local 学生后预计总可训练参数为 3,179,718。

## 判读

若宏平均 AUC 在已有 seed42 波动范围内，同时参数下降约 73%，则说明结构化融合具备轻量化价值。若 AUC 提升，需要结合四个场景的 gate 和深度混合系数判断是否确实利用了教师组关系；单个场景的提升不能单独作为结论。若下降，将分别检查 gate 和深度图残差是否造成正常信息过度耦合。

## 限制

本实验只有一个训练 seed，不能证明跨种子稳定性；DGT 的结构假设只作用于教师深度组，不提供 AUC 的直接代理损失；参数量不含冻结教师、行为分支和 KNN 数据库。
