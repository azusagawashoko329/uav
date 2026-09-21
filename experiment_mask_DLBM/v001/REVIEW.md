# DLBM v001 内部审查

## 1. 审查结论

`v001_relation_changepoint` 已完成七个场景的二阶段训练、冻结决策和软件审查。实现满足以下工程约束：

- 每个场景单独训练一个 DLBM；
- 只复用对应场景的 Probe10 Teacher/OCBE/Student 权重；
- `boundary_train` 只用于二阶段拟合和内部验证；
- `boundary_decision` 只在模型冻结后用于场景级决策；
- train 与 decision 被视为同分布正常数据，代码没有计算或利用 train–decision 失配；
- 输出始终是合法连续后缀 `{L_k, ..., L_11}`；
- 权重、decision、日志和状态文件分别落在 `ckpt/`、`decision/`、`logs/` 和 `status/`。

软件功能审查通过，但本版没有得到可采信的蒸馏边界。七个场景的边界后验均接近 12 类均匀分布，argmax 只是极小数值差异的结果，不能直接交给 final training。

## 2. 实际网络

对第 `l` 层的同位置 patch，冻结的 Probe 网络提供：

\[
T_{l,p}\in\mathbb{R}^{768},\qquad
S_{l,p}\in\mathbb{R}^{768},\qquad
D_{l,p}=1-\cos(T_{l,p},S_{l,p})\in\mathbb{R}.
\]

因此曾经讨论的 `256 × (768 + 768 + 1)` 表示一层的 256 个 patch，每个 patch 拼接 Teacher 768 维、Student 768 维和一个显式 cosine distance 标量。v001 并没有直接把这个大张量送进 Transformer，而是：

1. 用共享投影分别把 Teacher 和 Student patch 压到 64 维；
2. 用 relation query 对二者做 patch 内 cross-attention；
3. 将显式 D-map 经过 3×3 卷积，并把学习到的 T/S relation 作为残差注入；
4. 用按 `L11 → L0` 顺序运行的 3×3 ConvGRU，在看见 `L_k` 真实输入前，由更深层状态预测 `L_k` 的完整 D-map；
5. 将预测前状态、真实观测、更新后状态、预测 D 和真实 D 编为该层的 256 个局部 transition token，再加入 1 个全局 transition token。

所以边界头实际接收的是每层 `257 × 64` 的 transition 表征，不是只读一个全局均值。局部 token 保留 patch 级失配和空间关系，全局 token 表示整层新增方向。

表示学习只使用两个等权平均项：

\[
\mathcal{L}_{repr}
=\frac{1}{12}\sum_k
\operatorname{SmoothL1}(\hat D_k,D_k)
+
\frac{1}{12}\sum_k
\operatorname{SmoothL1}(\hat Z_k,Z_k).
\]

第一项要求深层后缀预测被 mask 的下一层 D-map；第二项要求全局 transition token 保留该层 transition 目标。这里没有通过 decision 失配构造监督。

最终 `OrderedChangePointDensity` 为每个 transition token 学习两个对角高斯状态，并枚举 12 个合法边界：

\[
\log p(B=k\mid Z)
=
\sum_{l<k}\log p_{\mathrm{out}}(Z_l)
+
\sum_{l\ge k}\log p_{\mathrm{in}}(Z_l),
\quad k\in\{0,\ldots,11\}.
\]

这个结构保证输出一定是从 `L_k` 到 `L11` 的连续后缀，但不手工规定 D/M/S 分组或开启阈值。

## 3. 训练长度

表示网络使用：

- 最大 30 epoch；
- 最少 10 epoch；
- patience 7；
- 按 `boundary_train` 内、逐 subvideo 的固定 80/20 fit/validation 划分选择最佳权重。

六个场景的最佳表示权重出现在 epoch 30，highway 出现在 epoch 28。由此可确认 5–10 epoch 对当前网络偏短；训练到 30 epoch 没有出现内部验证损失反弹所代表的经典过拟合。

密度边界头最大 300 epoch、最少 50 epoch、patience 35，七个场景最佳点都在 epoch 300。结合下面的状态偏好分析，这不表示单纯训练不足：当前目标正在稳定学习“深度几何分段”，继续延长更可能强化同一分段，而不会自动获得人工 Gate 的开启语义。

## 4. 七场景 decision

均匀 12 类后验的单类概率为 `0.08333`，归一化熵为 `1.0`。

| 场景 | 数值 argmax | 数值选择层 | 最大概率 | 归一化熵 |
|---|---:|---|---:|---:|
| bike | L11 | L11 | 0.08758 | 0.99980 |
| crossroads | L11 | L11 | 0.09111 | 0.99926 |
| farmland | L11 | L11 | 0.08554 | 0.99993 |
| highway | L11 | L11 | 0.08864 | 0.99967 |
| railway | L5 | L5–L11 | 0.08537 | 0.99993 |
| solar | L7 | L7–L11 | 0.08509 | 0.99995 |
| vehicle | L11 | L11 | 0.08596 | 0.99990 |

这些只能称为“数值 argmax”，不能称为“有效边界”。尤其是 solar 的 L7 即使接近人工 Gate 的粗边界 L8，其后验仍近乎均匀，且四个 subvideo 的 argmax 为 `[L6, L7, L7, L11]`，不构成可靠证据。

## 5. 网络是否真的使用了底层信号

在冻结最佳表示网络后做了两种不参与训练的干预：

- 将每层 T/S 一起独立空间滚动，破坏跨层 patch 对齐，但保留该层 Teacher–Student 配对；
- 关闭学习到的 T/S relation 残差，只保留显式 D-map 支路。

| 场景 | 跨层错位损失 / 原损失 | 去 relation 损失 / 完整损失 |
|---|---:|---:|
| bike | 3.59× | 249.32× |
| crossroads | 4.69× | 166.21× |
| farmland | 4.68× | 149.16× |
| highway | 3.92× | 108.44× |
| railway | 1.77× | 36.23× |
| solar | 3.45× | 50.36× |
| vehicle | 3.44× | 124.81× |

因此不能把失败归因于输入完全不可分，或认为网络只学了 D-map 均值。表示网络明确依赖：

- 同位置 Teacher–Student patch 关系；
- 跨层空间对应；
- T/S 特征方向，而不只是已经压成标量的距离。

## 6. 失败位置

当前边界头假设“正确边界等价于 transition token 在深度轴上由一个高斯状态切换到另一个高斯状态”。实验表明这一假设不充分。

对 bike、crossroads、farmland、highway 和 vehicle，`L0–L10` 的 token 都被 `out` 状态给出略高似然，只有固定锚点 L11 留在 `in` 状态，所以数值 argmax 系统性落在 L11。railway 和 solar 分别在约 L5、L7 附近发生似然符号切换，但后验差异仍很小。

这说明密度头主要按 ViT 层深的自然几何变化进行二分，而没有学到“把 `L_k` 加入已有深层后缀后，关系结构发生了何种可用于 Gate 的增量变化”。连续后缀约束本身是正确的；错误是把任意两段可分性直接等同于蒸馏边界。

## 7. 是否进入 final

本版不应运行 final training。原因不是数值边界与人工粗标签不完全一致，而是边界后验无置信度，当前 argmax 对微小扰动敏感。此时运行 final 无法区分 DLBM 的真实作用和近均匀后验的随机选择。

下一版可以保留已经通过干预验证的 relation-assisted masked encoder、257-token transition 表征和连续后缀输出，但应替换被动双高斯密度头。新的目标需要直接对“当前深层后缀状态”和“加入下一层后的学习响应”进行可学习比较；允许在 loss 层定义顺序开启问题，但增量响应仍由网络从 T/S/D token 中学习，不能重新引入 D/M/S、手工阈值或 Gate 公式。

## 8. 与人工 Gate 底层信号的覆盖关系

人工 Gate 的 normal 分支先从每层完整 patch cosine distance map

\[
D_l(x,p)=1-\cos(T_l(x,p),S_l(x,p))
\]

取得 top-5 patch 均值，再在帧、层组和候选开启前后计算排序变化、翻转率、尾部响应等。
因此，对同一批帧和同一组冻结权重，v001 输入的 `T/S/D` 在信息论意义上严格包含
normal 分支的逐层 top-k 原始量：top-k 是完整 `D_l` 的确定函数。v001 还额外保留了
人工 Gate 没有使用的 T/S 方向与跨层空间关系。

但“输入包含”不等于“训练后的 token 已保留”，更不等于“边界头已经利用”。当前证据只能
支持以下分层结论：

| 层次 | 当前结论 | 依据 |
|---|---|---|
| 原始输入是否包含 normal Gate 的逐层 patch 信号 | 是 | 完整 D-map 可确定地恢复每层 top-k |
| 表示是否使用 T/S 方向和跨层 patch 对齐 | 是 | relation ablation 与 spatial-roll 干预显著恶化 |
| 表示是否保留 top-k 尾部和跨帧排序变化 | 尚未证明 | SmoothL1 对 256 patches 平均，未单独审计 top-5；当前密度又只读取一、二阶矩 |
| 是否等价于完整正式人工 Gate 信号 | 否 | 正式 Gate 还使用早期 probe epoch、pseudo-normal 支路、跨帧 rank/flip 和 null calibration；v001 只读 Probe10 正常帧 |
| 当前边界头是否学会人工 Gate 的开启语义 | 否 | 后验近均匀并系统性偏向 L11 |

尤其需要注意，top-5 只占 256 个 patch 的约 `1.95%`。v001 的整图 SmoothL1 目标和后续
Gaussian moment aggregation 都可能在优化上优先保留多数容易 patch，而把人工 Gate 最依赖的
尾部次序压弱。当前 density 还把 frame 和 patch 汇总为均值与二阶矩，无法恢复“同两帧在
开启前后的次序是否翻转”这种依赖样本对应关系的量。

所以前端目前应评价为“原始信息覆盖、部分关系能力已证实，但人工 Gate 关键充分统计量尚未
完成 readout 证明”，不能直接判定前面全部完成。下一步在改变网络前，应冻结 v001 encoder，
仅做不参与 DLBM 训练和决策的 diagnostic readout：

1. 从 transition token 回归每帧每层真实 top-5 score；
2. 用回归结果复原任意相邻后缀开启前后的跨帧排序变化；
3. 在 `boundary_train` 拟合 readout，在同分布的 `boundary_decision` 只评估泛化；
4. readout 标签只用于审计信号是否仍在 token 中，不用于训练边界或选择 final taps。

如果逐层 top-k 和排序变化可被简单线性/小型 readout 稳定恢复，说明 relation/mask 前端可以
保留，问题集中在 opening-response head；如果恢复失败，则应先把表示损失改为显式保留 RD
top-k 尾部或使用可学习的 tail-aware pooling，而不是继续替换边界头。
