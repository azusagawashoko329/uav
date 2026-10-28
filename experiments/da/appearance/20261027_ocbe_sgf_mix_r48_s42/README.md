# SGF-48 低维组间交互，seed42

## 研究问题

Cross-Depth 在 `B×P×4×2×768` 高维张量上做跨组非线性混合，训练反向需要保存大规模中间激活。本实验验证把计算顺序改为“每组三层 Rank2 Q → 共享组内通道压缩 →（可选）低维组间交互”后，能否在保留 AUC 的同时降低参数、FLOPs 和训练开销。

## 结构与唯一变量

输入仍是四组三层教师特征。每组经过共享可学习随机正交 `Q∈R^(2×3)`，得到 `2×768`，展平为 `1536`，再由同一个 `Linear(1536,48)+GELU` 编码为 48 维。四组编码保留为 `4×48`，最后使用原来的 `Linear(192,128)+GELU+Linear(128,64)` 输出 OCBE code。

本目录的变体为 **sgf_mix**：

- `sgf_base`：不增加组间算子，用作共享组内投影控制组；四组在最终低维 head 中联合融合。
- `sgf_mix`：在 `4×48` 上增加 depthwise group convolution，沿四个有序教师组使用长度 3 的局部核。每个 latent channel 独立拥有一个 `[left, center, right]` 核，初始化为 `[0,1,0]`，因此初始是恒等映射，训练中学习相邻组交互。

Q、三层教师对应一个学生 block、GL192 学生、group-balanced top-k loss、15 epoch、seed42 和数据划分全部不变；不增加辅助损失。

## 数学动机

原融合矩阵可视作被约束为

`W_dense ≈ W_h · M_group · (I_4 ⊗ W_local)`，

其中共享 `W_local:1536→48` 先完成通道压缩，`M_group` 只在小的四组轴交互，`W_h` 完成最终跨组融合。这样组交互不再作用于 `2×768` 的高维特征，反向图也不需要保存 Cross-Depth 的两个 403 MB 级中间张量。

## 实验设置

七个 Drone-Anomaly 场景独立训练和测试：`bike, crossroads, farmland, highway, railway, solar, vehicle`。两张 RTX 5880 Ada 由 `scripts/dynamic_scheduler.py` 动态复用；权重只保存在本实验目录并由 Git 忽略。评测记录每场景 AUC、训练时间、参数量、检查点哈希；完成后再对比 Rank2 Learnable Q、LCDB 和 Cross-Depth。

## 预期与限制

预期 SGF-48 的融合参数和计算量低于 LCDB；AUC 是否保持需要七场景实测。单一种子不能证明稳定性，GPU 时钟和场景运行顺序也会影响绝对训练时间。

## 最终结果


七场景已完成，mean AUC 为 **81.6683**，可训练参数为 **3,073,286**。

- Rank2 Learnable Q mean AUC：82.9599；LCDB：82.8272；Cross-Depth：82.8661。
- 本变体相对 Rank2：-1.2916 pp；相对 LCDB：-1.1589 pp。
- 训练过程无失败；各场景 checkpoint 已生成并在 `artifacts.json` 登记。
- 训练权重速度结果见 [`speed_summary.md`](results/speed_summary.md)。

SGF-mix 没有修复 SGF-base 的信息损失，mean AUC 进一步下降，低维 depthwise 组间交互暂不作为后续主线。
