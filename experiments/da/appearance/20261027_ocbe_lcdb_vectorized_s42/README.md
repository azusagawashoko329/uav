# Vectorized-DGT与LCDB：重新设计OCBE右侧融合

## 研究问题

DGT在Rank2之后降低参数量，但相对GL192-Rank2的速度和Latency显著恶化，AUC也没有提升。本实验先区分实现调度问题与结构信息损失问题，再决定是否进行七场景训练。

## 数学判断

DGT对每个深度组独立执行两个`1536→64` Linear、GELU、tanh和逐元素门控，四组通过Python循环调用八个小Linear；随后创建`[B,P,4,64]`临时张量并执行一跳/二跳`einsum`。这些操作的FLOPs不大，但kernel启动、低占用小GEMM、临时读写和逐元素kernel会主导batch=1延迟。DGT还在每组独立压缩到64维后才进行组间融合，跨组互补信息可能在高维阶段已经丢失。

## 两个方向

1. `Vectorized-DGT`：参数、深度图和门控数学不变，只将四组展平为一次value Linear和一次gate Linear，保持组轴后再进行原图传播。它用于判断DGT是否只是实现低效，不作为新的AUC模型。
2. `LCDB`（Low-Rank Cross-Depth Bottleneck）：四组`2×768`先用固定的Mean＋Trend深度基压成两个`1536`模式，再执行一次`3072→128→64`统一瓶颈。它在高维阶段保留跨组线性混合，移除gate、tanh、1-hop/2-hop和多组小Linear，不增加新loss。

Mean＋Trend基为
\[
P=\begin{bmatrix}1/2&1/2&1/2&1/2\\-3/\sqrt{20}&-1/\sqrt{20}&1/\sqrt{20}&3/\sqrt{20}\end{bmatrix}.
\]
固定P避免再次引入可学习深度方向、正交化和额外loss，先单独评价结构本身。

## 执行顺序

先进行七场景、batch1/16的GPU合成输入速度审计，比较Dense128、Rank2、原DGT、Vectorized-DGT和LCDB；检查原始/缓存Q/融合Q的code、student重建和异常分数。只有Vectorized-DGT确实减少延迟且LCDB没有出现DGT式kernel爆炸，才运行LCDB七场景、seed42、15 epoch训练。若LCDB速度或AUC没有实质改善，停止该方向，转向单一共享投影的更激进轻量化。

训练固定：共享随机正交可学习Q、相邻三教师对一学生、GL192学生、group-balanced top-k loss、七场景独立训练。训练权重保存到本目录并登记SHA；速度审计不产生AUC结论。

## 限制

速度审计的输入是GPU常驻合成教师特征，不包含教师前向、预处理和H2D；不同GPU绝对值不直接比较。原DGT完整checkpoint不可用，速度审计只恢复七场景最终Q并使用配对初始化。历史AUC只用于后续训练结果对照，不由速度审计补造。
