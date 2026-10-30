# Group-Block Low-Rank Fusion OCBE，seed42

## 研究问题

在保持 Rank2 Q、学生和蒸馏损失不变的前提下，把原始四组 concat→Linear→GELU→Linear 的第一层改成组块低秩分解，观察能否减少参数而保留异常检测 gap。

## 结构

每组输入 `x_g∈R^1536` 先通过共享 `V:1536→128`，再通过组特定 `U_g:128→128`，四组结果在 GELU 前求和：`h=Σ_g U_g V x_g+b`，最后 `GELU→Linear(128→64)`。没有均值上下文、gate、hop、attention 或额外 loss。

## 公平性

seed42、七场景独立训练、batch64、AdamW、OneCycleLR、30 epoch、两张卡动态调度。唯一结构变量是 OCBE 右侧融合；对比对象为 Rank2 30 epoch baseline。

## 限制

单种子；AUC 与训练蒸馏 loss 不完全对齐；rank128 共享输入投影可能丢失组特定通道方向；推理测速不含教师和预处理。
