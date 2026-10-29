# Group-Context Fusion OCBE，seed42

## 研究问题

Rank2 Q 后四组特征仍采用 concat→Linear→GELU→Linear。这个实验用显式的组内压缩和四组全局上下文乘性交互替换右侧融合，判断能否在较少参数下保留或提升 AUC。

## 唯一变化

Q、12 个教师 tap、三层一组、学生 GL192、group-balanced top-k loss 和训练配置均与 Rank2 30 epoch baseline 相同。右侧改为共享 `1536→64` 投影，四组均值上下文 `tanh(mean)`，零初始化 64 维残差门控，以及 `256→128→64` 输出 MLP。没有新增 loss。

## 公平性

seed42、七场景独立训练、batch64、AdamW、OneCycleLR、30 epoch、两张卡动态调度。对比对象为 `20261029_rank2_baseline30_rerun1_s42`，不是历史 15 epoch 结果。

## 限制

单种子；AUC 与训练蒸馏 loss 不完全对齐；GPU 时钟未固定；推理测速不含教师和预处理。
