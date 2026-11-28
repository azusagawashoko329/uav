# HAP-RD-Lite-Q4 / DA / seed42

## 研究问题

检验HAP-RD-Lite的PODP是否应为四个相邻三层teacher组分别学习Q。共享Q只有一个2×3矩阵，四组独立Q为4×2×3矩阵，额外增加18个原始标量参数。该实验只回答投影共享约束的影响，不改变student、PDFB hidden、teacher taps、训练轮数、损失或评测。

## 模型与约束

- 基线为正式publication入口的HAP-RD-Lite，student为LLLG+gate、width=192，PDFB hidden=128，code_dim=64，teacher taps=0..11。
- 本变体使用`GroupLearnableDepthRank2OCBE`，四个三层组各有独立可学习2×3 Q。每个Q以seed=42驱动的随机正交基初始化。
- 前向对每个Q做可微Gram-Schmidt，保持两行正交。四个Q均作用于对应相邻三层，不跨组混合。
- 可训练参数量为3,761,240，较共享Q基线3,761,222增加18。teacher冻结。

## 训练与评测

DA七场景各自从头训练30 epoch，输入256×256，batch64，workers8，AdamW(max_lr=5e-4, weight_decay=0.05)+OneCycleLR，确定性cuDNN和drop_last。双卡调度器维护全局队列，每卡一个场景进程，任务完成后立即补位，不等待外部占用。

每个教师tap先独立LayerNorm。每组三层经对应Q压成两个深度模式，四组共8个模式进入同一个PDFB。重建仍为12个独立teacher tap head，训练和测试均使用`publication_match`，逐patch余弦距离取top-5，再按四组三层的alpha=0.5组平衡聚合。测试把场景所有视频帧拼接后计算AUC和离散ROC EER，最后做七场景宏平均。

## 运行状态

状态为 `completed`。三种子均已通过 per-group Q 的 forward/backward、独立初始化、更新后正交性和测试入口核验，并以记录中的精确 `source_commit` 完成脱离会话训练。日志、JSON、CSV和报告留在本目录，权重仅保存在被忽略的 `ckpt/` 并在 `artifacts.json` 登记。三种子配对比较与历史 shared-Q 边界见 [Q定义迁移说明](../../../../docs/HAP_RD_LITE_Q_DEFINITION.md)。

## 限制

该实验是DA三seed、七场景的结构消融。它不能单独分解PODP本身相对无PODP的影响，也不比较不同Q秩或不同正交化方法。
