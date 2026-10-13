# Seed 43：四 block Global-Local，学生宽度 192，dense128+depthrank2 OCBE（四组三层各保留均值/趋势，6144→128→64）

## 研究问题

在 seed=42 已完成的 [对应实验](../20261013_ocbe_depthrank2_hidden128_s42/README.md)基础上只改随机种子为 43，复核该结构的七场景 AUC 和参数量。本实验属于四个结构、两个种子的配对复核之一；与另外三个 seed43 实验共用双卡动态队列，但每场景独立初始化教师、OCBE、学生和优化器，训练权重不跨场景混用。

## 固定设置

- 冻结 DINOv3 ViT-B/16，12 个教师 patch tap；学生为四个 Global-Local block（1 global+3 local），内部宽度 192，每个学生 block 输出经三个独立的 768 维重建头与相邻三层教师反向对齐。
- OCBE：dense128+depthrank2 OCBE（四组三层各保留均值/趋势，6144→128→64）；每个教师 tap 单独仿射 LayerNorm，最终 code_dim=64。
- Drone-Anomaly 七场景分别以正常训练帧训练 15 epoch；seed=43，batch=64，256×256，AdamW，OneCycleLR，max_lr=5e-4，weight_decay=0.05。蒸馏为 group-balanced top-5 cosine loss，alpha=0.5。
- 推理时每帧分数为 12 个重建头的 top-5 patch cosine distance 平均，**越大越异常**；每个场景汇总所有测试视频的逐帧分数与配对 `.npy` 标签计算 ROC AUC×100，七场景取宏平均。与 seed42 对照核对帧键和划分指纹。
- 预计 OCBE＋学生可训练参数 3,761,216；不计冻结教师、行为分支和 KNN 库。

## 运行与限制

实验源码与配置先提交，`source_commit` 再指向精确源码提交。两张 GPU 各运行一个场景进程，结束后立即从 28 项共享队列领取下一项；每个场景写独立日志、逐帧分数和结果 JSON。完成后按日志、结果、复盘分阶段归档。seed43 是对 seed42 的重复验证；两颗种子仍不足以证明稳定增益，不能用正常特征能量或训练 loss 代替 AUC。
