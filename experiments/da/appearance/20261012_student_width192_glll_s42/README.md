# Global-Local 学生内部维度 192 实验

## 研究问题

在已完成的 [768 维 Global-Local 实验](../20261012_structural_mixer_glll_s42/README.md)上，**仅将学生内部宽度从 768 改为 192**，观察七场景逐帧 AUC 的损失是否可以接受，以及可训练参数下降多少。192 取自本机 MAD-ViT 复现的 `deit_tiny_distilled_patch16_224` 学生宽度；本实验仍使用 UAV 自己的模型和评测协议，不把两种架构视为相同。

## 唯一变化与必要的形状适配

- 学生 `proj_in` 输出、四个 block 和 gated FFN 的内部维度从 768 改为 192；Global-MHSA、三个 Local-DWConv、8 attention heads、FFN ratio=2 和 stage 次序不变。
- 教师 12 层 patch 特征仍为 768 维。12 个重建头从 `768→768` 相应变为 `192→768`，使每个头的输出形状和原有教师 target、loss、异常分数完全一致。这是内部维度改变所必需的投影适配。
- OCBE 保持 12 taps、768 维教师输入、64 维 code；每相邻三层教师对应一层学生的反向对齐不变；无 adapter。

## 固定训练与评测

- Drone-Anomaly 七场景各自独立训练，无跨场景混权重；seed=42；正常帧训练，15 epoch，batch=64，图像 256×256。
- 冻结 DINOv3 ViT-B/16；AdamW，OneCycleLR，max_lr=5e-4，weight_decay=0.05；group-balanced top-5 cosine loss，alpha=0.5。
- 测试对 12 个重建头的 top-5 patch cosine distance 取均值。每场景拼接测试帧求 ROC AUC×100，七场景做宏平均。与 768 维实验逐帧键、标签和数据指纹配对比较。
- 动态调度器先探测 GPU 0、1 的空闲显存；每卡先运行一个场景，结束后立即取队列中的下一个，轮询 2 秒。各场景日志及分数留在本目录。

## 预运行参数核算

使用实际 PyTorch 模块计数：768 维版学生 25,477,632，192 维版学生 2,947,968；OCBE 两版均为 4,770,368。可训练总量从 **30,248,000** 降至 **7,718,336**，减少 **22,529,664（74.48%）**。不含冻结教师、YOLO 行为分支和 KNN 数据库。输出形状均为 12×`[batch,256,768]`。

## 限制

单种子只检验这一个宽度点，不能确定最优宽度或方差。与 MAD-ViT 的参数量比较需采用一致的模块边界；这里报告的是 UAV 外观支路的 OCBE+学生可训练参数。完成前 AUC 不作推断。
