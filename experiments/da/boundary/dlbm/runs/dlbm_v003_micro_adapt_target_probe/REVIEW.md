# DLBM_v003_micro_adapt_target_probe — adapted target failed

日期：2026-07-14  
协议：seed42 / 每场景独立 / 复用 Probe10 / coarse suffix L0、L4、L8、L11 / normal-only 微调 5 epoch

## 目的

v002 在冻结 Probe 上使用 exact-suffix Normal/Pseudo pairwise 风险，七场景 Oracle 集中在 L5/L6。
本实验只检验一种可能解释：这种集中是否只是因为各 suffix 未经过真实训练。它不训练 DLBM，也不
运行 Final。

## 方法

- 从同一场景的 all-12 Probe10 权重，严格切片得到四个 fixed-suffix OCBE/Student。
- 每个 suffix 独立使用正常训练数据微调 5 epoch；优化器和 Probe 超参数保持不变。
- 在固定 held-out 正常 subvideo 和固定 within-frame pseudo 上，于 epoch 0、1、3、5 测量
  `mean softplus(score_normal - score_pseudo)`。
- 人工 Gate 粗边界只用于实验结束后的局外 sanity check，不进入训练、选模或损失。

## 结果

| scene | epoch0 | epoch1 | epoch3 | epoch5 | 外部粗边界 | ep5 匹配 |
|---|---:|---:|---:|---:|---:|:---:|
| bike | L4 | L4 | L8 | L8 | L8 | yes |
| crossroads | L4 | L8 | L8 | L8 | L4 | no |
| farmland | L4 | L8 | L8 | L8 | L8 | yes |
| highway | L4 | L8 | L8 | L8 | L0 | no |
| railway | L4 | L8 | L8 | L8 | L8 | yes |
| solar | L4 | L4 | L4 | L4 | L8 | no |
| vehicle | L4 | L4 | L8 | L8 | L8 | yes |

5 epoch 后 6/7 场景选择 L8。尤其 highway 与 crossroads 虽然外部粗边界不同，却与另外四个场景
一起收敛到 L8；solar 则始终为 L4。训练更久确实改变了 epoch0 的选择，但变化方向主要是统一向
深层边界收缩，而不是形成符合场景的区分。

## 结论

1. v002 的失败不能归因于 DLBM epoch 太少，也不能归因于 suffix 完全没有适应训练。
2. 直接最小化 Normal/Pseudo 分数的 pairwise 风险，本质上仍在奖励两支的绝对可分性；它会继承
   ViT 层间尺度/可分性偏置，而没有刻画“打开下一段蒸馏后”的场景增量价值。
3. 因而本 target 标记失败，不据此训练 predictor，也不进入 Final。
4. 下一版应保留 exact suffix supernet 和 top-k patch 距离输入，但学习信号改为两支各自内部的
   opening dynamics：比较同一支在 base/opened suffix 下的响应变化，再让网络学习这些变化是否
   构成增益；不能继续使用单个手工 Normal-Pseudo delta 作为答案。
