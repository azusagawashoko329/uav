# DLBM_v004_mask_localization_target_probe — spatial target failed

日期：2026-07-14  
协议：seed42 / 每场景独立 / Probe10 复用 / coarse suffix L0、L4、L8、L11 / suffix 微调 5 epoch /
共享 patch localizer 20 epoch（milestone 5、10、20）

## 目的与实现审查

本版本检验“不手写 Normal-Pseudo 标量增益，改由网络从 patch 误差图学习增益表征”是否足够。

- 四个候选 suffix 都从同一 Probe10 严格切片并独立 normal-only 微调。
- 同一场景所有 suffix 使用完全相同的 support/query 划分和 CutPaste 随机种子。
- 一个跨 suffix 共享的轻量 Transformer 对每个空间 patch 的有效层序列编码；无三段串联 encoder。
- 唯一训练目标是已知 CutPaste 空间支持的 soft patch BCE；不包含人工 Gate、层阈值、排序翻转公式
  或 Normal-Pseudo score delta。
- held-out patch AUC 只作 target 选择；外部粗 Gate 只在结果落盘后比较。
- 25 项工程单测通过；正式七场景无 NaN/Inf/OOM，峰值约 3.67 GiB。

## 七场景结果

| scene | ep5 | ep10 | ep20/final | 外部粗边界 | final 匹配 |
|---|---:|---:|---:|---:|:---:|
| bike | L0 | L0 | L0 | L8 | no |
| crossroads | L0 | L0 | L0 | L4 | no |
| farmland | L4 | L0 | L4 | L8 | no |
| highway | L0 | L0 | L0 | L0 | yes |
| railway | L4 | L4 | L0 | L8 | no |
| solar | L0 | L0 | L0 | L8 | no |
| vehicle | L0 | L8 | L0 | L8 | no |

20 epoch 最终六场景为 L0、一场景为 L4，仅 1/7 与外部粗边界一致。多数场景在 ep5 已经偏向
L0，继续训练并未恢复目标区分度；vehicle 的 ep10=L8 随后回到 L0，也不能作为稳定结果。

## 失败机制

定位器确实使用了 patch/layer 输入，但它回答的是“哪个 suffix 的误差图最容易定位局部 CutPaste”。
浅层特征天然保留更多空间细节，所以这个绝对定位任务系统性偏爱 L0/L4。它没有回答“相对于当前
base suffix，打开下一段之后是否产生对异常泛化有用的增量变化”。这与 v002/v003 的深层偏置方向
相反，但同样属于目标定义偏置，而不是 epoch 不足。

作为交叉检查，v003 的 held-out normal top-k 绝对误差在七场景全部选择 L11，说明仅换成正常集
绝对重建质量也会产生相反的 trivial 解；不能用它替代 v004 target。

## 决策

v004 target 不训练 DLBM predictor，不进入 Final。保留的组件只有 exact suffix supernet、带 mask
的可重复 CutPaste 和共享 patch/layer encoder。下一目标必须以 `(base, opened)` 配对响应为基本输入，
只建模 opening dynamics；不能再以某个 suffix 的绝对 pseudo 定位或绝对 normal 重建表现直接选边界。
