# DLBM_v006_paired_rollout_relation — target and signal-capture review

日期：2026-07-14  
协议：seed42 / 七场景独立 / 复用 Probe10 / 8 rollout episodes / 20 inner RD steps /
DLBM 50 epoch / 不训练 Final

## 实现边界

- 输入保留每层 top-5 patch 的距离、空间位置和 top-k 内次序，没有先平均成层级标量。
- 候选层与已选深层 suffix 通过共享 cross-attention 建模，Normal/Pseudo 完全共享编码器。
- 不包含人工 Gate 的 D/M/S 分组、rank/flip/tail/z-score、阈值或 coarse 标签。
- 唯一目标是 adjacent open 相对 stop 的 difference-in-differences：两个临时 suffix learner
  从各自 Probe10 切片初始化，使用同一 support 更新序列，只比较共同深层 anchor 上的
  hold-out normal 学习进步。
- DLBM 直接以 MSE 回归实测连续 gain；边界由 adjacent gain 的连续累积最大值产生。
- 人工 Gate 粗边界仅在全部训练结束后加入汇总表。

## 七场景边界

| scene | v006 | validation rollout oracle | all-8 rollout oracle | 外部人工 Gate 粗边界 |
|---|---:|---:|---:|---:|
| bike | L10 | L10 | L10 | L8 |
| crossroads | L11 | L11 | L11 | L4 |
| farmland | L10 | L10 | L10 | L8 |
| highway | L11 | L11 | L11 | L0 |
| railway | L11 | L11 | L11 | L8 |
| solar | L11 | L10 | L10 | L8 |
| vehicle | L11 | L11 | L11 | L8 |

最终只有 L10/L11 两种边界，与外部粗边界 0/7 精确匹配。50 epoch 已足以排除单纯训练轮数不足：
除 solar 外，各场景在 epoch20 后已经稳定在其最终或相邻深层边界。

## DLBM 是否真正使用配对输入

| scene | validation MSE | candidate-constant MSE | model / constant | paired-input sensitivity |
|---|---:|---:|---:|---:|
| bike | 9.107e-5 | 8.655e-5 | 1.052 | 3.378e-4 |
| crossroads | 4.167e-4 | 4.079e-4 | 1.022 | 3.866e-4 |
| farmland | 1.048e-3 | 1.033e-3 | 1.014 | 3.401e-4 |
| highway | 2.225e-3 | 2.222e-3 | 1.001 | 1.856e-4 |
| railway | 6.212e-4 | 4.841e-4 | 1.283 | 2.536e-4 |
| solar | 2.752e-3 | 2.144e-3 | 1.284 | 3.173e-3 |
| vehicle | 4.348e-4 | 4.294e-4 | 1.012 | 1.110e-4 |

七场景的网络 MSE 全部不优于“每个候选层恒定输出训练 episode 平均 gain”的基线。破坏候选层与
base 的同帧配对后，输出虽非严格不变，但变化通常远小于预测 gain 和 episode target 波动。因此，
v006 的最终边界主要来自候选层的平均 gain 曲线，不能证明关系网络学会了人工 Gate 所揭示的
帧级/patch 级增量结构。

## 失败机制

1. difference-in-differences 成功消除了 v005 的绝对外层损失偏置，gain 量级达到 1e-2 到 1e-1，
   不再是不可分辨的一步微小量。
2. 但“加入层后是否改善已有深层的正常重建”只衡量辅助蒸馏对深层优化的正迁移。它不直接衡量
   人工 Gate 有效的核心现象：同一批帧在 base/open 后的困难 patch 分布、相对排序和尾部结构是否
   发生稳定且非空的迁移。
3. 该目标因此普遍认为深层已经充分，七场景 oracle 自身就塌缩到 L10/L11。即使 predictor 完美
   拟合，也无法恢复 highway L0、crossroads L4 或其余场景 L8 的粗边界。
4. predictor 只有 6 个训练 episode，确实可能限制关系学习；但增加 episode 只能改善 predictor
   对当前 target 的拟合，不能修复 target oracle 的深层塌缩。

## 决策

- v006 保留为结构和目标分离的失败实验；不运行 Final，不根据外部 Gate 调阈值。
- 保留 patch token encoder、candidate/base cross-relation、跨帧 Set aggregation、精确 suffix
  rollout 和共同-anchor difference-in-differences 作为可复用组件。
- 下一版本不应继续扩大 normal-anchor rollout 或增加规则，而应首先让网络直接学习
  `(base, opened)` 配对响应相对于 frame/layer-shuffled null 的结构化迁移，再检验这种 learned
  transition statistic 是否在七场景产生与人工 Gate 同方向的粗边界差异。人工 Gate 仍只做局外
  检验，不作为训练标签。

## 工程审查

- 36 项全量单元测试通过；无 NaN、Inf、OOM。
- 七场景峰值显存均约 2.72 GiB。
- 七个 Probe10 checkpoint 的训练前后 SHA-256 完全一致。
- smoke 临时目录已删除；正式目录包含配置、逐场景日志、checkpoint、result、summary 与本审查。
- 未执行 Git commit、push 或 merge。

