# DLBM_v005_gdas_bilevel_rank — optimizer valid, gain target failed

日期：2026-07-14  
协议：seed42 / 每场景独立 / Probe10 冻结复用 / coarse suffix L0、L4、L8、L11 /
50 epoch × 8 episode / hard Gumbel / exact one-step full-parameter hyper-gradient

## 方法审查

- 每个场景的 DLBM 只有四个 boundary logits，没有人工统计输入或额外差异 encoder。
- 每个 episode 从不变的 Probe10 参数出发，在 support normal 上执行一次完整 OCBE/Student virtual
  RD update；Probe 文件和常驻权重均不修改。
- query 只使用两个 within-frame pseudo 视图之间的面积排序 RankNet loss，不计算 Normal-Pseudo
  delta，不使用人工 Gate 标签、mask localizer 或显式 layer-gain 公式。
- exact suffix 搜索输出与已有 fixed-suffix 数值测试一致；30 项全量单测通过。
- 七场景无 NaN/Inf/OOM，峰值显存约 11.16–11.17 GiB。

## 50 epoch 边界

| scene | ep5 | ep10 | ep20 | ep30 | ep50 | final confidence | 外部粗边界 | 匹配 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| bike | L8 | L8 | L8 | L8 | L8 | 0.995 | L8 | yes |
| crossroads | L8 | L8 | L8 | L8 | L8 | 0.996 | L4 | no |
| farmland | L11 | L8 | L8 | L8 | L8 | 0.986 | L8 | yes |
| highway | L8 | L8 | L8 | L8 | L8 | 0.997 | L0 | no |
| railway | L8 | L8 | L8 | L8 | L8 | 0.998 | L8 | yes |
| solar | L4 | L4 | L4 | L4 | L4 | 0.999 | L8 | no |
| vehicle | L4 | L8 | L8 | L8 | L8 | 0.995 | L8 | yes |

最终为六个 L8、solar=L4，与外部粗边界匹配 4/7。延长训练使分布高置信收敛，没有产生新的场景
区分；结果与 v003 的真实 suffix 微调目标完全相同。

## 固定动作审查

完成训练后，使用另一组固定 32 episode，对四个边界逐一强制执行相同的一步 normal RD，并独立
计算 outer RankNet loss。

| scene | policy | pre-update oracle | post-update oracle | post 最优-次优间隔 | 所选边界一步改善 |
|---|---:|---:|---:|---:|---:|
| bike | L8 | L8 | L8 | 0.0018762 | +0.00000680 |
| crossroads | L8 | L8 | L8 | 0.0012131 | +0.00000403 |
| farmland | L8 | L8 | L8 | 0.0011413 | +0.00000892 |
| highway | L8 | L8 | L8 | 0.0016747 | +0.00000959 |
| railway | L8 | L8 | L8 | 0.0011444 | +0.00005777 |
| solar | L4 | L4 | L4 | 0.0013915 | -0.00000075 |
| vehicle | L8 | L8 | L8 | 0.0023592 | -0.00000039 |

策略与 post-update oracle 7/7 一致，排除了 Gumbel 采样自强化或实现选错目标。但 pre/post oracle
也 7/7 完全相同，且一步改善远小于候选原有静态差异。高置信只表示 logits 对微弱静态偏好充分
收敛，不表示 opening-gain 证据很强。

## 决策

保留 exact functional inner update、hard-Gumbel boundary distribution、固定动作审查和 Probe cache
基础设施。淘汰当前 pseudo-area RankNet target，不运行 Final。

下一版若继续此结构，最小且可审查的改动应是让 architecture logits 只优化训练前后的外层目标变化：

`Delta_k = L_outer(theta'_k, k) - L_outer(theta_0, k)`

而不是直接最小化 `L_outer(theta'_k, k)`。这会显式消去 Probe 初始静态边界偏好，只保留 inner
training 带来的增量；是否需要更长 inner horizon 必须先由独立 fixed-action Delta 审查决定。
