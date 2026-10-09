# 历史实验与分支索引

本次重排将旧实验实体移动至 [实验导航](../experiments/INDEX.md) 所列的 `experiments/<数据集>/<方向>/<实验>/`；[迁移表](MIGRATION_MAP.csv)逐项保存旧路径。实验没有删除，历史目录名多数保留为实验 ID。旧文件内出现的原路径表示当时的运行环境；重跑应使用当前路径和实验脚本，不能直接照抄旧命令。

## 分支与源码

| 来源 | 归档位置 | 说明 |
|---|---|---|
| 当前 `research/next` | `model/`、`behavior/`、`fusion/` | 外观/行为/融合的当前分支代码；重排未合并其它模型版本。 |
| `explore/distill-layer-boundary@1a68d0c` | `experiments/da/boundary/source_snapshot_explore_1a68d0c/` | DLBM、mask DLBM、相关 model/test 文件的只读历史快照。各版本输出在 `boundary/dlbm/runs/` 与 `boundary/mask_dlbm/runs/`。快照提交不等于每个运行的精确源码提交。 |
| `master@fd6cce0` | `experiments/da/appearance/laa/source_snapshot_master_fd6cce0/` | LAA adapter、训练入口和笔记的历史快照；三组本地权重在 `appearance/laa/`。 |
| 整理前旧 `origin/main`/`behavior-baseline@25cffaf` | Git 历史 | 2026-06 行为分支基线；本地引用不证明新 GitHub 仓库已有该提交。 |
| `rd-baseline-20260611@c76ff07` | Git 历史 | RD 初始基线。 |

历史分支不线性包含彼此，不能把快照文件混入当前 `model/` 后声称是同一已验证模型。未来模型开发从明确的分支和源码提交开始，另立实验 ID。

## 证据边界

- `appearance/20260611_rd_baseline`、三组 RD 消融、教师消融、code_dim 等保留 CSV/日志；逐项见各自 `record.json`、`RECORD.md`。
- `aggregation/20260928_agg_explore` 的 `raw_raw12.txt` 仅三场景，`loss_compare.csv` 是另一项七场景损失对照；原计划的聚合三变体并未完整记录。
- `boundary/dlbm/runs/` 中每个版本独立成目录。`COMPLETE`、`FAILED_REVIEW` 等标记只说明运行/评审阶段，不表示方法被采纳。旧启动器副本在可定位版本的 `scripts/`，全量源码在上述快照。
- `gating/`、部分 2026-07 外观实验及 LAA 目录主要只有本地权重，缺运行时源码提交和完整评测结果，状态为 `unknown`。
- `experiments/uit/` 当前没有可核实的本地实验实体。`note/方法对比总表.md` 引用的 `DABG/` 与 UIT 路径属于本仓库外资料，不能伪装成当前 Git 树的结果。

“有 `.pth` 文件”不等于训练完成、评测成功或云端可还原。新实验的来源与状态字段按 [实验协议](EXPERIMENT_PROTOCOL.md) 填写。
