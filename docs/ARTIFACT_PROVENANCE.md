# 产物来源与本地资产

[本地二进制资产清单](LOCAL_ARTIFACTS.csv)逐项列路径、大小、Git 状态和可关联的生成入口；用 `python scripts/inventory_artifacts.py` 更新。它只反映本机磁盘，不是云端备份。`sha256` 空白表示历史资产尚未逐文件校验；`source_commit: null` 表示无法核实运行时源码。

| 实验目录内产物 | 对应脚本/来源 | 证据与限制 |
|---|---|---|
| `da/appearance/20260611_rd_baseline/visualizations/*.mp4` | 同实验 `scripts/visualize_appearance.py` | 脚本默认加载该实验 `ckpt/`；历史视频实际生成时的权重哈希和命令未知。 |
| `da/behavior/detection_preview_legacy/artifacts/*.mp4` | 同实验 `scripts/detect_vis.py`，共享 `behavior/detector.py` | `artifacts/bike_render.log` 可核对 bike；其它视频生成命令未知。 |
| `da/audit/20260619_leakage_audit/figs/*.png` | 同实验 `scripts/cloud_dist.py` | 小图与 `scores/*.npz` 在 Git；原始运行命令未知。 |
| `da/fusion/20260617_all_model/results_fusion.csv` 与 `ckpt/<scene>/*.npz` | `fusion/train_fusion.py`、`fusion/eval_fusion.py`；管线见 `fusion/README.md` | 当时训练/评测权重哈希未登记；不能把当前脚本提交当作运行时提交。 |
| `da/fusion/legacy_score_dump/scores/*.npz` | `fusion/dump_appearance_scores.py`、`fusion/dump_behavior_scores.py` | 历史分数缓存，当前保留在该实验组；生成命令未知。 |
| `da/aggregation/20260928_agg_explore/logs/*.log` 与 `loss_compare.csv` | 同实验 `scripts/probe_e2e.py`、`scripts/run_full_scenes.sh`；汇总步骤未完整记录 | 七场景损失对照与三场景 raw 文件不可混作同一运行。 |
| `da/boundary/dlbm/runs/<version>/` 的 `summary.json`、`REVIEW.md`、逐场景结果 | 同版本 `scripts/` 中的历史启动器、`source_snapshot_explore_1a68d0c/experiment_DLBM/` | 各版本保留 `COMPLETE`/评审失败标记；精确运行时 commit 不明。 |
| 各实验的 `ckpt/*.pth`、`*.pt` | 对应实验的 `record.json` 指向脚本或历史源码快照 | Git 忽略本体；多数旧运行缺哈希和外部备份。 |

新实验必须在自身 `artifacts.json` 逐文件登记 `producer_script`、`run_id`、输入、状态、大小、SHA-256 与保存位置。大视频和权重不提交 Git，不能用“已忽略”代替保存和来源记录。
