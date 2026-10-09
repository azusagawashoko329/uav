# 20260928_agg_explore｜历史实验审计记录

- **审计状态**：存在结果文件；是否完整以原报告与日志为准。本条目只说明本次整理时磁盘所见，不替代原始实验报告。
- **数据集**：Drone-Anomaly（按目录与历史说明推定，未逐文件验证）。
- **运行时源码提交**：unknown。不能将历史文件首次进入 Git 的提交当作运行时源码快照。
- **原路径最早可见提交**：`未进入整理前的 Git 历史`。
- **关联脚本**：`scripts/probe_e2e.py`, `scripts/run_full_scenes.sh`, `scripts/summarize.py`。具体运行参数以日志或原配置为准。
- **结果/配置**：`loss_compare.csv`；配置：未找到独立 config.json。
- **日志**：`logs/raw12_bike.log`, `logs/raw12_crossroads.log`, `logs/raw12_farmland.log`, `logs/raw12_group_balanced_bike.log`, `logs/raw12_group_balanced_crossroads.log`, `logs/raw12_group_balanced_farmland.log`；另有 8 项。
- **权重**：0 个本地文件，Git 不保存；不能据权重存在推断评测已完成。
- **原始说明**：[README.md](README.md)。缺失字段保持 unknown，不补造历史事实。

新实验按 [实验协议](../../../../docs/EXPERIMENT_PROTOCOL.md) 建立完整 `record.json`、脚本和产物映射。本记录保留旧目录名及原结果，不改动原始指标。

## 迁移位置

原路径：`experiments/20260928_agg_explore`；当前路径：`experiments/da/aggregation/20260928_agg_explore`。脚本和产物位置以当前目录与 `record.json` 为准。
