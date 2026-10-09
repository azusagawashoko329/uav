# 20260928_agg_explore｜相邻三层聚合探索

## 问题与范围

比较 DINOv3 教师的 12 层目标、四个代表层目标与相邻三层聚合目标，对 4 层学生的影响。本目录是历史探索记录；没有修改正式训练管线。数据集为 Drone-Anomaly，评测为每场景合并所有 test 子视频后的 frame-level AUC×100。

## 状态

**部分完成，不能引用为完整七场景对照。** `raw_raw12.txt` 只有 farmland、crossroads、highway 三场景；`logs/` 另有 raw12 和 raw12_group_balanced 的七场景训练日志，`loss_compare.csv` 记录二者七场景数值。这是训练损失的探索，不等于原计划的 base12/rep4_ocbe4/mean4_ocbe4 三变体对照。原 README 的待填结果表不能视为完成记录。

## 可核对结果

`loss_compare.csv` 中 raw12_mean 平均 AUC 84.66、raw12_group_balanced 平均 85.01、差值 +0.35；这是 seed 42 单次结果。逐场景分数和差值以 CSV 为准，训练过程见 `logs/<variant>_<scene>.log`。`raw_raw12.txt` 仅保留当时启动器写出的三个场景及起止时间，不能用它计算七场景均值。

## 来源与复现

- 脚本：`scripts/probe_e2e.py`、`scripts/run_full_scenes.sh`、`scripts/summarize.py`；其它早期探针见 `scripts/README.md`。
- 本实验脚本与原始记录在本次整理前均未进入 Git；历史运行时的**精确代码提交未知**。本次提交只为其建立保存点，不能将该保存点冒充运行时版本。
- 示例：`PYTHON=/path/to/python bash experiments/20260928_agg_explore/scripts/run_full_scenes.sh raw12 --seed 42 --loss_mode group_balanced`。这是重跑命令，不代表已按该命令重跑验证。
- 原始数据根目录：`/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly`。脚本含这一历史绝对路径；换机器时须显式配置或修正路径，并记录修改后的 commit。
- 权重未落盘；日志和 CSV 是当前可核对产物。没有多种子统计；探针与正式管线的绝对分数不可直接互换。

状态/来源的通用字段见 [`../../docs/EXPERIMENT_PROTOCOL.md`](../../docs/EXPERIMENT_PROTOCOL.md)。
