# UAV 异常检测研究仓库

从 [实验导航](experiments/INDEX.md) 开始查看研究记录：每个实验位于 `experiments/<数据集>/<方向>/<实验 ID>/`，目录中有 `record.json`、可阅读记录、脚本和证据。旧路径与新路径逐项列在 [迁移表](docs/MIGRATION_MAP.csv)。本次重排只移动和归档实验资产，没有开发模型或删除实验。

| 位置 | 内容 |
|---|---|
| `model/`、`train.py`、`eval.py` | 当前分支的共享外观模型与训练评测入口 |
| `behavior/`、`fusion/` | 当前分支的共享行为与融合代码 |
| `experiments/da/appearance/` | Drone-Anomaly 外观实验、消融、code_dim 与 LAA 历史 |
| `experiments/da/behavior/` | 运动探针与检测预览 |
| `experiments/da/fusion/` | 融合实验与历史逐帧分数 |
| `experiments/da/aggregation/` | 教师层聚合探索 |
| `experiments/da/boundary/` | DLBM 与 mask DLBM 各版本运行；历史源码快照单独存放 |
| `experiments/da/gating/` | 门控及端到端门控历史 |
| `experiments/da/audit/` | 审计代码、分数和图 |
| `experiments/uit/` | UIT-ADrone 实验入口；本仓库尚无可核实的本地 UIT 实验目录 |

接手规范见 [AGENTS.md](AGENTS.md)、[实验协议](docs/EXPERIMENT_PROTOCOL.md)、[数据集约定](docs/DATASETS.md)和 [Git 同步规范](docs/GIT_WORKFLOW.md)。[本地二进制资产索引](docs/LOCAL_ARTIFACTS.csv)仅是磁盘快照；权重和视频不由 Git 备份。历史笔记在 `note/`，但其中部分 `DABG/` 路径属于仓库外资料，不能直接作为本仓库的源码指针。

`master`、`research/next` 与 `explore/distill-layer-boundary` 是不同模型演进线。边界源码快照来自 `explore` 的 `1a68d0c`，LAA 快照来自 `master` 的 `fd6cce0`；快照提交不等于每次历史运行的精确源码提交。详情见 [历史索引](docs/LEGACY_EXPERIMENTS.md)。
