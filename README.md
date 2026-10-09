# UAV 异常检测研究仓库

本仓库保留外观分支、行为分支、融合及后续蒸馏层边界探索的历史。它是研究过程记录，不把多个分支或不同数据集的指标混作同一正式模型。接手顺序：阅读 [AGENTS.md](AGENTS.md) → [实验协议](docs/EXPERIMENT_PROTOCOL.md) → [数据集说明](docs/DATASETS.md) → [历史实验索引](docs/LEGACY_EXPERIMENTS.md) → 对应实验 `RECORD.md` 和原 README。

- 复用模型代码：`model/`；正式入口：`train.py`、`eval.py`；行为/融合：`behavior/`、`fusion/`。
- 新实验：`experiments/<规范 ID>/`，实验专用脚本放各自 `scripts/`。历史目录保留原名，并以 `RECORD.md` 补足审计信息。
- 文献与开发历史：`note/`；输出来源：[`docs/ARTIFACT_PROVENANCE.md`](docs/ARTIFACT_PROVENANCE.md)。
- Git 提交及同步：[`docs/GIT_WORKFLOW.md`](docs/GIT_WORKFLOW.md)。大权重、视频、缓存不会由 Git 保存；实验报告、配置、可阅读指标和日志应进入 Git。

当前分支的 `experiments/20260928_agg_explore/` 是未完成的聚合探索，不能当作正式模型结论。`master` 与 `explore/distill-layer-boundary` 是独立演进线，详见历史索引。旧笔记中出现的 `DABG/` 指针指向此仓库之外的历史目录，当前仓库无法仅凭该相对路径复现；请使用 Git 提交与实验记录定位。
