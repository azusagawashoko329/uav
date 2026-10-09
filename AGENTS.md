# UAV 项目交接规则

1. 先读 [实验导航](experiments/INDEX.md)、[实验协议](docs/EXPERIMENT_PROTOCOL.md)及目标实验的 `RECORD.md`/`record.json`；检查 `git status -sb`、`git log --all --graph --oneline -20`。不能凭日期或分支名称推断正式模型。
2. 新实验只建在 `experiments/<dataset>/<family>/<YYYYMMDD>_<question>_<variant>_s<seed>/`。dataset 仅 `da`、`uit` 或跨数据集 `da_uit`。family 采用现有方向名。每个实验先提交 README、record/config、`scripts/`，得到精确 `source_commit` 后再运行。单实验脚本放自己的 `scripts/`；共享模型/工具仍分别在 `model/`、`behavior/`、`fusion/`、仓库级 `scripts/`。
3. 运行日志、小型 JSON/CSV、报告和状态文件留在实验目录并提交 Git。权重、视频和大型缓存留在实验目录但由 `.gitignore` 排除；在 `artifacts.json` 登记生成脚本、run_id、路径、状态、大小和哈希。失败、中止、评审未通过要客观记录，不能丢弃。
4. 不覆盖旧实验。改参数、脚本或种子就建新实验 ID；提交按源码计划、运行、评测、复盘分阶段。旧实验的 `source_commit: null` 表示无法证实，不能把本次迁移提交冒充原运行版本。历史分支源码快照是资料，不是当前模型入口。
5. 变更路径时更新 [迁移表](docs/MIGRATION_MAP.csv)、[实验导航](experiments/INDEX.md)和产物索引，并检查代码里的路径引用。运行 `python scripts/check_experiments.py`、`git diff --check`。完成后推送分支并比对本地和远端完整 SHA；推送失败必须说明。
6. 规范以 [实验协议](docs/EXPERIMENT_PROTOCOL.md) 为唯一详细标准。本文件是简明入口；不另装与其冲突的项目 skill。
