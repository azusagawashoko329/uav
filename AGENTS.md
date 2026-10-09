# UAV 交接规则

本文件是所有接手 agent 的入口；细节以 [实验协议](docs/EXPERIMENT_PROTOCOL.md) 为准。不要凭旧 README 中过时的「进行中」勾选框判断状态。

1. 先运行 `git status -sb`、`git log --all --graph --oneline -20`，阅读 `README.md`、`docs/DATASETS.md`、`docs/LEGACY_EXPERIMENTS.md`，确认当前分支和实验记录。不得把其他分支成果默认为当前模型。
2. 新实验先建 `experiments/YYYYMMDD_<dataset>_<question>_<variant>_s<seed>/`；同日同名增加有意义的 variant，不能覆写旧实验。实验专用 `.py/.sh` 放在该目录 `scripts/`，通用模型代码放 `model/`，通用工具才放 `scripts/`。每次运行前提交源码、配置和实验 README；记录完整的 `source_commit`。运行产生的指标、状态、日志和报告再单独提交。
3. 每项实验必须有客观的 `README.md` 和 `record.json`，结果必须能找到脚本、命令、数据集/划分、源提交、产物和状态。失败/中止也是结果；未知字段写 `unknown` 并解释，禁止猜测。旧实验按兼容规则登记，不删除、不伪造运行时 commit。
4. 禁止把 checkpoint、视频、临时缓存和数据集提交到 Git；小型 CSV/JSON、配置、可阅读日志、报告、源码要提交。大产物在 `artifacts.json` 登记路径、大小、哈希和生成脚本；本地文件可缺失，不能声称 Git 可还原。
5. 每个有意义的实验阶段单独提交。提交消息格式见 [Git 规范](docs/GIT_WORKFLOW.md)。提交前检查引用路径和 `git diff --check`。完成后推送当前分支并核对远端哈希；推送失败必须明确说明，不得声称已同步。
6. 本次整理不开发新模型。涉及模型的未来工作须另立实验与提交。不要重写已存在的历史提交/标签；用追加记录修正历史文档。
