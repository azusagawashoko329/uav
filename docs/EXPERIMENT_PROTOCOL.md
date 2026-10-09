# 实验与产物记录协议 v1

## 命名和目录

新实验唯一目录名：`YYYYMMDD_<dataset>_<question>_<variant>_s<seed>`，全部小写 ASCII 与下划线。dataset 只用 `da`（Drone-Anomaly）或 `uit`（UIT-ADrone）；跨数据集实验用 `da_uit`。日期采用运行开始日（Asia/Taipei，YYYYMMDD），问题/变体使用能区分假设和关键改动的短词，不用 `final`、`new`、`test2`。多种子实验按种子分别建目录，另用汇总报告关联；禁止覆盖旧目录。历史命名保留原路径，由 `docs/LEGACY_EXPERIMENTS.md` 和各目录补充记录映射，避免破坏旧指针。

```
experiments/<id>/
  README.md          # 假设、设置、结果、限制；含状态和证据链接
  record.json        # 机器可读来源/状态
  config.json        # 完整固定参数、环境和数据划分
  scripts/           # 该实验独有的启动、训练、评测、汇总脚本
  logs/              # 小型文本日志
  results/           # CSV/JSON/小型图；每个产物登记来源
  artifacts.json     # 大文件清单（Git 不保存本体）
```

`model/` 只存复用的模型本体。模型变更必须有独立提交，说明影响的实验；实验脚本通过 import 使用确定的源码提交。未来脚本以仓库根目录定位相对路径，数据根目录从参数/环境变量读取。不要把运行中 `*.py` 散放仓库根、把实验输出写到别的实验，或修改已发表结果而不留版本。共享工具须在 `scripts/`，并在实验 `record.json` 指向其相对路径及源码 commit。

## 开始、运行、复核、提交

1. 从正确分支创建实验目录、README/record/config/scripts，先提交源码与配置，得到 40 位 `source_commit`。可用命令 `git rev-parse HEAD`；运行前工作树须干净。若脚本依赖未提交文件，先提交或标为不可复现，不得写假哈希。
2. 每次运行使用唯一 `run_id`（UTC 时间戳 `YYYYMMDDTHHMMSSZ` + 可选重试号），在 `record.json` 的 `runs` 中写 `started_at`、完整命令、源码 commit、数据集/划分、环境、seed、输入指纹、输出路径和当前状态。运行时状态只用 `planned/running/completed/failed/cancelled/unknown`；`completed` 表示命令成功结束，不能等同于科学结论成立。
3. 评测写明 frame/video 粒度、异常分数方向、AUC/EER 口径、筛选规则、标签读取路径。报告区分观测、推断和限制；保留负结果，不能从单种子小差异直接得出稳定提升。日志和指标只追加或新建 run，不静默覆盖。
4. `results/` 中每个文件在 `artifacts.json` 登记：相对路径、`kind`、生产脚本、输入文件、run_id、状态、SHA-256。权重/视频也登记本地路径、大小、SHA-256、保存位置（若有外部存储 URI）。绝不把 Git 忽略等同于已备份。
5. 提交顺序：`experiment: plan` → `experiment: run` → `experiment: evaluate` → `docs: review`。每个阶段包含该阶段证据，提交说明列明实验 ID、数据集、种子、source commit、主要指标或失败原因。推送并用远端哈希核对；详见 [Git 规范](GIT_WORKFLOW.md)。

`record.json` 至少包含：`schema_version`、`id`、`dataset`、`question`、`variant`、`status`、`source_commit`、`scripts`、`runs`、`result_summary`、`limitations`。`source_commit` 是运行时源码版本，不是文档整理提交；历史无法证实则写 `null`，并在 `limitations` 解释。状态以记录和日志证据为准，不能凭空补完。

## 旧实验兼容

历史目录与内容不强制改名，避免破坏代码中的路径。旧实验可只补 `RECORD.md`，逐项注明实际存在的配置、脚本、日志、指标、权重、已知提交及缺口。若缺运行时哈希、命令或 seed，直接标 `unknown`。旧 README 中过时的状态不删；新的 `RECORD.md` 为整理时的审计视图，原报告保留作历史原文。位于其它 Git 分支的实验按分支+commit 引用，不把分支代码混成一个“最新模型”。
