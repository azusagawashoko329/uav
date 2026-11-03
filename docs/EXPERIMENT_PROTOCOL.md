# 实验协议 v2：一个实验一个目录

## 唯一目录结构

```
experiments/<dataset>/<family>/YYYYMMDD_<question>_<variant>_s<seed>/
  README.md           # 研究问题、设置、观察、限制；客观表述
  record.json         # 状态、源码提交、运行清单和结果指针
  config.json         # 全部固定参数、数据划分、环境
  artifacts.json      # 每个产物的生成链、大小、SHA-256 和存放方式
  scripts/            # 此实验独有的 .py、.sh
  logs/               # 每次运行独立日志
  results/            # 小型 CSV/JSON/图与表
  ckpt/               # 本地权重，Git 忽略
  visualizations/     # 本地大图/视频，Git 忽略
```

目录名是 `YYYYMMDD_<question>_<variant>_s<seed>`（ASCII 小写字母、数字、下划线）。dataset 为 `da`（Drone-Anomaly）、`uit`（UIT-ADrone）或 `da_uit`；family 从 `appearance`、`behavior`、`fusion`、`aggregation`、`boundary`、`gating`、`audit` 中选。日期为运行开始日（Asia/Shanghai）。单独种子单独目录；多个变体不能共用结果目录。不用 `final`、`new`、`test2` 充当唯一差异说明。

现存旧实验沿用原始 ID，如 `20260611_rd_baseline`，但已物理归入数据集/方向。旧运行可能缺完整配置、确切源码或脚本；通过 `RECORD.md` 与 `record.json` 表示缺口，不伪造事实。旧路径到新路径见 [迁移表](MIGRATION_MAP.csv)。`source_snapshot_*` 目录是历史分支代码证据，不能作为当前分支的运行入口。

## 变体 ID

六个反向蒸馏结构变体使用统一 ID（`rd_s12_t4_h512` … `rd_s4_t12_gl192_h128_qr2`），完整定义、代码入口和正式运行见 [VARIANTS.md](../experiments/VARIANTS.md)，机器可读版本见 [VARIANTS.csv](../experiments/VARIANTS.csv)。变体 ID 只标识模型结构与超参，不标识数据集、种子或 checkpoint；具体训练结果仍由实验 ID、dataset、seed 和 run_id 唯一确定。引用时不要用 `12block`、`GL192` 等口语名作为唯一标识。

## 运行和状态

1. 新建目录，写假设、基线、唯一变化因素、数据集/划分/标签与评测口径；放入实验专用脚本。通用实现放共享代码目录，通过 import 使用。先提交这一阶段，运行前记录 `git rev-parse HEAD` 的 40 位源码提交；工作树须干净。
2. 每次运行分配唯一 `run_id`（UTC `YYYYMMDDTHHMMSSZ`，重试加序号）。`record.json` 的 `runs` 写启动/结束时间、命令、源码提交、seed、数据根与划分指纹、环境、脚本、日志、产物和状态。状态只取 `planned/running/completed/failed/cancelled/unknown`。`completed` 只表示程序结束，不表示方法有效。
3. 逐帧异常分数方向、AUC/EER 的聚合粒度、标签读取、筛选规则要在 README 写清。结果说明区分**观察值、推断、限制**；负结果、评审失败和多种子方差照实保留。历史运行信息不足时写 unknown。
4. 每个结果、图、权重在 `artifacts.json` 写 `path`、`producer_script`、`run_id`、输入、生成状态、大小、SHA-256、Git 或外部存储位置。大文件不入 Git 并不等于有备份；如需共享，另记稳定 URI 和校验哈希。
5. 提交按计划源码 → 运行日志 → 评测结果 → 复盘拆开。改动旧记录采用新提交解释，不能重写已被引用的旧提交。推送后比对远端和本地 SHA，见 [Git 规范](GIT_WORKFLOW.md)。

新实验的 `record.json` 至少含 `schema_version`、`id`、`path`、`dataset`、`family`、`status`、`source_commit`、`scripts`、`runs`、`result_summary`、`limitations`。历史记录可加 `legacy`、`original_path`、`source_ref`、`review_markers`。`source_commit` 只能指运行时源码快照，未知用 `null`。

## 旧实验查找

从 [导航](../experiments/INDEX.md) 进入实验，再看 `record.json`、`RECORD.md`、原 README、结果和日志。旧目录没有脚本或指标时，状态为 unknown；不能由 `.pth` 存在推断训练、评测或复核成功。边界/LAA 源码在带分支提交号的 `source_snapshot_*` 中，若要复现需先按原分支对齐环境与路径，不能直接将快照代码并入当前 `model/`。
