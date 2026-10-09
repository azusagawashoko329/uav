# 历史实验与分支索引（2026-10-09 整理视图）

本页只编目已存在的实验，不合并模型分支，不重新评估指标。老目录名作为稳定指针保留；补充的 `RECORD.md` 标示磁盘证据与 unknown 字段。表中“有结果”只表示发现 CSV/JSON，不能自动认定报告完整。

## Git 演进线

| 分支/标签 | 本地头提交 | 可确认内容 |
|---|---|---|
| 整理前的本地 `origin/main` 引用 / `behavior-baseline` | `25cffaf` | 2026-06 行为分支基线；这是整理前保存在本机的引用，不能证明新 GitHub 仓库已有该提交。后续推送可能使 `origin/main` 指向新提交。 |
| `research/next` | `9cba699`（整理前） | 融合、审计、开发日志及早期外观实验；本次文档整理从此分支继续。 |
| `master` | `fd6cce0` | LAA adapter 与相关笔记/实验；从 `9cba699` 分叉。 |
| `explore/distill-layer-boundary` | `1a68d0c` | DLBM v001–v022、mask DLBM v001、边界探针与测试；从 `9cba699` 分叉。 |
| `rd-baseline-20260611` | `c76ff07` | RD 初始基线。 |

这些分支不线性包含彼此的代码。要查看历史实验的源码，请用对应分支和确切提交，不要在当前分支上从目录名推断版本。

## 当前分支下的旧实验

| 目录 | 现有证据 | 关联入口/说明 |
|---|---|---|
| `experiments/20260611_rd_baseline` | 配置、日志、CSV、原报告 | `run_experiment.sh`、`train.py`、`eval.py`；旧 README 的“训练进行中”已过时，优先读结果与 `RECORD.md`。 |
| `experiments/20260612_rd_{noln,notopk,alltap}` | 各有 CSV 和逐场景日志 | `run_ablations.sh`，原名称保留。 |
| `experiments/20260612_motion_probe1` | CSV、运行日志 | `probe_motion_v1.py`。 |
| `experiments/20260613_deit_teacher` | CSV、训练和评测日志 | `train_deit.py`、`eval_deit.py`。 |
| `experiments/20260617_all_model` | 融合 CSV、日志、小型 NPZ、本地权重 | `fusion/` 与 `behavior/`；NPZ 在旧历史已追踪，勿把本地 `.pth` 误认为云端资产。 |
| `experiments/codedim`、`experiments/codedim16` | CSV、shell 脚本 | 原名保留；参数与口径按各自记录。 |
| `experiments/20260706_signal_analysis`、`20260708_*`、`20260709_*`、`20260710_*` | 多数仅本地二进制权重/缓存 | 运行时源码与评测状态未知；已有目录逐个生成 `RECORD.md`，不能补造结果。 |
| `experiments/20260928_agg_explore` | 7 场景损失对照 CSV、日志；原 `raw_raw12.txt` 只有 3 场景 | 实验脚本已归入 `scripts/`，原三变体计划未完成。 |

## 其它历史目录

`experiment_DLBM/` 与 `experiment_mask_DLBM/` 的**已提交代码及部分审计结果**在 `explore/distill-layer-boundary`（提交 `1a68d0c`）；该分支原 README、`run/boundary/*/REVIEW.md`、`summary.json` 和 `run/dlbm/v001_relation_changepoint/` 是首选证据。本地 `experiment_DLBM/run/boundary/` 还有 v001–v022 的目录、审计和权重，但其中有尚未进入 Git 的文件，且本机 `.git/info/exclude` 曾忽略整个 `run/`。这些未核验文件不可宣称已同步。运行前检查分支与 `git check-ignore -v`，只追加可核验的小型元数据。

`experiment_LAA/`、`experiment_gate/`、`experiment/end2end_gate_20260710/` 和若干 `experiments/202607*` 主要存放未入 Git 的权重/缓存；现有目录不删除。它们缺可证实的运行时源码提交或完整指标，状态登记为 unknown；未来复核后以追加记录补全。`experiments/fusion/` 是历史分数缓存，不是独立科学实验；生成入口见 [产物来源](ARTIFACT_PROVENANCE.md)。

历史笔记 `note/方法对比总表.md` 引用了仓库外 `DABG/` 与 UIT 实验路径。当前仓库没有这些文件，引用视为**外部历史指针**；不能将笔记数值误写成当前分支可直接复现的产物。
