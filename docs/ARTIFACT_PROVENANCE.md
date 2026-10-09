# 已有结果与可视化来源索引

本页区分“由源码/文档可关联”和“精确运行时来源已核实”。旧实验缺运行命令或源码快照时，保持 unknown，不虚构指针。新实验按 [协议](EXPERIMENT_PROTOCOL.md) 在本目录的 `artifacts.json` 逐文件登记。

| 产物路径 | 类型与 Git 状态 | 可关联生成入口 | 运行状态/缺口 |
|---|---|---|---|
| `Visualization/output/<scene>.mp4` | 本地视频；`.gitignore` 排除 | `Visualization/visualize_appearance.py` 文件头明确该输出路径 | 具体生成 commit、命令、权重哈希 unknown；不能仅由视频确认使用了哪次训练。 |
| `detect_preview/<scene>.mp4` | 本地视频；`.gitignore` 排除 | `behavior/detect_vis.py` 的检测预览流程 | `detect_preview/bike_render.log` 可核对局部运行；其它视频运行源码/权重 unknown。 |
| `audit/figs/cloud_<scene>.png` | 小型图，已在历史 Git 中 | `audit/cloud_dist.py` | 审计解读见 `audit/REPORT.md`；生成时的精确命令 unknown。 |
| `audit/scores/<scene>.npz` | 小型分数数组，已在历史 Git 中 | `audit/` 相关脚本与 `audit/REPORT.md` | 逐文件命令/源码快照 unknown；不要把图像和 NPZ 当作可自行还原的原始数据。 |
| `experiments/20260617_all_model/results_fusion.csv` | 小型结果，已在历史 Git 中 | `fusion/eval_fusion.py`，管线说明见 `fusion/README.md` | 训练/评测权重哈希未记录；数值按原报告引用，运行时源码未独立证实。 |
| `experiments/20260617_all_model/ckpt/<scene>/<scene>.npz` | 小型缓存，已在历史 Git 中 | `fusion/` 评分导出流程 | 与 `.pth` 区分；生成命令 unknown。 |
| `experiments/fusion/scores/*.npz` | 本地可复算缓存，未入 Git | `fusion/dump_appearance_scores.py`、`fusion/dump_behavior_scores.py` | 未建立逐文件哈希；不能宣称云端保存。 |
| `experiments/20260928_agg_explore/logs/*.log`、`loss_compare.csv` | 小型日志/表，本次整理纳入 Git | `scripts/run_full_scenes.sh`、`scripts/probe_e2e.py`；CSV 来自对照汇总 | 运行时源码提交 unknown；`raw_raw12.txt` 只有 3 场景。 |
| `experiments/*/ckpt/*.pth`、`experiment*/**/*.pt` | 本地权重，Git 排除 | 每项需读对应 `RECORD.md` 与原脚本 | 大多数缺训练时 source commit/哈希；保留原文件，不能以 Git 作为备份。 |

历史笔记中的 `DABG/...` 与若干 `fusion/eval_eer.py` 等路径在此仓库不存在。保留笔记原文，同时以 [历史索引](LEGACY_EXPERIMENTS.md) 提醒这些指针是外部或缺失状态；引用前须补回原始文件和提交。
