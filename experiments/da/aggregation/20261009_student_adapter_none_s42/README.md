# 20261009_student_adapter_none_s42

研究问题：4-block 学生承担 12 个教师重建目标时，学生 block 之间的轻量适配是否改善异常检测排序？本实验唯一变化因素是 adapter 类型。不加 adapter，对照组；12 个教师层各有重建头，每三个头由同一个学生 block 输出。

## 固定设置

冻结 DINOv3 12 层教师，OCBE 同时接收 12 层；学生 4 个 block，各挂 3 个原始教师层重建头，按深到浅的反向次序对齐。训练使用同一份 group-balanced top-5 余弦重建损失（alpha=0.5），不训练教师，不使用异常标签。seed=42，15 epoch，batch=64，输入 256×256，code_dim=64，AdamW/OneCycleLR。七个场景**逐场景重新初始化** OCBE、学生和优化器，场景间不传权重。

正常训练帧为该场景全部 `sequence*/train/*/*.jpg`；测试帧为该场景全部 `sequence*/test/*/*.jpg`，标签只从对应视频的 `.npy` 读取。每帧异常分数为 12 个重建头各自 top-5 patch 余弦距离的平均，越高越异常。先在每场景合并测试帧计算 ROC AUC×100，再对七场景 AUC 做算术平均。逐帧原始分数保存以便复核。划分指纹来自所用路径清单，运行日志记载。

对照：与同日 `none/channel/spatial` 另两目录比较。旧 `20260928_agg_explore` 只有历史单种子 loss 对照，运行时源码未知，不作配对主对照。

脚本：`scripts/run.sh SCENE GPU` 调用共享 `scripts/run_student_adapter.py`；目录内 `record.json` 和 `config.json` 是准绳。正式运行前先提交源码并记录完整 `source_commit`。输出写入 `logs/`、`results/`。当前状态：计划，尚未训练。

限制：只有 seed=42；不能由单次 AUC 波动断言稳健增益。训练参数量、整条管线计算量与推理延迟分开报告。
