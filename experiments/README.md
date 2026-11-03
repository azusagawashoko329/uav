# 实验库

请打开 [INDEX.md](INDEX.md) 按数据集和方向查找实验。`CATALOG.csv` 可供脚本读取，`docs/MIGRATION_MAP.csv` 保存旧路径映射。每个实验的 `record.json` 是状态和来源入口，`RECORD.md` 供人工阅读。历史缺口照实写 unknown，不把“有权重”视为“完成”。

六个反向蒸馏结构变体的统一 ID（`rd_s12_t4_h512` … `rd_s4_t12_gl192_h128_qr2`）见 [VARIANTS.md](VARIANTS.md) 和 [VARIANTS.csv](VARIANTS.csv)。变体 ID 只标识结构/超参；具体训练结果仍由实验 ID、dataset、seed 和 run_id 确定。

新实验创建和提交要求见 [实验协议](../docs/EXPERIMENT_PROTOCOL.md)。
