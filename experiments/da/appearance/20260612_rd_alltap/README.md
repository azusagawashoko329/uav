# 20260612_rd_alltap — 消融：层选择（全层蒸馏 vs 选 [5,7,9,11]）

> 此为迁移前的历史说明，旧路径/命令可能失效。当前状态见 [RECORD.md](RECORD.md)，旧新路径见 [迁移表](../../../../docs/MIGRATION_MAP.csv)。

**问题**: baseline 只在 teacher 的 4 层 [5,7,9,11] 抽 tap 做反向蒸馏。本实验改为**蒸馏全部 12 层**(teacher block 0..11 全部对齐到 student 12 个 block 反向重建),验证"选层"是否比"全层蒸馏"有帮助。

**唯一改动**: `--taps 0 1 2 3 4 5 6 7 8 9 10 11`(n_taps=12;OCBE 融合 12 层 → code_dim=64;student depth=12 在每个 block 都出一个重建头,reverse-align:teacher 浅层↔student 深层)。每层仍用 **top-k(k=5)** 计算 loss。其余与 baseline 完全一致(code_dim=64, img=256, AdamW lr5e-4 wd5e-2, OneCycleLR, bs64, ep15, per-scene 合并, 7 场景)。可训练参数 89.1M→97.0M(OCBE fuse 变大)。

**匹配打分**: RD-**topk**(与 baseline 同),global-concat。

启动(与 notopk 共用 `run_ablations.sh` 同一队列双卡):`bash run_ablations.sh`
eval: `python eval.py --exp 20260612_rd_alltap --gpu 0 --agg concat`
