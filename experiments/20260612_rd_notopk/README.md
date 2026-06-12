# 20260612_rd_notopk — 消融：top-k vs flatten 余弦（MADViT 式）

**问题**: baseline 的 loss 是逐 patch 余弦距离 → 取 **top-5 最难 patch** 均值(难例挖掘)。本实验改为 **MADViT 式 flatten 余弦**:把每个 tap 的 (P,D) patch 图**整张摊平**成一个 (P·D) 向量,teacher 与 student **算一次 cosine**(不是逐 patch 平均)。验证 top-k 在本方法(反向蒸馏)里是否有优势——MADViT 论文里验证过,自己的方法里也要消融一遍。

**唯一改动**: `--loss full`(`full_loss`:每 tap 摊平 (B,P·D) → 1−cosine → 层均值 → batch 均值)。taps 仍是 **[5,7,9,11]**,code_dim/depth/img/配方/LN 全部与 baseline 一致(7 场景, per-scene 合并, AdamW lr5e-4 wd5e-2, OneCycleLR, bs64, ep15)。

**匹配打分**: RD-**flat** 读数(摊平 cosine → 逐帧),与训练 loss 一致;global-concat。

启动(与 alltap 共用 `run_ablations.sh` 同一队列双卡):`bash run_ablations.sh`
eval: `python eval.py --exp 20260612_rd_notopk --gpu 0 --agg concat`（取输出里的 `flat` 列为匹配打分）
