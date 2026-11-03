# 20261103_variant_matrix_v1_rd_s4_t12_h512_s42

统一变体 ID：`rd_s4_t12_h512`（V1，4block）。定义见 [experiments/VARIANTS.md](../../../VARIANTS.md)。

## 研究问题

在 Drone-Anomaly 七场景 per-scene 协议下，用统一 30 epoch、seed42、双卡动态调度复跑 V1，作为六变体公平比较矩阵的一格。本实验是 DA 数据集上的重跑，不是 UIT 45ep 结果的替代。

## 结构

- **Student**：PlainFourBlockDecoder：4 个标准 Transformer Block；12 个重建头，每个 stage 3 个。
- **Teacher**：DINOv3 ViT-B/16 frozen，taps 0..11（全部 12 层）。
- **OCBE**：OCBE(n_taps=12, code_dim=64, fuse_hidden=512)。
- **对齐**：12 个教师层按相邻 3 层分 4 组；第 1 个 student stage 重建最深组 9–11，最后一个 stage 重建 0–2。
- **损失**：`group_balanced_top5_cosine_alpha0.5`。
- **可训练参数量**：40,258,880（OCBE + student；teacher 冻结）。

## 固定设置

- 数据集：Drone-Anomaly，七个场景分别训练/测试；每个场景合并所有 sequence 的 train 帧训练，合并该场景 test 帧计算 global concat AUC。
- seed42，30 epoch，batch64，workers8，输入 256×256；AdamW，max_lr=5e-4，weight_decay=5e-2，OneCycleLR。
- 每个场景独立初始化 OCBE、student 和优化器；不跨场景共享状态。
- 每场景只保存 final checkpoint；权重在 `ckpt/`，由 `.gitignore` 排除，`artifacts.json` 登记大小与 SHA-256。

## 双卡动态调度

`scripts/da_variant_scheduler.py` 在两张 GPU 上维护一个全局场景队列：每张卡一次跑一个场景进程，场景结束立即在同一张卡启动队列中的下一个场景；没有固定时长轮转，也不在队列非空时空卡等待。GPU 用 nvidia-smi 索引选择，但子进程通过 UUID 设置 `CUDA_VISIBLE_DEVICES`，避免 CUDA 枚举顺序与 nvidia-smi 不一致。

## 评测

- `anomaly_readouts(topk, k=5)` 对全部 tap 取平均作为逐帧分数。
- 每个场景把所有 test 视频的逐帧分数拼接后计算一次 ROC AUC×100 与 EER。
- 七场景宏平均作为该变体的汇总；结果写入 `results/summary.json` / `results/summary.md`。

## 与 UIT 六变体矩阵的关系

结构定义与 [`experiments/VARIANTS.md`](../../../VARIANTS.md) 的 V1 一致。该变体的 UIT 45ep 正式实验是 `20261101_uit_45ep_4block_dense512_retry2_s42`；本目录只记录 DA 30ep 运行，两者不能混引。

## 限制

单 seed42；七个独立场景模型；AUC/EER 不直接优化训练损失；行为分支未启用；V0 与 V1–V5 的 teacher taps/loss 不同，V0→V1 不是严格单因素对照。
