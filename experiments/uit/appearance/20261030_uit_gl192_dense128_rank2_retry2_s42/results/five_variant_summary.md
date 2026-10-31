# UIT-ADrone 关键轻量化版本对比：seed42 / 30 epoch

五个版本均完整训练 30 epoch，调度器返回码为 0。指标由保存的逐帧分数独立复算；最终权重 SHA-256、严格加载和参数量已核验。

## 统一口径

仅外观分支。41 个 train 视频 / 59,186 帧；51 个 test 视频 / 147,005 帧，其中异常帧 63,485、正常帧 83,520。
教师为冻结 DINOv3 ViT-B/16，输入 256×256，12 层 patch taps；相邻三层对应一个学生 stage，反向对齐不变。
AdamW、OneCycleLR、batch64、top-5 group-balanced cosine loss（alpha=0.5）。使用完整 epoch：drop_last 后每轮 924 个 batch / 59,136 帧，30 轮共 27,720 个更新。
异常分数越高越异常：各教师层取 patch 余弦距离最大的 5 个取均值，再对 12 层取均值。整个 test 直接拼接计算一个全局 AUC/EER，不按视频归一化、不平滑、不选取最好 epoch。EER 采用离散 ROC 上 |FNR-FPR| 最小点的 FPR。

保留历史 UIT 官方 train/test 使用方式，不对 test 中可能与 train 重叠的帧去重。本批次没有重新审计这种重叠。Params 是 OCBE + Student 的可训练参数，不含冻结教师。

## 结果

| 方法 | AUC↑ (%) | EER↓ (%) | Params (M) | 相对前一版本 AUC (百分点) | 相对 4block 减参 |
|---|---:|---:|---:|---:|---:|
| 4block + Dense512 | 81.7450 | 25.9195 | 40.258880 | +0.0000 | 0.00% |
| 4block + GL768 + Dense512 | 80.6047 | 27.2713 | 30.248000 | -1.1402 | 24.87% |
| 4block + GL192 + Dense512 | 75.4872 | 32.4246 | 7.718336 | -5.1175 | 80.83% |
| 4block + GL192 + Dense128 | 75.0842 | 32.6221 | 4.154432 | -0.4030 | 89.68% |
| 4block + GL192 + Dense128 + Rank2 | 75.0613 | 32.6580 | 3.761222 | -0.0229 | 90.66% |

## 观察与解释

1. GL768 相对普通 4block 的 AUC 下降 1.1402 个百分点，参数减少 24.87%。
2. GL768 → GL192 的 AUC 下降 5.1175 个百分点，是这条迭代链最大的掉点来源。相同 30 epoch 下，这说明学生宽度压缩在本次 UIT 设置中的代价明显；不能把 Drone-Anomaly 上的表现直接外推到 UIT。
3. Dense512 → Dense128 再下降 0.4030 个百分点，参数从 7.718M 降至 4.154M。
4. Dense128 → Rank2 的 AUC 仅变化 -0.0229 个百分点，同时继续减少 393,210 个参数。本次 Rank2 为共享、随机正交初始化、可学习的 2×3 Q，不是固定 Mean+Trend，也不是 Group-Block LR。
5. 最轻版本相对普通 4block 共减少 90.66% 参数，但 AUC 下降 6.6837 个百分点。因此本批次支持 Rank2 的额外压缩代价很小，不支持整条轻量化链在 UIT 上几乎无损的说法。
这些是单个 seed42 的观察值。尚未估计多种子方差，不能把 0.023 个百分点的差距解释为稳定优劣。最终蒸馏 loss 也不等价于 AUC。

## 训练耗时（仅日志观察，非速度基准）

| 方法 | GPU（运行时名称） | 30 轮训练 (h) | 第 1 轮 (s) | 第 2–30 轮中位数 (s) | 评测与保存 (h) |
|---|---|---:|---:|---:|---:|
| 4block + Dense512 | NVIDIA RTX 5880 Ada Generation | 3.172 | 376.9 | 381.0 | 2.130 |
| 4block + GL768 + Dense512 | NVIDIA RTX A6000 | 3.770 | 425.6 | 425.7 | 3.205 |
| 4block + GL192 + Dense512 | NVIDIA RTX 5880 Ada Generation | 5.138 | 3336.5 | 369.5 | 2.550 |
| 4block + GL192 + Dense128 | NVIDIA RTX A6000 | 2.756 | 1379.1 | 285.8 | 1.929 |
| 4block + GL192 + Dense128 + Rank2 | NVIDIA RTX A6000 | 3.646 | 1871.5 | 300.2 | 1.013 |

GPU 型号不同，双任务共享存储与 CPU，轮间耗时波动明显；上述时间包含训练读取和冻结教师计算，不能据此比较推理 FPS/Latency。评测还调用了全部 anomaly_readouts 后选择 topk，没有使用 Q 推理缓存或融合内核；耗时不代表优化后的部署速度。

## 权重与追溯

训练源码快照：`ff5660e4077484f895755cce95a9a3917582edf6`；评审脚本源码：`dccc43d5641194afbe15790b916a2a89e7b1d332`。调度器 run_id：`20261030T070733Z_uit_five_variant_matrix`。
旧 wrapper 失败和 retry1 中断均保留；本报告只使用 retry2。未新增训练，也未用已有 ckpt 再微调。

| 实验目录 | 权重（本地，Git 忽略） | SHA-256 |
|---|---|---|
| [20261030_uit_4block_plain_retry2_s42](../../20261030_uit_4block_plain_retry2_s42/README.md) | `experiments/uit/appearance/20261030_uit_4block_plain_retry2_s42/ckpt/uit_final.pth` | `ea5d90481d5f2c04aef77c867eb6632957cb96a4f35849e9ecbfe77cce56b750` |
| [20261030_uit_4block_gl768_retry2_s42](../../20261030_uit_4block_gl768_retry2_s42/README.md) | `experiments/uit/appearance/20261030_uit_4block_gl768_retry2_s42/ckpt/uit_final.pth` | `927426f6716763668ab433de4f7e2d451ffc667b3069df80c5e3fad84fab4e89` |
| [20261030_uit_4block_gl192_retry2_s42](../../20261030_uit_4block_gl192_retry2_s42/README.md) | `experiments/uit/appearance/20261030_uit_4block_gl192_retry2_s42/ckpt/uit_final.pth` | `ef8e278fcd6b8ce00ea63349a5e2b1e7de467dfb87d0010044c53d6af8a7badc` |
| [20261030_uit_gl192_dense128_retry2_s42](../../20261030_uit_gl192_dense128_retry2_s42/README.md) | `experiments/uit/appearance/20261030_uit_gl192_dense128_retry2_s42/ckpt/uit_final.pth` | `e6d704f480f69a200eb4f45425430c1299b3dda53de3aab34528e6b28860e3c0` |
| [20261030_uit_gl192_dense128_rank2_retry2_s42](../../20261030_uit_gl192_dense128_rank2_retry2_s42/README.md) | `experiments/uit/appearance/20261030_uit_gl192_dense128_rank2_retry2_s42/ckpt/uit_final.pth` | `94fb3b84e91521fbc35769d19677a8345ed7d6f730e988ec5b16581af2e54947` |
