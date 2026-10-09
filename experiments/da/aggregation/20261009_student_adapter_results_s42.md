# 四层学生的层间 adapter 对照（Drone-Anomaly，seed 42）

## 结论与证据

在固定 12 层教师输入、12 个重建目标、4 个学生 block、group-balanced Top-k 损失和评测协议下，通道 adapter 的七场景平均 AUC 为 **84.11**，比无 adapter 的 **83.95** 高 **0.16** 个百分点；带局部卷积的 spatial adapter 为 **83.94**，比无 adapter 低 **0.01** 个百分点。这是单种子结果，不足以确认 adapter 有稳定收益。逐场景分数来自三个实验目录的 `results/<scene>.json` 与逐帧分数 CSV；[机器可读对照表](20261009_student_adapter_comparison_s42.csv)记录未四舍五入的数值。

| 场景 | 无 adapter | 通道 adapter | Δ | 通道+空间 adapter | Δ |
|---|---:|---:|---:|---:|---:|
| highway | 95.54 | 96.07 | +0.53 | 95.07 | -0.47 |
| farmland | 99.24 | 99.24 | +0.00 | 99.15 | -0.08 |
| bike | 97.71 | 97.70 | -0.00 | 97.71 | +0.00 |
| solar | 89.09 | 88.93 | -0.17 | 88.93 | -0.17 |
| vehicle | 53.98 | 54.77 | +0.79 | 53.32 | -0.66 |
| crossroads | 59.43 | 59.29 | -0.14 | 60.59 | +1.16 |
| railway | 92.67 | 92.81 | +0.14 | 92.79 | +0.12 |
| **七场景算术平均** | **83.95** | **84.11** | **+0.16** | **83.94** | **-0.01** |

通道 adapter 的改善主要在 vehicle 和 highway；spatial adapter 在 crossroads 上改善，但 vehicle 和 highway 下降。没有出现七场景一致改善。各列是**场景内合并所有测试帧后的 frame-level AUC×100**，最后一行是七个场景 AUC 的等权平均，不是把七场景测试帧全部拼接后的 AUC。

## 设置

- 每个场景单独从头训练一个模型：冻结 DINOv3 ViT-B/16 教师，取 L0–L11 patch 特征，经 12-tap OCBE（code_dim 64）进入 4-block 学生。每个学生 block 挂三个教师层的重建头，按深层教师到浅层学生的反向次序对齐。
- 仅比较学生 block 之间的三个位置：无 adapter、零初始化残差通道瓶颈（768→96→768）、相同瓶颈加 3×3 depthwise 卷积。其余实现相同。固定 seed 42、15 epoch、batch 64、256×256、AdamW、OneCycleLR、group-balanced Top-k（k=5，α=0.5）。测试分数仍为十二个头的 Top-k 距离等权平均，越大越异常。
- 正常训练集来自各场景的 `sequence*/train/*/*.jpg`；测试集来自同场景 `sequence*/test/*/*.jpg`，对应 `.npy` 标签只用于评测。三个变体在每个场景的划分指纹完全一致；逐帧 CSV 重新计算出的 AUC 与 JSON 一致，21 个任务日志均有第 15 轮和最终指标。
- 可训练参数：无 adapter **40,258,880**；通道 **40,708,448**（+449,568，+1.12%）；spatial **40,711,328**（+452,448，+1.12%）。这是 OCBE+学生的可训练参数，不包含冻结教师，也不是整条推理链的延迟或 FLOPs。

## 解释与限制

这组结果支持“轻量学生内部结构可以影响异常排序”，但目前的均值变化很小且方向不一致。不能仅因训练 Top-k loss 降低就推断 AUC 会升高，也不能把单种子 +0.16 解释为可靠的结构增益。若继续研究 adapter，优先对 `none` 与 `channel` 做多 seed 配对验证，并同时记录正常/异常分数分布；`spatial` 在本轮没有显示整体优势。

本轮**没有保存模型权重**，保留了运行源码提交、完整训练日志、每场景指标和逐帧分数，可复核本次 AUC，但无法对现有权重重新推理。历史基线的训练目标、头数或运行源码记录与本轮不完全相同，因此本文只将同轮 `none` 作为直接对照；不把旧实验数值并入主表。

首次调度的 21 个任务均在导入阶段因缺少仓库 Python 路径而失败，未进入训练；失败日志保留在各目录 `logs/<scene>.log`。修复提交 `c38dfa6d715f846b1b0526791814ae298aa7d344` 后，21 个正式重试任务全部完成，日志为 `logs/<scene>_retry1.log`。训练时的源码提交以该修复提交为准。

实验入口：[none](20261009_student_adapter_none_s42/README.md) · [channel](20261009_student_adapter_channel_s42/README.md) · [spatial](20261009_student_adapter_spatial_s42/README.md)。
