# HAP-RD-Lite-Per-Q / UIT-ADrone / seed44

本实验补充 HAP-RD-Lite 的 Q 共享约束消融。F2 之后加入 PODP，但四个相邻三层 teacher group 各自使用一个独立可学习的 2×3 Q。它与 UIT 正式 HAP-RD-Lite 的唯一区别是 Q 从四组共享变为四组独立。

## 模型与约束

- 学生固定 LLLG+GATE，3 个 GLMB 后接 1 个 GGMB，width=192，PDFB hidden=128，code_dim=64，12 个独立重建头。
- `GroupPDFB` 保存四个独立的 `[2,3]` 原始 Q，合计 24 个原始标量，比共享 Q 多 18 个可训练参数，总参数量为 3,761,240。
- 四个 Q 按 seed 随机正交初始化。前向对每个 group 单独做可微 Gram-Schmidt，训练和测试都使用这四个 group-specific Q。
- teacher 冻结，teacher taps 为 0..11，四组分别对应 taps 0–2、3–5、6–8、9–11。

## 数据与训练

沿用已完成共享-Q UIT 正式实验的官方协议。train/frames 为 41 个正常训练视频、59186 帧，test/frames 为 51 个测试视频、147005 帧，标签为 `test/test_frame_mask/<video>.npy`，保留官方 train/test overlap。每个 seed 训练一个覆盖整个 UIT 训练集的模型。

训练 45 epoch，batch64，workers8，AdamW(max_lr=5e-4, weight_decay=0.05)+OneCycleLR，确定性 cuDNN，drop_last=True。checkpoint 保存 epoch10/20/30/40/45 与 final，resume 保存完整 optimizer、scheduler 和 RNG。

## 测试

测试重新加载同一 per-Q 模型实例的最终权重。每个 teacher/reconstruction 对按 patch 计算 1-cosine，取 top-5 后按四组三层的 alpha=0.5 计算 PLDD，再拼接全部 51 个视频帧计算全局 AUC×100 和离散 ROC EER。不会调用共享-Q模型或共享-Q评分分支。

## 入口与运行

训练入口为 `scripts/uit_hap_rd_lite_per_q_runner.py`，模型入口为 `model/hap_lllg_per_q.py`，调度入口为 `scripts/uit_hap_rd_lite_per_q_scheduler.py`。preflight 必须确认实际 guard 为 `q_shape=[4,2,3]`、`q_shared_across_groups=false`、参数量 3,761,240，并检查 forward/backward、独立初始化、更新后正交性和 PLDD 测试入口。

当前状态为 planned。先提交计划和 preflight，再绑定精确 source_commit，使用双卡 UUID 动态队列脱离会话运行。

## 限制

这是 UIT 上对 Q 共享约束的三 seed 结构消融。三个 seed 只支持描述性比较，不作统计显著性断言。