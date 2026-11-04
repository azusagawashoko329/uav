# 20261104_variant_matrix_v5_rd_s4_t12_gl192_h128_qr2_s42

在 UIT-ADrone 上训练六变体矩阵的 **V5 / `rd_s4_t12_gl192_h128_qr2`**，45 epoch，seed42，双卡动态调度。

## 固定设置

- 冻结 DINOv3 ViT-B/16，输入 256×256；训练/测试沿用 UIT-ADrone 官方目录：
  `train/frames/<video>/*.jpg`（41 视频 / 59186 帧），
  `test/frames/<video>/*.jpg`（51 视频 / 147005 帧），
  `test/test_frame_mask/<video>.npy`。
- batch64、AdamW、OneCycleLR（max_lr 5e-4, weight_decay 5e-2），
  epoch 10/20/30/40/45 保存 checkpoint，最终另存 `uit_final.pth`。
- 变体：GL192 student, 12 taps, OCBE h128 + learnable rank2 random-orth Q。
- 教师 taps：`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`；bottleneck `code_dim=64`，`h512`/`h128` 指 OCBE
  `fuse_hidden`，不是 `code_dim`。V5 的 Q 为可学习 rank2、随机正交初始化。

## 版本对应检查

`scripts/uit_appearance_runner.py` 在训练前强制校验：

1. `config.variant_id` 必须映射到唯一 builder；
2. `config.variant` 必须等于该 builder；
3. `config.teacher_layers` 必须等于该变体的 canonical taps；
4. 构造出的 `ocbe+student` 参数量必须等于
   `resources.expected_trainable_parameters = 3761222`。

任一不匹配会直接报错，不进入训练。该检查用于避免把
`DA Student12 GL768`、旧 `DepthRank2` 等不同结构误当成 V2/V5。

## 评测

将 51 个 test 视频的 147005 个逐帧分数拼接后计算全局 AUC 与 EER；
保留 UIT 官方 train/test 重叠，不做去重。

## 限制

单 seed42；只训练外观分支；结果用于六变体同预算对照，不构成多种子结论。
