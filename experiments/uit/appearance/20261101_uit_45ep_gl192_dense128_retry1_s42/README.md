# 20261031_uit_45ep_gl192_dense128_s42

在 UIT-ADrone 上比较 45 epoch 训练预算下的 `uit_gl192_dense128`。本实验是对先前 15/30 epoch 结果的延长训练验证。

## 固定设置

- 冻结 DINOv3 ViT-B/16，12 层 patch taps，输入 256×256；
- 相邻三个教师层与学生反向对齐，蒸馏损失保持 group-balanced top-5 cosine；
- seed42、45 epoch、batch64、AdamW、OneCycleLR；
- 在 epoch 10、20、30、40、45 分别保存 checkpoint，同时保存最终 `uit_final.pth`；
- 仅运行外观分支，UIT 官方 train/test 协议保留，不对 test 中可能与 train 重叠的帧去重。

## 评测

将 51 个 test 视频的 147005 个逐帧分数拼接后计算全局 AUC 与 EER。

## 限制

单 seed42；45 epoch 结果只说明训练预算影响，不能替代多种子稳定性分析。
