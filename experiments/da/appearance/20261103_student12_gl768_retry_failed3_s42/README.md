# 20261103_student12_gl768_retry_failed3_s42

按 `20261102_student12_gl768_retry5_s42` 的 GL12 配置重跑昨晚失败的三个场景：`railway`、`solar`、`vehicle`。使用双卡动态调度（GPU0/GPU1），初始各占用一个场景；任一场景结束后立即将队列中的下一个场景顶到同一张卡，保持可用卡不空置。教师 taps `[5,7,9,11]`、GL12 结构、Dense512 OCBE、seed42、30 epoch 与原配置一致。
