# 20261103_student12_plain12_failed3_retry_s42

针对 `20261102_student12_plain_retry2_s42` 在训练前失败的三个场景 `bike`、`crossroads`、`farmland` 做独立重跑。唯一设置为使用已修正的四个教师 taps `[5,7,9,11]`、Dense512 OCBE 和 Plain12 学生；seed、30 epoch、batch、数据划分与对应正式 retry4 保持一致。每个场景独立输出 AUC/EER、分数 CSV 和 checkpoint。

失败记录来自原实验日志中的 `AssertionError: expected 12 taps, got 4`；本目录不覆盖原失败记录。
