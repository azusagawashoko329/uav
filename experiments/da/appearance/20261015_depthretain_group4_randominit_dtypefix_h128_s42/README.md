# 四组独立深度 Q：随机正交初始化，seed42

本实验与[Mean+Trend 初值版本](../20261015_depthretain_group4_fixedinit_dtypefix_h128_s42/README.md)成对检验四组独立 Q 是否依赖初始化。唯一差异是四个 `2×3` Q 各从 seed42 独立抽样的随机正交基开始；单独 CPU 生成器不影响 OCBE 其他层和学生的初始随机序列。每个场景两版在同一物理 GPU 上先后运行。

四组三层教师各自形成 `C_g`，没有跨组平均。损失、Q 单独优化 300 步、15 epoch 联合训练、GL192 四 block、OCBE hidden128/code64、三对一反向对齐、测试帧 AUC 口径与配对版本一致。正常教师能量保留不是 AUC 代理；既比较本版对[共享 Q 随机初值对照](../20261014_depthretain_randominit_h128_s42/results/summary.md)的 AUC，也比较四组 Q 对两种初值是否收敛到相同的各组行空间。失败和负结果保留。

## 重跑来源

[首轮实验](../20261015_depthretain_group4_randominit_h128_s42/README.md)因轨迹快照计算发生 float32/float64 混用，在 Q 预训练前失败，无 AUC。此目录使用修正后的快照计算，重新运行同一研究设置；首轮日志和状态保留在原目录。

## 运行归档

本次七场景均完成 15 epoch；配对调度 14/14 返回码为 0。原始逐帧分数、Q 轨迹、预训练数据和日志已登记。AUC 与方向比较待独立复核。
