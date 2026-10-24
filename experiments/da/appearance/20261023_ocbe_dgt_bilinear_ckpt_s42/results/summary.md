# ocbe_dgt_bilinear_randomorth_glll_width192_ckpt：seed42 七场景结果

三组三层教师 Rank-2 Q、组内 value/gate、192 维 Global-Local 学生、三对一对齐和 group-balanced top-k loss 保持不变；本实验只改变四组深度融合算子。每个场景均保存最终 OCBE 和 student 权重。

| 场景 | AUC | 相对 DGT pp | 相对 Dense pp | epoch15 loss |
|---|---:|---:|---:|---:|
| bike | 97.6348 | +0.0631 | +0.0031 | 0.121868 |
| crossroads | 60.8573 | -1.2615 | -0.9245 | 0.156617 |
| farmland | 99.1266 | -0.1165 | -0.1344 | 0.143826 |
| highway | 93.7282 | -1.8992 | -0.7179 | 0.157420 |
| railway | 81.7658 | -1.3351 | -3.9149 | 0.249349 |
| solar | 81.2760 | -1.4526 | -1.8904 | 0.232237 |
| vehicle | 58.1310 | +0.9772 | -0.6209 | 0.161531 |

宏平均 AUC：**81.7885**；相对 DGT **-0.7178 pp**；相对 Dense **-1.1714 pp**。

总可训练参数：**3,182,758**；相对 DGT 变化 **+3,040**。

权重：每个场景的 `ckpt/<scene>_final.pth` 均已生成，哈希已在 `artifacts.json` 登记。

该结果为 seed42 单种子观察，不能代表跨种子稳定性；本轮没有加入额外融合损失。
