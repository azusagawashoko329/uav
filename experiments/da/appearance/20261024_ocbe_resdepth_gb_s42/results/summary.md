# ocbe_residual_depth_group_balanced_randomorth_glll_width192：seed42 七场景结果

三组三层教师 Rank-2 Q、组内 value/gate、192 维 Global-Local 学生、三对一对齐和 group-balanced top-k loss 保持不变；本实验只改变四组深度融合算子。每个场景均保存最终 OCBE 和 student 权重。

| 场景 | AUC | 相对 DGT pp | 相对 Dense pp | epoch15 loss |
|---|---:|---:|---:|---:|
| bike | 97.5745 | +0.0028 | -0.0571 | 0.122457 |
| crossroads | 63.5480 | +1.4293 | +1.7662 | 0.157387 |
| farmland | 99.1596 | -0.0834 | -0.1013 | 0.144666 |
| highway | 95.1927 | -0.4347 | +0.7466 | 0.158710 |
| railway | 82.6312 | -0.4697 | -3.0496 | 0.251043 |
| solar | 81.8218 | -0.9068 | -1.3445 | 0.234446 |
| vehicle | 55.3803 | -1.7735 | -3.3716 | 0.163079 |

宏平均 AUC：**82.1869**；相对 DGT **-0.3194 pp**；相对 Dense **-0.7730 pp**。

总可训练参数：**3,184,806**；相对 DGT 变化 **+5,088**。

权重：每个场景的 `ckpt/<scene>_final.pth` 均已生成，哈希已在 `artifacts.json` 登记。

该结果为 seed42 单种子观察，不能代表跨种子稳定性；本轮没有加入额外融合损失。
