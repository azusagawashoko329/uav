# ocbe_residual_depth_score_aligned_randomorth_glll_width192：seed42 七场景结果

三组三层教师 Rank-2 Q、组内 value/gate、192 维 Global-Local 学生、三对一对齐和 group-balanced top-k loss 保持不变；本实验只改变四组深度融合算子。每个场景均保存最终 OCBE 和 student 权重。

| 场景 | AUC | 相对 DGT pp | 相对 Dense pp | epoch15 loss |
|---|---:|---:|---:|---:|
| bike | 97.5810 | +0.0093 | -0.0507 | 0.124914 |
| crossroads | 63.7385 | +1.6197 | +1.9566 | 0.154753 |
| farmland | 99.1667 | -0.0764 | -0.0943 | 0.141729 |
| highway | 95.0752 | -0.5522 | +0.6292 | 0.156318 |
| railway | 82.7179 | -0.3830 | -2.9629 | 0.247643 |
| solar | 82.0431 | -0.6855 | -1.1232 | 0.229586 |
| vehicle | 54.4041 | -2.7498 | -4.3479 | 0.160350 |

宏平均 AUC：**82.1038**；相对 DGT **-0.4025 pp**；相对 Dense **-0.8562 pp**。

总可训练参数：**3,184,806**；相对 DGT 变化 **+5,088**。

权重：每个场景的 `ckpt/<scene>_final.pth` 均已生成，哈希已在 `artifacts.json` 登记。

该结果为 seed42 单种子观察，不能代表跨种子稳定性；本轮没有加入额外融合损失。
