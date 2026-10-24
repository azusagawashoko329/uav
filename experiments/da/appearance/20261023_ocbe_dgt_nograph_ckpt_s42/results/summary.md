# ocbe_dgt_no_graph_randomorth_glll_width192_ckpt：seed42 七场景结果

三组三层教师 Rank-2 Q、组内 value/gate、192 维 Global-Local 学生、三对一对齐和 group-balanced top-k loss 保持不变；本实验只改变四组深度融合算子。每个场景均保存最终 OCBE 和 student 权重。

| 场景 | AUC | 相对 DGT pp | 相对 Dense pp | epoch15 loss |
|---|---:|---:|---:|---:|
| bike | 97.5484 | -0.0233 | -0.0833 | 0.127383 |
| crossroads | 62.0721 | -0.0466 | +0.2903 | 0.156752 |
| farmland | 99.2409 | -0.0022 | -0.0200 | 0.142980 |
| highway | 95.5810 | -0.0464 | +1.1349 | 0.157737 |
| railway | 83.0864 | -0.0145 | -2.5943 | 0.248059 |
| solar | 82.7592 | +0.0306 | -0.4072 | 0.231007 |
| vehicle | 56.8938 | -0.2600 | -1.8582 | 0.161485 |

宏平均 AUC：**82.4545**；相对 DGT **-0.0518 pp**；相对 Dense **-0.5054 pp**。

总可训练参数：**3,179,590**；相对 DGT 变化 **-128**。

权重：每个场景的 `ckpt/<scene>_final.pth` 均已生成，哈希已在 `artifacts.json` 登记。

该结果为 seed42 单种子观察，不能代表跨种子稳定性；本轮没有加入额外融合损失。
