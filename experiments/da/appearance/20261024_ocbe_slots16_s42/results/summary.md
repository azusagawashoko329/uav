# ocbe_slot16_group_balanced_randomorth_glll_width192：seed42 七场景结果

三组三层教师 Rank-2 Q、组内 value/gate、192 维 Global-Local 学生、三对一对齐和 group-balanced top-k loss 保持不变；本实验只改变四组深度融合算子。每个场景均保存最终 OCBE 和 student 权重。

| 场景 | AUC | 相对 DGT pp | 相对 Dense pp | epoch15 loss |
|---|---:|---:|---:|---:|
| bike | 97.6343 | +0.0626 | +0.0027 | 0.128404 |
| crossroads | 63.6957 | +1.5769 | +1.9139 | 0.165034 |
| farmland | 99.2290 | -0.0141 | -0.0320 | 0.153062 |
| highway | 93.7867 | -1.8407 | -0.6593 | 0.168099 |
| railway | 81.6267 | -1.4742 | -4.0541 | 0.259299 |
| solar | 83.1987 | +0.4701 | +0.0324 | 0.244463 |
| vehicle | 54.5145 | -2.6393 | -4.2374 | 0.171940 |

宏平均 AUC：**81.9551**；相对 DGT **-0.5512 pp**；相对 Dense **-1.0048 pp**。

总可训练参数：**3,015,590**；相对 DGT 变化 **-164,128**。

权重：每个场景的 `ckpt/<scene>_final.pth` 均已生成，哈希已在 `artifacts.json` 登记。

该结果为 seed42 单种子观察，不能代表跨种子稳定性；本轮没有加入额外融合损失。
