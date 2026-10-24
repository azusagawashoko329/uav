# ocbe_dgt_crossattn_pool_randomorth_glll_width192_ckpt：seed42 七场景结果

三组三层教师 Rank-2 Q、组内 value/gate、192 维 Global-Local 学生、三对一对齐和 group-balanced top-k loss 保持不变；本实验只改变四组深度融合算子。每个场景均保存最终 OCBE 和 student 权重。

| 场景 | AUC | 相对 DGT pp | 相对 Dense pp | epoch15 loss |
|---|---:|---:|---:|---:|
| bike | 97.6314 | +0.0597 | -0.0003 | 0.130187 |
| crossroads | 62.3546 | +0.2358 | +0.5727 | 0.159099 |
| farmland | 99.3292 | +0.0862 | +0.0683 | 0.146748 |
| highway | 95.2689 | -0.3585 | +0.8228 | 0.161616 |
| railway | 81.8941 | -1.2068 | -3.7867 | 0.253018 |
| solar | 82.7636 | +0.0350 | -0.4028 | 0.237525 |
| vehicle | 56.6537 | -0.5002 | -2.0983 | 0.166569 |

宏平均 AUC：**82.2708**；相对 DGT **-0.2356 pp**；相对 Dense **-0.6892 pp**。

总可训练参数：**3,182,182**；相对 DGT 变化 **+2,464**。

权重：每个场景的 `ckpt/<scene>_final.pth` 均已生成，哈希已在 `artifacts.json` 登记。

该结果为 seed42 单种子观察，不能代表跨种子稳定性；本轮没有加入额外融合损失。
