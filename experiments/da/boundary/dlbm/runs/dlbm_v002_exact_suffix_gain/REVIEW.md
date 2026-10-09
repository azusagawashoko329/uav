# DLBM_v002_exact_suffix_gain — target/predictor failed, supernet retained

日期：2026-07-14  
协议：seed42 / 每场景独立 / 复用 Probe10 / DLBM 30 epoch / milestone 5、10、20、30

## 成功部分

- all-12 OCBE 第一层按 suffix 切片，与真正 fixed-suffix OCBE 前向严格等价。
- absolute Student 的 blocks、heads 可按边界复制，重建输出和最终 top-k score 均通过数值测试。
- 每场景的 12-suffix Normal/Pseudo 响应只缓存一次；DLBM 长训练不重复运行 Probe。
- 20 项单元测试通过，峰值缓存显存约 1.61 GiB，无 NaN/Inf。

## 长训练结果

| scene | exact-risk Oracle | ep5 | ep10 | ep20 | ep30 | validation 最优节点/边界 |
|---|---:|---:|---:|---:|---:|---:|
| bike | L6 | L6 | L6 | L4 | L6 | ep5 / L6 |
| crossroads | L6 | L6 | L6 | L6 | L4 | ep20 / L6 |
| farmland | L6 | L6 | L7 | L6 | L5 | ep30 / L5 |
| highway | L6 | L3 | L5 | L5 | L5 | ep30 / L5 |
| railway | L5 | L5 | L6 | L5 | L6 | ep20 / L5 |
| solar | L5 | L5 | L3 | L5 | L4 | ep5 / L5 |
| vehicle | L6 | L5 | L6 | L5 | L4 | ep10 / L6 |

30 epoch 已足以排除“只因 DLBM 没训练够”。后期出现的额外边界差异经常伴随更差 validation
MSE，例如 vehicle ep10=L6，继续到 ep30 后变成 L4，但验证误差明显恶化。

## 关键失败证据

1. exact-suffix 静态 pairwise 风险的 Oracle 五场景为 L6、两场景为 L5，本身缺少场景区分度。
2. 在每个场景上，用验证 episode 的平均 target gain 作为常数输出，MSE 都低于 DLBM；DLBM/常数
   MSE 比为 1.38–9.55。
3. 将 12 层输入顺序完全反转，预测平均只变化约 1.5e-5–4.6e-5，而预测 gain 本身通常为
   1e-2 量级，说明网络主要依赖输出 head bias/总体尺度，没有利用层序结构。
4. 该目标仍然属于冻结 Probe 下的静态 Normal/Pseudo 可分性，实测再次出现此前 LAA 所暴露的
   ViT 中层偏置。

## 决策

保留 `model/boundary_supernet.py` 作为下一版基础设施；v002 的静态风险 target 和 predictor 不进入
Final。下一步先做少量 coarse boundary 的 normal-only micro-adaptation，测量“同一起点、真实训练
若干步后”的 held-out 风险，而不是继续调整 v002 的 epoch、阈值或模型宽度。
