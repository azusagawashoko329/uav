# 20261107_patchmean_12mean_s42

DA / Loss 消融，配置：**Patch Mean Cosine; direct 12-layer mean**，seed42。以 V2 的 GLLL + gate student、OCBE hidden 512、12 taps、30 epoch 训练为母模板，只改变误差计算和组聚合。

训练与测试使用同一 `loss_score` match：error=`patchmean`，k=`196`，aggregation=`mean12`，alpha=`0.5`。使用脱离会话的双卡动态 scheduler。
