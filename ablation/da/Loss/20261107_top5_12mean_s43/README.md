# 20261107_top5_12mean_s43

DA / Loss 消融，配置：**Top-5 Patch Mean; direct 12-layer mean**，seed43。以 V2 的 GLLL + gate student、OCBE hidden 512、12 taps、30 epoch 训练为母模板，只改变误差计算和组聚合。

训练与测试使用同一 `loss_score` match：error=`topk`，k=`5`，aggregation=`mean12`，alpha=`0.5`。使用脱离会话的双卡动态 scheduler。
