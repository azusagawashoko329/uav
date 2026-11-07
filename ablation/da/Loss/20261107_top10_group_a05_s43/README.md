# 20261107_top10_group_a05_s43

DA / Loss 消融，配置：**Top-10 Patch Mean; four groups, alpha=0.5**，seed43。以 V2 的 GLLL + gate student、OCBE hidden 512、12 taps、30 epoch 训练为母模板，只改变误差计算和组聚合。

训练与测试使用同一 `loss_score` match：error=`topk`，k=`10`，aggregation=`group`，alpha=`0.5`。使用脱离会话的双卡动态 scheduler。
