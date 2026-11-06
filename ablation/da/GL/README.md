# DA / GL 消融实验

八个配置均已完成：六项由双卡动态队列训练，`GLLL + gate` 和 `GGGG + gelumlp` 迁移自已完成的 DA V2/V1 矩阵。

完整矩阵、逐场景 AUC/EER、FFN 与 block 顺序配对分析见 [summary.md](summary.md)，机器可读结果见 [summary.json](summary.json)。所有测试均使用训练匹配的 `matched_score(k=5, alpha=0.5)`。
