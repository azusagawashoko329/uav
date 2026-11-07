# DA / GL 消融实验

seed42、seed43、seed44 的八个 GL 变体均已完成。单 seed42 结果见 [summary.md](summary.md)；三 seed 合并矩阵、均值/标准差和配对分析见 [multiseed_summary.md](multiseed_summary.md)，机器可读结果见 [multiseed_summary.json](multiseed_summary.json)。

所有训练和测试均使用 `matched_score(k=5, alpha=0.5)`。seed43/44 共 112 个场景任务由混合 seed 双卡动态队列完成，scheduler 无失败。
