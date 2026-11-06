# 20261106_gggg_gate_s44

DA / GL 多 seed 消融实验，配置 `GGGG + gate`，seed44。沿用 V2 母模板的 teacher、OCBE、student width、loss、训练和评测设置；仅 seed 与八项矩阵身份变化。

使用 `scripts/da_gl_ablation_multi_seed_scheduler.py` 脱离会话窗口进行双卡动态调度。训练和测试均使用 `matched_score(k=5, alpha=0.5)`。当前状态为 planned。
