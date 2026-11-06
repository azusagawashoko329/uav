# 消融实验

本目录按数据集分开保存消融实验。`ablation/da/GL/` 是 DA seed42 的 GL 消融矩阵；`ablation/uit/` 为后续 UIT 消融预留。

DA/GL 的八个单元固定 V2 母模板：DINOv3 ViT-B/16、12 个 teacher taps、OCBE hidden 512、code_dim 64、student width 768、4 stages、12 reconstruction heads、group-balanced top-5 cosine（alpha=0.5）、30 epoch、batch64、AdamW + OneCycleLR。只有 student 的 block order（GGGG、LLLL、GLLL、LLLG）和 FFN 类型（gate、gelumlp）变化。

其中 `20261106_glll_gate_s42` 从 DA V2 矩阵迁移，`20261106_gggg_gelumlp_s42` 从 DA V1 矩阵迁移；其余六项由 `scripts/da_gl_ablation_scheduler.py` 使用两张物理 GPU 动态排队运行。
