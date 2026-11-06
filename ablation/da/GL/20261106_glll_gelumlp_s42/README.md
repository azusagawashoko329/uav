# 20261106_glll_gelumlp_s42

DA / GL 消融实验，配置 `GLLL` + `gelumlp`。以 `rd_s4_t12_gl768_h512`（V2）为母模板，只改变 student 的四块顺序与 FFN 类型。

固定项：DINOv3 ViT-B/16 冻结，teacher taps 0..11，OCBE hidden 512、code_dim 64，student width 768、4 stages、12 heads，group-balanced top-5 cosine（alpha=0.5），DA 七场景、seed42、30 epoch、batch64、AdamW + OneCycleLR。

运行前必须使用 `scripts/da_gl_ablation_scheduler.py` 的双卡动态队列，并检查每个进程首行 `model_guard` 与本配置一致。当前状态为 planned。
