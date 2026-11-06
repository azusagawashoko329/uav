# 20261106_gggg_gelumlp_s42

DA / GL 消融实验，配置 `GGGG` + `gelumlp`。这是 V2 母模板（`rd_s4_t12_gl768_h512`）下的一个消融单元。

本目录直接迁移已完成的 `experiments/da/appearance/20261103_variant_matrix_v1_rd_s4_t12_h512_s42`，保留原始日志、结果、产物登记和 checkpoint。没有在本目录重新训练；原始 `source_commit`、`run_id` 与评分证据以 `record.json` 和 `artifacts.json` 为准。

固定项：DINOv3 ViT-B/16 冻结，teacher taps 0..11，OCBE hidden 512、code_dim 64，student width 768、4 stages、12 heads，group-balanced top-5 cosine（alpha=0.5），DA 七场景、seed42、30 epoch。
