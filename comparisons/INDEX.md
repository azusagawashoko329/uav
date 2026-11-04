# 对比实验导航

共 21 个条目：completed=6，planned=15。

字段规范见 [SCHEMA.md](SCHEMA.md)；统一协议与添加外部方法见 [README.md](README.md)。

> 原始实验仍以 `experiments/` 为准；本目录只做对比编排，不移动/删除原始实验。

## da

| ID | 方法 | 变体 | 状态 | macro AUC | macro EER | 来源 |
|---|---|---|---|---:|---:|---|
| [da-ours-rd_s12_t4_h512](da/da-ours-rd_s12_t4_h512/README.md) | Ours | `rd_s12_t4_h512` | `completed` | 84.7916 | 0.2029 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v0_rd_s12_t4_h512_s42) |
| [da-ours-rd_s4_t12_h512](da/da-ours-rd_s4_t12_h512/README.md) | Ours | `rd_s4_t12_h512` | `completed` | 84.9435 | 0.2011 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v1_rd_s4_t12_h512_s42) |
| [da-ours-rd_s4_t12_gl768_h512](da/da-ours-rd_s4_t12_gl768_h512/README.md) | Ours | `rd_s4_t12_gl768_h512` | `completed` | 85.3707 | 0.1932 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v2_rd_s4_t12_gl768_h512_s42) |
| [da-ours-rd_s4_t12_gl192_h512](da/da-ours-rd_s4_t12_gl192_h512/README.md) | Ours | `rd_s4_t12_gl192_h512` | `completed` | 84.0864 | 0.1992 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v3_rd_s4_t12_gl192_h512_s42) |
| [da-ours-rd_s4_t12_gl192_h128](da/da-ours-rd_s4_t12_gl192_h128/README.md) | Ours | `rd_s4_t12_gl192_h128` | `completed` | 83.9186 | 0.2024 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v4_rd_s4_t12_gl192_h128_s42) |
| [da-ours-rd_s4_t12_gl192_h128_qr2](da/da-ours-rd_s4_t12_gl192_h128_qr2/README.md) | Ours | `rd_s4_t12_gl192_h128_qr2` | `completed` | 84.0137 | 0.2021 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v5_rd_s4_t12_gl192_h128_qr2_s42) |
| [da-advae-default](da/da-advae-default/record.json) | adVAE | `default` | `planned` | — | — | awaiting URL |
| [da-ganomaly-default](da/da-ganomaly-default/record.json) | GANomaly | `default` | `planned` | — | — | awaiting URL |
| [da-memae-default](da/da-memae-default/record.json) | MemAE | `default` | `planned` | — | — | awaiting URL |
| [da-mkd-default](da/da-mkd-default/record.json) | MKD | `default` | `planned` | — | — | awaiting URL |
| [da-andt-default](da/da-andt-default/record.json) | ANDT | `default` | `planned` | — | — | awaiting URL |
| [da-future-frame-default](da/da-future-frame-default/record.json) | Future Frame | `default` | `planned` | — | — | awaiting URL |
| [da-st-dissoc-default](da/da-st-dissoc-default/record.json) | ST Dissoc. | `default` | `planned` | — | — | awaiting URL |
| [da-mlep-default](da/da-mlep-default/record.json) | MLEP | `default` | `planned` | — | — | awaiting URL |
| [da-astt-default](da/da-astt-default/record.json) | ASTT | `default` | `planned` | — | — | awaiting URL |
| [da-dad-fsm-default](da/da-dad-fsm-default/record.json) | DAD-FSM | `default` | `planned` | — | — | awaiting URL |
| [da-hstforu-default](da/da-hstforu-default/record.json) | HSTforU | `default` | `planned` | — | — | awaiting URL |
| [da-madvit-full](da/da-madvit-full/record.json) | MADViT | `full` | `planned` | — | — | awaiting URL |
| [da-madvit-multi-layer](da/da-madvit-multi-layer/record.json) | MADViT | `multi-layer` | `planned` | — | — | awaiting URL |
| [da-madvit-last-layer](da/da-madvit-last-layer/record.json) | MADViT | `last-layer` | `planned` | — | — | awaiting URL |
| [da-lightvit-ad-default](da/da-lightvit-ad-default/record.json) | LightViT-AD | `default` | `planned` | — | — | awaiting URL |

## uit

（暂无条目）

重新生成：`python comparisons/build_catalog.py`；校验：`python comparisons/check_comparisons.py`。
