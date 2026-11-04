# 对比实验导航

共 21 个条目：completed=6，planned=15。

字段规范见 [SCHEMA.md](SCHEMA.md)；统一协议与添加外部方法见 [README.md](README.md)。

> 原始实验仍以 `experiments/` 为准；本目录只做对比编排，不移动/删除原始实验。

## da

| ID | 方法 | 变体 | 状态 | macro AUC | macro EER | 来源 |
|---|---|---|---|---:|---:|---|
| [da-ours-rd_s12_t4_h512](da/ours/rd_s12_t4_h512/README.md) | Ours | `rd_s12_t4_h512` | `completed` | 84.7916 | 0.2029 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v0_rd_s12_t4_h512_s42) |
| [da-ours-rd_s4_t12_h512](da/ours/rd_s4_t12_h512/README.md) | Ours | `rd_s4_t12_h512` | `completed` | 84.9435 | 0.2011 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v1_rd_s4_t12_h512_s42) |
| [da-ours-rd_s4_t12_gl768_h512](da/ours/rd_s4_t12_gl768_h512/README.md) | Ours | `rd_s4_t12_gl768_h512` | `completed` | 85.3707 | 0.1932 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v2_rd_s4_t12_gl768_h512_s42) |
| [da-ours-rd_s4_t12_gl192_h512](da/ours/rd_s4_t12_gl192_h512/README.md) | Ours | `rd_s4_t12_gl192_h512` | `completed` | 84.0864 | 0.1992 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v3_rd_s4_t12_gl192_h512_s42) |
| [da-ours-rd_s4_t12_gl192_h128](da/ours/rd_s4_t12_gl192_h128/README.md) | Ours | `rd_s4_t12_gl192_h128` | `completed` | 83.9186 | 0.2024 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v4_rd_s4_t12_gl192_h128_s42) |
| [da-ours-rd_s4_t12_gl192_h128_qr2](da/ours/rd_s4_t12_gl192_h128_qr2/README.md) | Ours | `rd_s4_t12_gl192_h128_qr2` | `completed` | 84.0137 | 0.2021 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v5_rd_s4_t12_gl192_h128_qr2_s42) |
| [da-advae-default](da/advae/default/record.json) | adVAE | `default` | `planned` | — | — | [upstream](https://github.com/WangXuhongCN/adVAE.git) |
| [da-ganomaly-default](da/ganomaly/default/record.json) | GANomaly | `default` | `planned` | — | — | [upstream](https://github.com/samet-akcay/ganomaly.git) |
| [da-memae-default](da/memae/default/record.json) | MemAE | `default` | `planned` | — | — | [upstream](https://github.com/donggong1/memae-anomaly-detection.git) |
| [da-mkd-default](da/mkd/default/record.json) | MKD | `default` | `planned` | — | — | [upstream](https://github.com/rohban-lab/Knowledge_Distillation_AD.git) |
| [da-andt-default](da/andt/default/record.json) | ANDT | `default` | `planned` | — | — | [upstream](https://github.com/Jin-Pu/Drone-Anomaly.git) |
| [da-future-frame-default](da/future-frame/default/record.json) | Future Frame | `default` | `planned` | — | — | [upstream](https://github.com/StevenLiuWen/ano_pred_cvpr2018.git) |
| [da-st-dissoc-default](da/st-dissoc/default/record.json) | ST Dissoc. | `default` | `planned` | — | — | [upstream](https://github.com/ChangYunPeng/VideoAnomalyDetection.git) |
| [da-mlep-default](da/mlep/default/record.json) | MLEP | `default` | `planned` | — | — | [upstream](https://github.com/svip-lab/MLEP.git) |
| [da-astt-default](da/astt/default/record.json) | ASTT | `default` | `planned` | — | — | [upstream](https://github.com/Tungufm/ASTT.git) |
| [da-dad-fsm-default](da/dad-fsm/default/record.json) | DAD-FSM | `default` | `planned` | — | — | [upstream](https://github.com/AhmedFakhry47/DAD-FSM.git) |
| [da-hstforu-default](da/hstforu/default/record.json) | HSTforU | `default` | `planned` | — | — | [upstream](https://github.com/vt-le/HSTforU.git) |
| [da-madvit-full](da/madvit/full/record.json) | MADViT | `full` | `planned` | — | — | [upstream](https://github.com/manojbalwant/MAD-ViT.git) |
| [da-madvit-multi-layer](da/madvit/multi-layer/record.json) | MADViT | `multi-layer` | `planned` | — | — | [upstream](https://github.com/manojbalwant/MAD-ViT.git) |
| [da-madvit-last-layer](da/madvit/last-layer/record.json) | MADViT | `last-layer` | `planned` | — | — | [upstream](https://github.com/manojbalwant/MAD-ViT.git) |
| [da-lightvit-ad-default](da/lightvit-ad/default/record.json) | LightViT-AD | `default` | `planned` | — | — | [upstream](https://github.com/manojbalwant/lightvit-ad.git) |

## uit

（暂无条目）

重新生成：`python comparison/build_catalog.py`；校验：`python comparison/check_comparisons.py`。
