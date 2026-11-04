# 对比实验导航

共 21 个条目：completed=6，planned=15。

字段规范见 [SCHEMA.md](SCHEMA.md)；统一协议与添加外部方法见 [README.md](README.md)。

> 原始实验仍以 `experiments/` 为准；本目录只做对比编排，不移动/删除原始实验。

## da

| ID | 方法 | 变体 | 状态 | 轻量化 | macro AUC | macro EER | 来源 |
|---|---|---|---|---|---:|---:|---|
| [da-ours-rd_s12_t4_h512](da/ours/rd_s12_t4_h512/README.md) | ours | `rd_s12_t4_h512` | `completed` | `measured` | 84.7916 | 0.2029 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v0_rd_s12_t4_h512_s42) |
| [da-ours-rd_s4_t12_h512](da/ours/rd_s4_t12_h512/README.md) | ours | `rd_s4_t12_h512` | `completed` | `measured` | 84.9435 | 0.2011 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v1_rd_s4_t12_h512_s42) |
| [da-ours-rd_s4_t12_gl768_h512](da/ours/rd_s4_t12_gl768_h512/README.md) | ours | `rd_s4_t12_gl768_h512` | `completed` | `measured` | 85.3707 | 0.1932 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v2_rd_s4_t12_gl768_h512_s42) |
| [da-ours-rd_s4_t12_gl192_h512](da/ours/rd_s4_t12_gl192_h512/README.md) | ours | `rd_s4_t12_gl192_h512` | `completed` | `measured` | 84.0864 | 0.1992 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v3_rd_s4_t12_gl192_h512_s42) |
| [da-ours-rd_s4_t12_gl192_h128](da/ours/rd_s4_t12_gl192_h128/README.md) | ours | `rd_s4_t12_gl192_h128` | `completed` | `measured` | 83.9186 | 0.2024 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v4_rd_s4_t12_gl192_h128_s42) |
| [da-ours-rd_s4_t12_gl192_h128_qr2](da/ours/rd_s4_t12_gl192_h128_qr2/README.md) | ours | `rd_s4_t12_gl192_h128_qr2` | `completed` | `measured` | 84.0137 | 0.2021 | [experiment](../experiments/da/appearance/20261103_variant_matrix_v5_rd_s4_t12_gl192_h128_qr2_s42) |
| [da-adVAE-default](da/adVAE/default/README.md) | adVAE | `default` | `planned` | — | — | — | [upstream](https://github.com/WangXuhongCN/adVAE.git) |
| [da-GANomaly-default](da/GANomaly/default/record.json) | GANomaly | `default` | `planned` | `measured` | — | — | [upstream](https://github.com/samet-akcay/ganomaly.git) |
| [da-MemAE-default](da/MemAE/default/record.json) | MemAE | `default` | `planned` | `measured` | — | — | [upstream](https://github.com/donggong1/memae-anomaly-detection.git) |
| [da-MKD-default](da/MKD/default/record.json) | MKD | `default` | `planned` | `measured` | — | — | [upstream](https://github.com/rohban-lab/Knowledge_Distillation_AD.git) |
| [da-ANDT-default](da/ANDT/default/README.md) | ANDT | `default` | `planned` | `measured` | — | — | [upstream](https://github.com/Jin-Pu/Drone-Anomaly.git) |
| [da-Future Frame-default](da/Future%20Frame/default/record.json) | Future Frame | `default` | `planned` | — | — | — | [upstream](https://github.com/StevenLiuWen/ano_pred_cvpr2018.git) |
| [da-ST Dissoc.-default](da/ST%20Dissoc./default/record.json) | ST Dissoc. | `default` | `planned` | — | — | — | [upstream](https://github.com/ChangYunPeng/VideoAnomalyDetection.git) |
| [da-MLEP-default](da/MLEP/default/record.json) | MLEP | `default` | `planned` | — | — | — | [upstream](https://github.com/svip-lab/MLEP.git) |
| [da-ASTT-default](da/ASTT/default/record.json) | ASTT | `default` | `planned` | — | — | — | [upstream](https://github.com/Tungufm/ASTT.git) |
| [da-DAD-FSM-default](da/DAD-FSM/default/record.json) | DAD-FSM | `default` | `planned` | — | — | — | [upstream](https://github.com/AhmedFakhry47/DAD-FSM.git) |
| [da-HSTforU-default](da/HSTforU/default/record.json) | HSTforU | `default` | `planned` | — | — | — | [upstream](https://github.com/vt-le/HSTforU.git) |
| [da-MADViT-Full](da/MADViT/Full/record.json) | MADViT | `Full` | `planned` | — | — | — | [upstream](https://github.com/manojbalwant/MAD-ViT.git) |
| [da-MADViT-Multi-layer](da/MADViT/Multi-layer/record.json) | MADViT | `Multi-layer` | `planned` | — | — | — | [upstream](https://github.com/manojbalwant/MAD-ViT.git) |
| [da-MADViT-Last-layer](da/MADViT/Last-layer/record.json) | MADViT | `Last-layer` | `planned` | — | — | — | [upstream](https://github.com/manojbalwant/MAD-ViT.git) |
| [da-LightViT-AD-default](da/LightViT-AD/default/record.json) | LightViT-AD | `default` | `planned` | — | — | — | [upstream](https://github.com/manojbalwant/lightvit-ad.git) |

## uit

（暂无条目）

重新生成：`python comparison/build_catalog.py`；校验：`python comparison/check_comparisons.py`。
