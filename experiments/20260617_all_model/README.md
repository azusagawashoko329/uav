# 20260617_all_model — 双分支统一训练(外观 RD + 行为 检测/描述子/kNN)

一个脚本、一个环境(DMAD: timm DINOv3 + ultralytics YOLO)同时产出两分支产物,**固定 seed=42 可复现**;行为分支的目标检测**从头重跑**(不读旧 `behavior/cache`,检测只缓存到本实验 `det_cache/`)。完整前向与融合见 [../../fusion/PIPELINE.md](../../fusion/PIPELINE.md)。

## 训练配置(策略与历史一致)
- **外观**:DINOv3 ViT-B/16 frozen,taps=[5,7,9,11],OCBE code_dim=64,student depth=12,top-k(k=5)重建损失,img=256,**15 epoch**,bs=64,AdamW lr=5e-4 wd=5e-2 OneCycleLR,seed=42。
- **行为**:YOLOv8x-VisDrone frozen 从头检测,5 维描述子 [d_min,d_mean,n_close,count,max_local],logz 标准化(计数维 log1p 后 z-score),合并 kNN-OOD(K=8),检测质量 q=min(1,0.1/jitter)。
- **融合**:门控 `w=0.5·q·(1−2|p_app−0.5|)`,`fused=(1−w)·p_app+w·p_beh`,test 平均秩标准化,每场景 concat-micro AUC(MADViT 可比)。

## 产物(按场景分目录,两分支同落 ckpt/<scene>/)
- `ckpt/<scene>/<scene>.pth` — 外观 student+OCBE 权重 + cfg
- `ckpt/<scene>/<scene>.npz` — 行为正常点云状态(Xtr_proc / mu / sd / q / jitter)
- `det_cache/` — 本次从头检测的车辆中心点云  ·  `logs/<scene>.log` — 每场景训练日志  ·  `logs/eval.log`
- `results_fusion.csv` — 评估结果

## 结果(concat-micro AUC %)
| 场景 | 外观 | 行为 | q | 融合 | Δ |
|---|---|---|---|---|---|
| highway | 92.34 | 48.61 | 1.000 | **93.69** | +1.35 |
| farmland | 99.40 | 49.67 | 0.068 | **99.40** | +0.00 |
| bike | 98.08 | 55.82 | 0.345 | **97.40** | -0.68 |
| solar | 87.21 | 48.29 | 0.108 | **87.18** | -0.03 |
| vehicle | 60.54 | 72.60 | 1.000 | **68.76** | +8.22 |
| crossroads | 59.72 | 54.78 | 0.946 | **58.88** | -0.84 |
| railway | 94.94 | 49.69 | 0.050 | **94.94** | +0.00 |
| **均值** | **84.60** | | | **85.75** | **+1.15** |

## 训练耗时
| 场景 | GPU | 秒 | rc |
|---|---|---|---|
| vehicle | 1 | 1295 | 0 |
| crossroads | 0 | 2786 | 0 |
| bike | 1 | 1939 | 0 |
| highway | 0 | 1664 | 0 |
| solar | 0 | 518 | 0 |
| railway | 0 | 234 | 0 |
| farmland | 1 | 2103 | 0 |
