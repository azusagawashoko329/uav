# 20260612_rd_noln — ablation: OCBE 去掉 per-tap LayerNorm

**唯一改动**:OCBE 在 concat 多 taps 之前**不做** per-tap LayerNorm(`--no_tap_norm`,即 `tap_norms` 用 `nn.Identity`)。
其余**全部与 baseline 20260611_rd_baseline 一致**:DINOv3 ViT-B/16 teacher(冻结)、taps=[5,7,9,11]、code_dim=64、depth=12、k=5、img=256、AdamW lr(max)=5e-4 wd=5e-2、OneCycleLR、bs=64、ep=15、per-scene 合并、7 场景、动态双卡。

**目的**:验证 per-tap LayerNorm 是否为正向贡献。baseline 加 LN 的动机是各 tap 深度特征尺度差 ~30×(block11 量级远大于 block5),不归一化时 squeeze 会被深层主导。本实验直接去掉 LN 对照。

**对比口径**:与 baseline 同样用匹配打分(RD-topk)在 concat(micro)协议下逐场景对比;附 `--agg mean`(macro)。

启动:
```
bash run_experiment.sh 20260612_rd_noln --no_tap_norm
python eval.py --exp 20260612_rd_noln --gpu 0 --agg concat
python eval.py --exp 20260612_rd_noln --gpu 0 --agg mean
```

产物:`ckpt/<scene>.pth`(7)、`logs/<scene>.log`、`results.csv`(concat)、`results_mean.csv`(macro)、`run.log`。
