# DA / Loss three-seed ablation matrix summary

This matrix uses the V2 mother template: frozen DINOv3 ViT-B/16, 12 teacher taps, OCBE hidden 512, code_dim 64, student width 768, GLLL + gate student, 30 epochs, batch64, AdamW + OneCycleLR. Only reconstruction error and tap/group aggregation change.

Training and testing use the same configured loss_score match. Seed42 top5_group_a05 reuses the completed V2 matched-score results; the other 23 experiment cells are newly trained. AUC/EER are macro means over seven scenes, followed by mean and sample SD across seeds 42/43/44.

## Overall matrix

| configuration | seed42 AUC | seed43 AUC | seed44 AUC | mean AUC +/- SD | mean EER +/- SD |
|---|---:|---:|---:|---:|---:|
| Top-5 / 4 groups alpha=0.5 (V2) | 85.2147 | 85.7279 | 85.5780 | 85.5068 +/- 0.2639 | 0.1966 +/- 0.0003 |
| Top-5 / 4 groups alpha=1 | 84.9691 | 85.8098 | 85.2344 | 85.3377 +/- 0.4298 | 0.1980 +/- 0.0013 |
| Top-5 / 12-layer mean | 84.7171 | 84.6729 | 85.3494 | 84.9131 +/- 0.3785 | 0.1975 +/- 0.0053 |
| Top-10 / 4 groups alpha=0.5 | 84.4618 | 84.7936 | 85.1397 | 84.7984 +/- 0.3390 | 0.2050 +/- 0.0046 |
| Top-1 / 4 groups alpha=0.5 | 83.6151 | 85.2763 | 84.7188 | 84.5368 +/- 0.8454 | 0.1958 +/- 0.0106 |
| Flatten / 12-layer mean | 63.4987 | 63.2911 | 63.3213 | 63.3704 +/- 0.1122 | 0.4050 +/- 0.0022 |
| Patch Mean / 12-layer mean | 63.3080 | 62.8178 | 62.9628 | 63.0295 +/- 0.2518 | 0.4082 +/- 0.0032 |
| Patch Mean / 4 groups alpha=0.5 | 62.3398 | 62.5787 | 62.6101 | 62.5095 +/- 0.1478 | 0.4200 +/- 0.0019 |

## Scene AUC means across three seeds

| scene | Flatten/12 | PatchMean/12 | Top5/12 | Top5/group a.5 | Top5/group a1 | Top1/group a.5 | Top10/group a.5 | PatchMean/group a.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bike | 81.05 | 80.00 | 97.63 | 97.72 | 97.80 | 98.42 | 96.90 | 80.21 |
| crossroads | 56.29 | 56.58 | 61.58 | 61.89 | 60.26 | 60.95 | 60.28 | 55.38 |
| farmland | 55.79 | 55.12 | 99.19 | 99.47 | 99.59 | 99.70 | 98.40 | 55.17 |
| highway | 72.14 | 70.44 | 94.84 | 94.35 | 92.99 | 93.59 | 95.30 | 69.15 |
| railway | 71.58 | 70.14 | 94.21 | 95.61 | 96.19 | 96.08 | 93.42 | 71.45 |
| solar | 40.19 | 41.08 | 89.65 | 89.57 | 89.53 | 89.94 | 88.19 | 39.49 |
| vehicle | 66.56 | 67.83 | 57.28 | 59.94 | 61.01 | 53.07 | 61.10 | 66.71 |

## Analysis

- Best overall: Top-5 / four groups alpha=0.5 (V2), mean AUC 85.5068 +/- 0.2639 and mean EER 0.1966 +/- 0.0003. It combines the average and worst layer within each three-layer group and is stable across seeds.
- Flatten / 12-layer mean reaches 63.3704 +/- 0.1122 AUC and Patch Mean / 12-layer mean 63.0295 +/- 0.2518, about 22 percentage points below V2. Global or all-patch averaging dilutes localized anomaly evidence.
- With Top-5 error fixed, direct 12-layer averaging reaches 84.9131 +/- 0.3785 AUC, 0.5937 points below V2. Preserving within-group worst-layer information helps.
- Alpha=1 gives 85.3377 +/- 0.4298 AUC, slightly below alpha=0.5 and less stable; taking only the worst layer is not a consistent improvement.
- Top-1 gives 84.5368 +/- 0.8454 AUC and Top-10 gives 84.7984 +/- 0.3390. Top-5 is the best balance between hard-patch focus and stability in this matrix.
- Patch Mean with four-group aggregation reaches 62.5095 +/- 0.1478 AUC, so group aggregation cannot recover the local signal lost by averaging all patches.
- Crossroads and Vehicle are the most sensitive scenes. Bike, Farmland and Railway are generally high-AUC scenes with smaller configuration gaps. Top-1 helps some easy scenes but drops on Vehicle, producing high cross-seed variance.

## Limitations and evidence

- Three seeds and seven independent scene models per seed; SD is descriptive and not a significance test.
- Scheduler completed 161 new scene jobs, skipped 7 migrated seed42 V2 scene cells, and reported zero failures.
- Because training and test use the same configured loss_score, this is a matched loss-design comparison rather than a test-only readout ablation.
- The current teacher input is 256x256, giving 256 patches. Patch Mean uses all 256 patches; its config k=196 is unused as a non-topk placeholder.
