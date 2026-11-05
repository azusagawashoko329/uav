# V4 `rd_s4_t12_gl192_h128`：DA 30 epoch（matched score）

- macro AUC: 83.9925
- macro EER: 0.2029
- scoring: V0: mean over 4 taps of top5 mean; V1-V5: mean over 4 groups of 0.5*mean_tap(top5 mean)+0.5*max_tap(top5 mean).
- original top-k results: results_legacy_topk/

| scene | AUC | EER |
|---|---:|---:|
| bike | 97.7566 | 0.0459 |
| crossroads | 59.7789 | 0.4266 |
| farmland | 99.4880 | 0.0542 |
| highway | 93.5796 | 0.1748 |
| railway | 91.2036 | 0.1265 |
| solar | 86.5823 | 0.1942 |
| vehicle | 59.5584 | 0.3982 |
