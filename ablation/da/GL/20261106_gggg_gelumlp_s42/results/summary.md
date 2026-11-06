# V1 `rd_s4_t12_h512`：DA 30 epoch（matched score）

- macro AUC: 85.1314
- macro EER: 0.2031
- scoring: V0: mean over 4 taps of top5 mean; V1-V5: mean over 4 groups of 0.5*mean_tap(top5 mean)+0.5*max_tap(top5 mean).
- original top-k results: results_legacy_topk/

| scene | AUC | EER |
|---|---:|---:|
| bike | 97.7429 | 0.0448 |
| crossroads | 60.1778 | 0.4400 |
| farmland | 99.4896 | 0.0537 |
| highway | 94.8154 | 0.1567 |
| railway | 95.6081 | 0.1069 |
| solar | 89.4196 | 0.1803 |
| vehicle | 58.6661 | 0.4392 |
