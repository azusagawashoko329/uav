# V3 `rd_s4_t12_gl192_h512`：DA 30 epoch（matched score）

- macro AUC: 84.2137
- macro EER: 0.2001
- scoring: V0: mean over 4 taps of top5 mean; V1-V5: mean over 4 groups of 0.5*mean_tap(top5 mean)+0.5*max_tap(top5 mean).
- original top-k results: results_legacy_topk/

| scene | AUC | EER |
|---|---:|---:|
| bike | 97.7989 | 0.0440 |
| crossroads | 58.3984 | 0.4485 |
| farmland | 99.5709 | 0.0481 |
| highway | 94.6346 | 0.1554 |
| railway | 92.0129 | 0.1204 |
| solar | 86.5489 | 0.2021 |
| vehicle | 60.5315 | 0.3819 |
