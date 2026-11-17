# V0 `rd_s12_t4_h512`：DA 30 epoch（matched score）

- macro AUC: 84.7916
- macro EER: 0.2028
- scoring: V0: mean over 4 taps of top5 mean; V1-V5: mean over 4 groups of 0.5*mean_tap(top5 mean)+0.5*max_tap(top5 mean).
- original top-k results: results_legacy_topk/

| scene | AUC | EER |
|---|---:|---:|
| bike | 98.0516 | 0.0420 |
| crossroads | 58.8085 | 0.4290 |
| farmland | 99.4230 | 0.0529 |
| highway | 93.1645 | 0.2040 |
| railway | 96.7101 | 0.0995 |
| solar | 87.1368 | 0.1779 |
| vehicle | 60.2465 | 0.4145 |
