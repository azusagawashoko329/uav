# V2 `rd_s4_t12_gl768_h512`：DA 30 epoch（matched score）

- macro AUC: 85.2147
- macro EER: 0.1965
- scoring: V0: mean over 4 taps of top5 mean; V1-V5: mean over 4 groups of 0.5*mean_tap(top5 mean)+0.5*max_tap(top5 mean).
- original top-k results: results_legacy_topk/

| scene | AUC | EER |
|---|---:|---:|
| bike | 97.7002 | 0.0459 |
| crossroads | 61.5545 | 0.4332 |
| farmland | 99.4576 | 0.0550 |
| highway | 94.9450 | 0.1593 |
| railway | 95.4762 | 0.1069 |
| solar | 89.7237 | 0.1683 |
| vehicle | 57.6453 | 0.4066 |
