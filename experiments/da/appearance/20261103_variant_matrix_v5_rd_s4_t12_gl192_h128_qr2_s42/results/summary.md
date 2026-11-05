# V5 `rd_s4_t12_gl192_h128_qr2`：DA 30 epoch（matched score）

- macro AUC: 84.6233
- macro EER: 0.1970
- scoring: V0: mean over 4 taps of top5 mean; V1-V5: mean over 4 groups of 0.5*mean_tap(top5 mean)+0.5*max_tap(top5 mean).
- original top-k results: results_legacy_topk/

| scene | AUC | EER |
|---|---:|---:|
| bike | 97.8152 | 0.0432 |
| crossroads | 60.4277 | 0.4345 |
| farmland | 99.5703 | 0.0485 |
| highway | 94.4218 | 0.1528 |
| railway | 92.1990 | 0.1229 |
| solar | 87.5511 | 0.1936 |
| vehicle | 60.3781 | 0.3837 |
