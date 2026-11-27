# UIT HAP-RD-Lite shared-Q vs per-Q

The two variants use the same UIT split, 45 epochs, LLLG+GATE decoder, PLDD scoring, and global test-frame aggregation. Per-Q adds 18 trainable scalars.

| seed | shared-Q AUC | per-Q AUC | ΔAUC (pp) | shared-Q EER | per-Q EER | ΔEER |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 75.6727 | 75.6215 | -0.0512 | 0.321552 | 0.322055 | +0.000503 |
| 43 | 75.5713 | 75.6312 | +0.0598 | 0.323360 | 0.321300 | -0.002059 |
| 44 | 75.6484 | 75.6609 | +0.0125 | 0.321444 | 0.321348 | -0.000096 |

| mean ± SD | 75.6308 ± 0.0529 | 75.6379 ± 0.0205 | +0.0070 | 0.322118 ± 0.001076 | 0.321568 ± 0.000422 | -0.000551 |

AUC is higher-is-better and EER is lower-is-better. The paired mean difference is small relative to the seed spread, so this comparison does not show a material per-Q gain.

Three paired seeds support descriptive comparison only, not significance testing.
