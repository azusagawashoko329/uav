# DA Student12 Plain12 vs GL12 summary

## Scope

- Plain12: `20261102_student12_plain_retry4_s42`, 7 scenes, seed 42.
- GL12: `20261103_student12_gl768_retry_failed3_s42`, 3 previously failed scenes, seed 42.
- Same teacher taps `[5,7,9,11]`, Dense512 OCBE, 30 epochs, batch 64. Metrics are scene-level concatenated test-frame AUC and EER.

## Matched three-scene comparison

| Scene | Plain AUC | GL AUC | Δ AUC | Plain EER | GL EER | Δ EER |
|---|---:|---:|---:|---:|---:|---:|
| railway | 96.389 | 97.969 | +1.581 | 0.1032 | 0.0737 | -0.0295 |
| solar | 86.999 | 87.281 | +0.282 | 0.1834 | 0.1797 | -0.0036 |
| vehicle | 64.783 | 60.912 | -3.871 | 0.3554 | 0.3922 | +0.0367 |

Matched three-scene macro mean: Plain AUC **82.724**, GL AUC **82.054** (GL − Plain **-0.669**); Plain EER **0.2140**, GL EER **0.2152** (GL − Plain **+0.0012**).

Across all seven Plain12 scenes, macro mean is AUC **85.551** and EER **0.1948**. The GL run covers only the three rerun scenes, so its three-scene mean should not be compared with the seven-scene Plain12 mean.

## Observations

- GL12 improves railway AUC by 1.581 points and solar AUC by 0.282 points on the matched scenes.
- GL12 lowers vehicle AUC by -3.871 points and has higher vehicle EER.
- On the matched three scenes, GL12 has a slightly lower macro AUC and higher macro EER; this is a seed-42 observation from one run, not a significance claim.

See [summary.csv](summary.csv) for exact values and the per-scene result JSON files for provenance.
