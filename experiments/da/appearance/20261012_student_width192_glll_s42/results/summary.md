# Global-local student width-192 experiment

The loss alignment and Global-Local topology are unchanged. The only model change is the student internal width: 768 to 192; each reconstruction head projects back to the unchanged 768-dimensional teacher target.

## Seven-scene result

| scene | width-192 AUC | width-768 AUC | delta (pp) | normal mean | abnormal mean | score gap | GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| highway | 95.020229 | 95.233562 | -0.213333 | 0.178096 | 0.245327 | 0.067231 | 0 |
| farmland | 99.327048 | 99.252818 | +0.074231 | 0.150804 | 0.268858 | 0.118054 | 0 |
| bike | 97.607091 | 97.706654 | -0.099562 | 0.118941 | 0.194229 | 0.075289 | 0 |
| solar | 82.057837 | 89.569245 | -7.511408 | 0.227640 | 0.252916 | 0.025276 | 1 |
| vehicle | 57.485141 | 58.442681 | -0.957540 | 0.231785 | 0.237775 | 0.005990 | 1 |
| crossroads | 61.253510 | 61.194955 | +0.058555 | 0.202712 | 0.207578 | 0.004866 | 1 |
| railway | 85.015898 | 92.014742 | -6.998844 | 0.272720 | 0.293664 | 0.020944 | 1 |
| **macro mean** | **82.538108** | **84.773522** | **-2.235415** | | | | |

## Parameter and scheduler checks

- Trainable parameters: 7,718,336; width-768 Global-Local comparison: 30,248,000; reduction: 74.48%.
- All seven scene score CSVs have the same frame and label keys as the width-768 comparison and matching split fingerprints.
- The scheduler recorded seven launches, seven zero-return finishes, and an all-finished event. GPU workers were reused on completion rather than assigned fixed scene slots.
- The Global-Local topology and loss alignment are unchanged; only the student internal width and output projection dimensions changed.

## Interpretation and limitations

The width-192 student loses 2.235415 percentage points in macro AUC relative to the 768-dimensional Global-Local student. The decrease is concentrated in Solar (-7.511408 pp) and Railway (-6.998844 pp); Highway, Bike and Vehicle decline by less than 1 pp, while Farmland and Crossroads are nearly unchanged. This is a single-seed width ablation and does not establish multi-seed variance. The benchmark was run on two GPUs, so deployment-device latency remains unmeasured.
