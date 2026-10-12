# Global-local student structural experiment

The loss alignment is unchanged: consecutive teacher layers are grouped in fours of three, and each student stage owns three reconstruction heads. The only model change is the student block topology: stage 0 uses global MHSA; stages 1–3 use 3x3 depthwise spatial mixing plus pointwise projection; all stages use a SiLU gated FFN.

## Seven-scene result

| scene | structural AUC | no-adapter AUC | delta (pp) | normal mean | abnormal mean | score gap | GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| highway | 95.233562 | 95.540873 | -0.307311 | 0.136891 | 0.204561 | 0.067671 | 0 |
| farmland | 99.252818 | 99.237646 | +0.015171 | 0.113844 | 0.221108 | 0.107264 | 0 |
| bike | 97.706654 | 97.706750 | -0.000096 | 0.078626 | 0.156888 | 0.078262 | 0 |
| solar | 89.569245 | 89.093590 | +0.475655 | 0.162026 | 0.198929 | 0.036903 | 1 |
| vehicle | 58.442681 | 53.980052 | +4.462630 | 0.203528 | 0.208438 | 0.004911 | 1 |
| crossroads | 61.194955 | 59.429594 | +1.765362 | 0.174956 | 0.180509 | 0.005553 | 1 |
| railway | 92.014742 | 92.665125 | -0.650383 | 0.184012 | 0.222932 | 0.038920 | 1 |
| **macro mean** | **84.773522** | **83.950518** | **+0.823004** | | | | |

## Parameter and scheduler checks

- Trainable parameters: 30,248,000; direct no-adapter comparison: 40,258,880; reduction: 24.87%.
- All seven scene score CSVs have the same frame and label keys as the direct no-adapter comparison and matching split fingerprints.
- The scheduler recorded seven launches, seven zero-return finishes, and an all-finished event. GPU workers were reused on completion rather than assigned fixed scene slots.
- The structural model has one global mixer followed by three local mixers. This is an operator/topology change, not an input-width-only ablation.

## Interpretation and limitations

The structural student improves the direct four-block no-adapter mean by 0.823 percentage points in this seed while using 24.87% fewer trainable OCBE+student parameters. Gains are concentrated in vehicle and crossroads; highway and railway decline. This is a single-seed result and does not establish a stable improvement. The benchmark was run on two GPUs, so deployment-device latency remains unmeasured.

## Inference latency microbenchmark

The following measurements use B=1 and 256 patch tokens, 50 warmup iterations and 200 timed iterations. They are GPU measurements, not edge-device measurements.

| device | student baseline ms | student structural ms | student delta | OCBE+student baseline ms | OCBE+student structural ms | pipeline delta | teacher ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| NVIDIA RTX 5880 Ada Generation | 1.6741 | 1.6640 | +0.60% | 2.0412 | 2.0943 | -2.60% | 8.0777 |
| NVIDIA RTX A6000 | 1.5914 | 1.3681 | +14.03% | 1.8891 | 1.6566 | +12.31% | 8.5154 |

The structural block is faster on the RTX A6000 but not on the RTX 5880 Ada once OCBE is included. Therefore this experiment does not establish a hardware-independent speedup; the real-time claim requires a benchmark on the target UAV processor and the full image-to-score path.
