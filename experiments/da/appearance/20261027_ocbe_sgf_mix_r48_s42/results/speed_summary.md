# SGF-48 训练权重速度复核

变体：**sgf_mix**；RTX 5880 Ada、FP32、GPU 常驻教师形状输入；不包含教师和预处理。所有速度使用训练完成的各场景 checkpoint，Q 使用 fused inference projection。

| 场景 | core latency batch1 (ms) | core FPS batch1 | score latency batch1 (ms) | score FPS batch1 | core latency batch16 (ms) | core FPS batch16 |
|---|---:|---:|---:|---:|---:|---:|
| bike | 1.6972 | 589.2 | 3.1696 | 315.5 | 2.6444 | 6050.6 |
| crossroads | 1.7230 | 580.4 | 3.2021 | 312.3 | 2.6834 | 5962.6 |
| farmland | 1.7070 | 585.8 | 3.1679 | 315.7 | 2.6729 | 5986.1 |
| highway | 1.7180 | 582.1 | 3.1886 | 313.6 | 2.6797 | 5970.9 |
| railway | 1.7808 | 561.6 | 3.2687 | 305.9 | 2.6737 | 5984.2 |
| solar | 1.7755 | 563.2 | 3.2024 | 312.3 | 2.6832 | 5963.0 |
| vehicle | 1.7129 | 583.8 | 3.1680 | 315.7 | 2.6969 | 5932.7 |

| 平均 | core latency | core FPS | score latency | score FPS |
|---|---:|---:|---:|---:|
| batch=1 | 1.7306 | 578.0 | 3.1953 | 313.0 |
| batch=16 | 2.6763 | 5978.6 | — | — |

融合和缓存校验均通过；fused code 最大绝对误差低于 `1.2e-7`，fused score 最大绝对误差为 `0`。
