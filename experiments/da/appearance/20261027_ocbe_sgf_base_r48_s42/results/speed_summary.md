# SGF-48 训练权重速度复核

变体：**sgf_base**；RTX 5880 Ada、FP32、GPU 常驻教师形状输入；不包含教师和预处理。所有速度使用训练完成的各场景 checkpoint，Q 使用 fused inference projection。

| 场景 | core latency batch1 (ms) | core FPS batch1 | score latency batch1 (ms) | score FPS batch1 | core latency batch16 (ms) | core FPS batch16 |
|---|---:|---:|---:|---:|---:|---:|
| bike | 1.7074 | 585.7 | 3.2918 | 303.8 | 2.6329 | 6077.1 |
| crossroads | 1.7072 | 585.8 | 3.2238 | 310.2 | 2.6766 | 5977.8 |
| farmland | 1.6643 | 600.8 | 3.2484 | 307.8 | 2.6511 | 6035.3 |
| highway | 1.6889 | 592.1 | 3.1954 | 312.9 | 2.6774 | 5975.9 |
| railway | 1.6612 | 602.0 | 3.2389 | 308.7 | 2.6571 | 6021.6 |
| solar | 1.6330 | 612.4 | 3.2497 | 307.7 | 2.6722 | 5987.6 |
| vehicle | 1.6373 | 610.8 | 3.2393 | 308.7 | 2.6681 | 5996.8 |

| 平均 | core latency | core FPS | score latency | score FPS |
|---|---:|---:|---:|---:|
| batch=1 | 1.6713 | 598.5 | 3.2410 | 308.6 |
| batch=16 | 2.6622 | 6010.3 | — | — |

融合和缓存校验均通过；fused code 最大绝对误差低于 `1.2e-7`，fused score 最大绝对误差为 `0`。
