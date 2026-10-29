# svd_rank64 seed43 速度复核

RTX 5880 Ada、FP32、GPU 常驻特征；不包含 teacher、预处理和 H2D。

| batch | core latency | core FPS | score latency | score FPS |
|---|---:|---:|---:|---:|
| 1 | 1.6526 ms | 605.2 | 3.1954 ms | 313.0 |
| 16 | 2.7120 ms | 5899.7 | 3.7317 ms | 4287.6 |
