# v1 轻量化测量归档（只读）

这些文件记录的是 **v1 口径**（`total_params_m`/`trainable_params_m` 混用、FLOPs 仅 `torch.profiler`、MemAE `÷16` 未核实、MKD 仅最终层距离）。从 2026-11-04 起：

- **不得修改、覆盖或删除**这些文件；
- 不得把它们作为 v2 正式对比结果；
- 修正后的数据写入对应方法/变体目录的 `benchmark_v2.json`（例如 `comparison/da/GANomaly/default/benchmark_v2.json`），并在其中写 `protocol_version: 2`、`supersedes`、`measurement_status`、`flops_verified`、`flops_ops_covered/missing`。

哈希清单见 [LEGACY_v1.sha256](LEGACY_v1.sha256)。在仓库根目录验证：

```bash
sha256sum -c comparison/da/LEGACY_v1.sha256
```

| 文件 | 大小 (bytes) | SHA-256 | 说明 |
|---|---:|---|---|
| `comparison/da/ours/legacy_v1/benchmark_ours.json` | 5323 | `1afd866ad745f8d7e3c6cc54078486df57d7a41046970099ae820c5c195df591` | 六个 Ours 变体；v1 profiler-only |
| `comparison/da/ours/legacy_v1/benchmark_ours.log` | 6799 | `4043d98a33e97bc8bbcc524a6104f05998edd2506496ff40f64300dd0e24d8b5` | 同上原始日志 |
| `comparison/da/GANomaly/default/legacy_v1/benchmark_ganomaly.json` | 841 | `666f690123b0b8cffeb66a86205db4cf33d69673c2d8ab645391ecc95e333590` | G 推理/ G+D 训练参数混列 |
| `comparison/da/GANomaly/default/legacy_v1/benchmark_ganomaly.log` | 1183 | `9a23255a8c1cf2280e41bd8875fb132298f88ab9fe885e6db54aa0894d881982` | 同上原始日志 |
| `comparison/da/MemAE/default/legacy_v1/benchmark_memae.json` | 787 | `d655a41d9deb22c01ce8f4e33acf5c0899cde61424ec345db49542c657a32df0` | `÷16` 未核实 |
| `comparison/da/MemAE/default/legacy_v1/benchmark_memae.log` | 1148 | `10b511ebc8ed022ef7883055784b17b55cc69943aa451ce1c5016d6611cf7310` | 同上原始日志 |
| `comparison/da/MKD/default/legacy_v1/benchmark_mkd.json` | 825 | `76e2ed6655db172dc24f6db34a28900a78054967b848c67ca1ba1217500cd53d` | 仅最终层距离 |
| `comparison/da/MKD/default/legacy_v1/benchmark_mkd.log` | 1164 | `2a05136801e8f43e09fdb935fa6602c2453184b9f4c6374e6c072fd53c4b2632` | 同上原始日志 |
| `experiments/da/audit/20261026_lightweight_profile_fp32_s42/results/summary.json` | 116000 | `ffcbc78083358f3561f13ef24a9066faa83e306f6243f2fe0cbd5936bad37231` | 旧轻量化审计 |
| `experiments/da/audit/20261026_lightweight_profile_fp32_s42/results/summary.csv` | 41414 | `b133fb3c4be310fb353d37a63c711a4b0b04e25d3050ee7d550113e291370c64` | 旧审计表格 |
| `experiments/da/audit/20261026_lightweight_profile_fp32_s42/results/summary.md` | 18884 | `c3dca1ae8870c5d5dad91f6b86987ee5ebb00b53fdfb77ee86be51a3169eff8e` | 旧审计报告 |

## 已知不再正式的数字

- GANomaly `10.952 G / 13.357 ms / 74.9 FPS / 622.7 FPS@BS16`：FLOPs 未做算子覆盖核验；参数列 `total_params_m=143.950` / `trainable_params_m=188.690` 有列名歧义，v2 改为 `inference_params_m=143.950`、`training_trainable_params_m=188.690`、`training_only_params_m=44.739`。
- MemAE `0.066 G / 0.437 ms / 2287 FPS`：按 `K=T=16` 摊销；v2 已核实原测试为 clip 级误差，`K=1, stride=1`，旧 `÷16` 废弃。
- MKD `10.477 G / 2.273 ms`：只算最终层距离；v2 已按原方法 taps 6/9/12 多层评分，P50 = 2.495ms。
- Ours V0 `90.205 G / 17.800 ms`、V5 `46.572 G / 19.638 ms`：profiler-only；v2 同图复核为 `95.107G / 49.106G`（与旧审计一致），v1 漏 student native MHA。
- ANDT `70.347M`：参数结构可数，但 decoder 只支持 1 帧，FLOPs/latency `blocked`。
- adVAE `0.329M`：表格版，不是 DA 视频模型，`blocked`。
- DwCVAE/CVAE/DCVAE/SCVAE/CLSTMVAE 的用户表速度：不同硬件/batch，`paper-only`，不可与本机 5880 Ada 实测混排。
