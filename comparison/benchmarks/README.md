# 轻量化 benchmark 脚本规范（v2）

现有 `bench_ours.py`、`bench_external.py` 是 **v1 legacy**：它们写出的
`comparison/da/benchmark_*.json` 已被冻结（见 [../da/LEGACY_v1.md](../da/LEGACY_v1.md)），
**不得再用它们产出正式 v2 数字**。v2 重测时按本文件改造脚本或新增
`bench_v2_*.py`，输出一律写 `comparison/da/benchmark_v2_<method>.json`。

## 1. 输出文件命名

```
comparison/da/benchmark_v2_ours.json
comparison/da/benchmark_v2_ganomaly.json
comparison/da/benchmark_v2_memae.json
comparison/da/benchmark_v2_mkd.json
comparison/da/benchmark_v2_<method>.json
```

每个文件必须含：

```json
{
  "protocol_version": 2,
  "measurement_status": "measured|partial/unverified|paper-only|blocked",
  "supersedes": "comparison/da/benchmark_<method>.json",
  "method": "<官方方法名>",
  "variant": "<variant>",
  "code_commit": "<完整 40 位 SHA>",
  "input_hwt": "HxWxT",
  "weights_state": "scratch|pretrained|DA-trained|architecture-init|...",
  "da_adapted": true,
  "auc_source": "<AUC 来源；与效率配置分开>",
  "profiling_config": "<GPU UUID/精度/BS/warmup/iters/rounds/同步方式>",
  "gpu": "...", "torch": "...", "cuda": "...", "cudnn": "...",
  "warmup": 50, "iterations": 500, "rounds": 3,
  "inference_params_m": 0.0,
  "training_trainable_params_m": 0.0,
  "training_only_params_m": 0.0,
  "frozen_inference_params_m": 0.0,
  "flops_g_per_output_frame": 0.0,
  "flops_window_g": 0.0,
  "flops_verified": false,
  "flops_counter": "<工具/版本/覆盖说明>",
  "flops_ops_covered": ["Conv2d", "Linear", "MatMul", "Attention-QK", "Attention-AV"],
  "flops_ops_missing": [],
  "flops_breakdown": [
    {"module": "...", "op": "Conv2d", "input_shape": [...], "output_shape": [...], "macs": 0}
  ],
  "input_frames_t": 1,
  "effective_output_scores_k": 1,
  "window_stride": 1,
  "window_latency_ms": 0.0,
  "latency_bs1_p50_ms_per_output_frame": 0.0,
  "latency_bs1_mean_ms_per_output_frame": 0.0,
  "latency_bs1_p95_ms_per_output_frame": 0.0,
  "fps_bs1": 0.0,
  "fps_bs16": 0.0,
  "fps_independent_window": 0.0,
  "fps_stream": 0.0,
  "score_coverage": 1.0,
  "bs1_round_times_s": [[...], [...], [...]],
  "bs16_round_times_s": [...],
  "benchmark_source": "comparison/da/benchmark_v2_<method>.json",
  "notes": "<与 AUC 配置的差异、漏计、假设>"
}
```

## 2. 旧字段 → v2 字段映射

| v1 字段 | v2 字段 | 处理 |
|---|---|---|
| `total_params_m` | `inference_params_m` | 推理图参数；GANomaly 只含 G |
| `trainable_params_m` | `training_trainable_params_m` | 训练更新参数；GANomaly 含 G+D |
| `training_only_params_m` | `training_only_params_m` | 保留；D 等只训练模块 |
| — | `frozen_inference_params_m` | 教师/冻结主干拆分，仅透明用 |
| `flops_g` | `flops_g_per_output_frame` | 必须来自三方核验；同时保留 `flops_window_g` |
| — | `flops_verified` | 未做算子覆盖核验必须 `false` |
| — | `flops_ops_covered` / `flops_ops_missing` | 逐项列出 |
| `latency_bs1_p50_ms_per_output_frame` | 同名 | 保留；另存 `window_latency_ms` |
| `fps_bs1` | 同名 | 必须由 **Mean** 反算，不是 P50 |
| `fps_bs16` | 同名 | 按输出分数/秒；另报 `fps_stream` |

## 3. 每个脚本的改造点

### 3.1 `bench_ours.py`

- 参数量改为三列：`inference_params_m = teacher + OCBE + student`，`training_trainable_params_m = OCBE + student`，`training_only_params_m = 0`，`frozen_inference_params_m = teacher`。
- FLOPs 不能只跑 `torch.profiler`：加模块 hook 计数器，覆盖 ViT 的 Linear/MatMul/Attention QK/AV 和 patch embedding；用同一图跑 profiler 交叉检查；差异 ≤1% 才 `flops_verified=true`。
- 固定 teacher tap 配置：测哪个变体就用哪个 tap 集合，不能先建 12 taps 再选 4 个却把 12-tap 计算算进去；同时记录 `teacher_taps`。
- 输入 `256x256x1`、`T=1`、`K=1`、`stride=1`；输出 `window_latency_ms` 与 per-output 值。
- 权重状态写 `architecture-init`（效率测结构）；AUC 另引 `experiments/da/appearance/20261103_variant_matrix_*`，不要暗示同一权重。
- 结果写 `benchmark_v2_ours.json`；不得覆盖 `benchmark_ours.json`。

### 3.2 `bench_external.py`

- GANomaly：推理只走 G，`inference_params_m=G`；`training_trainable_params_m=G+D`；`training_only_params_m=D`；FLOPs 覆盖 G 的 Conv2d/ConvTranspose2d/Linear。
- MemAE：先核对原测试的 `K`（clip 级 vs 逐帧），不要把 `K=T=16` 当默认；`flops_ops_covered` 必须含 Conv3d/ConvTranspose3d；未核实前 `measurement_status=partial/unverified`。
- MKD：按原方法多层特征评分实现，不能只取最终层距离；输入保持原论文配置（128×128）或明确标 `da_adapted=true`；FLOPs 覆盖 source/cloner 的 Conv2d/Linear。
- 外部方法统一从本地 clone 的完整 commit 取 `code_commit`，并写 `source.local_path`。
- 结果写 `benchmark_v2_<method>.json`；不得覆盖 v1 文件。

## 4. 脚本级验收

- [ ] 同一输入/同一代码 commit/同一分支跑三次，FLOPs 一致，P50 波动在可解释范围；
- [ ] 算子计数器与 profiler 差异 ≤1%，否则 `flops_verified=false`；
- [ ] 参数量三列非空，且 `training_only <= training_trainable`、`frozen <= inference`；
- [ ] 时序方法 `T/K/stride` 写清，`K` 有代码/原测试依据；
- [ ] BS1 原始耗时和 BS16 三轮时间都保存；
- [ ] `Status` 与 `flops_verified` 一致：`measured` 必须 `flops_verified=true` 且 `flops_ops_missing=[]`；
- [ ] 输出文件名是 `benchmark_v2_*`，`supersedes` 指向对应 v1 文件。
