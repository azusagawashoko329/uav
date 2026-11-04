# 轻量化 benchmark 脚本规范（v2）

现有 `bench_ours.py`、`bench_external.py` 是 **v1 legacy**：它们写出的
`benchmark_*.json` 已冻结在 `<Method>/<variant>/legacy_v1/`（见 [../da/LEGACY_v1.md](../da/LEGACY_v1.md)），
**不得再用它们产出正式 v2 数字**。

v2 已实现：

- `flops_counter.py`：FlopCounterMode + native MHA 公式 + 手写模块公式 + profiler 共同子集 + 去重检查。
- `bench_v2_common.py`：50 warmup、BS1 3×500、BS16 3×500、JSON 输出。
- `bench_v2_ours.py`：六个 Ours 变体。
- `bench_v2_external.py`：GANomaly / MemAE / MKD。
- `bench_v2_andt.py`：ANDT adapted 6-frame（适配器在 `comparison/da/ANDT/default/scripts/`）。
- `bench_v2_extra.py`：LightViT-AD / MADViT Full / ST Dissoc. / ASTT(STE-TTE) / DAD-FSM / HSTforU。
- `bench_v2_madvit_variants.py`：MADViT Full / Multi-layer / Last-layer（论文 Sec. 4.4 的 scoring 变体）。
- `bench_tf1_extra.py`：Future Frame / MLEP（TF1.15，FLOPs 标 partial/unverified）。

结果写在对应方法/变体目录里：

```
comparison/da/ours/<variant>/benchmark_v2.json
comparison/da/ours/benchmark_v2_ours.json      # 六变体聚合
comparison/da/GANomaly/default/benchmark_v2.json
comparison/da/MemAE/default/benchmark_v2.json
comparison/da/MKD/default/benchmark_v2.json
comparison/da/ANDT/default/benchmark_v2.json   # adapted 6-frame
comparison/da/LightViT-AD/default/benchmark_v2.json
comparison/da/MADViT/Full/benchmark_v2.json
comparison/da/MADViT/Multi-layer/benchmark_v2.json
comparison/da/MADViT/Last-layer/benchmark_v2.json
comparison/da/ST Dissoc./default/benchmark_v2.json
comparison/da/ASTT/default/benchmark_v2.json
comparison/da/DAD-FSM/default/benchmark_v2.json
comparison/da/HSTforU/default/benchmark_v2.json
comparison/da/Future Frame/default/benchmark_v2.json
comparison/da/MLEP/default/benchmark_v2.json
```

## 1. 输出文件命名

- Ours：每个变体 `ours/<variant>/benchmark_v2.json`；聚合 `ours/benchmark_v2_ours.json`。
- 外部方法：`<Method>/default/benchmark_v2.json`。
- 外部方法日志与 JSON 同目录；Ours 分批日志在 `ours/logs/`；v1 归档在同方法目录的 `legacy_v1/`。

每个文件必须含：

```json
{
  "protocol_version": 2,
  "measurement_status": "measured|partial/unverified|paper-only|blocked",
  "supersedes": "comparison/da/<Method>/<variant>/legacy_v1/benchmark_<method>.json",
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
  "flops_common_subset_check": "<与 profiler 共同支持算子的差异；不是完整总数比较>",
  "flops_duplicate_check": "<module hook / matmul / SDPA 去重说明>",
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
  "score_density": 1.0,
  "frame_coverage": 1.0,
  "fps_stream_estimated": 0.0,
  "fps_stream_measured": null,
  "bs1_round_times_s": [[...], [...], [...]],
  "bs16_round_times_s": [...],
  "benchmark_source": "comparison/da/<Method>/<variant>/benchmark_v2.json",
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
| `flops_g` | `flops_g_per_output_frame` | 必须来自所有 MAC 算子的计数器/公式核验（profiler 只核验共同子集）；同时保留 `flops_window_g` |
| — | `flops_verified` | 未做算子覆盖核验必须 `false` |
| — | `flops_ops_covered` / `flops_ops_missing` | 逐项列出 |
| `latency_bs1_p50_ms_per_output_frame` | 同名 | 保留；另存 `window_latency_ms` |
| `fps_bs1` | 同名 | 必须由 **Mean** 反算，不是 P50 |
| `fps_bs16` | 同名 | 按输出分数/秒；另报 `fps_stream_estimated`（未做连续流测试不得写 measured） |

## 3. 每个脚本的改造点

### 3.1 `bench_ours.py`

- 参数量改为三列：`inference_params_m = teacher + OCBE + student`，`training_trainable_params_m = OCBE + student`，`training_only_params_m = 0`，`frozen_inference_params_m = teacher`。
- FLOPs 不能只跑 `torch.profiler`：加模块 hook 计数器，覆盖 ViT 的 Linear/MatMul/Attention QK/AV 和 patch embedding；**只对 profiler 共同支持的算子比较（≤1%）**，profiler 不支持的算子用独立公式核验；完整 FLOPs = 所有已验证 MAC 算子之和 ×2，不要求等于 profiler 总数；同时做 module hook / matmul / SDPA 去重检查。
- 固定 teacher tap 配置并记录 `teacher_taps`；**tap 是输出选择，不是计算裁剪**：V0 取 `[5,7,9,11]` 时仍必须前向 `0..11`，FLOPs 按实际前向层数计，不能按 4 个 tap × 单层算；可省的是未用 tap 的保存/后处理。
- 输入 `256x256x1`、`T=1`、`K=1`、`stride=1`；输出 `window_latency_ms` 与 per-output 值。
- 权重状态写 `architecture-init`（效率测结构）；AUC 另引 `experiments/da/appearance/20261103_variant_matrix_*`，不要暗示同一权重。
- 结果写 `ours/<variant>/benchmark_v2.json` 和 `ours/benchmark_v2_ours.json`；不得覆盖 `ours/legacy_v1/benchmark_ours.json`。

### 3.2 `bench_v2_andt.py`

- ANDT 必须标 `adapted`：发布代码 `Decoder.forward` 只支持 1 帧；适配器用 `de_dense(feat[:,0])` 预测下一帧。
- 输入 `(B,6,3,256,256)`，`T=6, K=1, stride=1`；不能把 adapted 数字当 official。
- `mlp_dim=4096`、`num_layers=2`、`num_heads=6`；参数量 70.347M。

### 3.3 `bench_external.py`

- GANomaly：推理只走 G，`inference_params_m=G`；`training_trainable_params_m=G+D`；`training_only_params_m=D`；FLOPs 覆盖 G 的 Conv2d/ConvTranspose2d/Linear。
- MemAE：先核对原测试的 `K`（clip 级 vs 逐帧），不要把 `K=T=16` 当默认；`flops_ops_covered` 必须含 Conv3d/ConvTranspose3d；同时记录 `score_density=K/stride` 与按分数映射计算的 `frame_coverage`，不要用 K/stride 冒充覆盖率；未核实前 `measurement_status=partial/unverified`。
- MKD：按原方法多层特征评分实现，不能只取最终层距离；输入保持原论文配置（128×128）或明确标 `da_adapted=true`；FLOPs 覆盖 source/cloner 的 Conv2d/Linear。
- 外部方法统一从本地 clone 的完整 commit 取 `code_commit`，并写 `source.local_path`。
- 结果写 `<Method>/default/benchmark_v2.json`；不得覆盖同目录 `legacy_v1/` 下的 v1 文件。

## 4. 脚本级验收

- [ ] 同一输入/同一代码 commit/同一分支跑三次，FLOPs 一致，P50 波动在可解释范围；
- [ ] 只对 profiler 共同支持的算子比较且差异 ≤1%；ConvTranspose/Conv3d/融合 Attention 用独立公式核验；完整 FLOPs 不要求等于 profiler 总数；
- [ ] 防重复计数：module hook、matmul 包装、SDPA 包装按 `op_id` 去重，无同一矩阵乘累计两次；
- [ ] 参数量三列非空，且 `training_only <= training_trainable`、`frozen <= inference`；
- [ ] 时序方法 `T/K/stride` 写清，`K` 有代码/原测试依据；`score_density=K/stride`，`frame_coverage` 按分数映射/重叠/融合单独计算（不能由 K/stride 推出）；没有连续流测试只能写 `fps_stream_estimated`；
- [ ] BS1 原始耗时和 BS16 三轮时间都保存；
- [ ] `Status` 与 `flops_verified` 一致：`measured` 必须 `flops_verified=true` 且 `flops_ops_missing=[]`；
- [ ] 输出路径是 `<Method>/<variant>/benchmark_v2.json`，`supersedes` 指向同方法 `legacy_v1/` 下的对应 v1 文件。
