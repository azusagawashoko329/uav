# comparison record schema (v1)

> 2026-11-04：新增可选的 `efficiency` 对象（轻量化协议 v2）；`schema_version` 仍为 `1`，旧记录无需改动。

每个对比条目是一个目录 `comparison/<dataset>/<Method>/<variant>/`，其中 `record.json` 是唯一机器入口；`<Method>` 保留官方方法名的大小写和标点（`adVAE`、`GANomaly`、`Future Frame`、`ST Dissoc.`、`MADViT`、`LightViT-AD`），我的方法为小写 `ours`。JSON Schema 见 [schema.json](schema.json)；`check_comparisons.py` 做不依赖第三方库的等价校验。

## 必填字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | int | 固定 `1` |
| `id` | str | 必须等于 `<dataset>-<Method>-<variant>`；目录路径为 `<dataset>/<Method>/<variant>` |
| `dataset` | enum | `da` / `uit` |
| `method` | str | 官方方法名，保留大小写/标点（`adVAE`、`GANomaly`、`Future Frame`、`ST Dissoc.`、`MADViT`、`LightViT-AD`）；我的方法为 `ours` |
| `method_display` | str | 展示名，如 `Ours` / `MADViT` |
| `year` | int/null | 方法年份；未知填 `null` |
| `variant` | str | 方法内变体；单变体用 `default` |
| `status` | enum | `planned` / `running` / `completed` / `failed` / `cancelled` / `blocked` / `unknown` |
| `source` | object | 来源（内部实验或外部仓库） |
| `protocol` | object/null | 统一协议；`completed` 时必须有 |
| `results` | object/null | 结果摘要；`completed` 时必须有 |
| `notes` | str | 备注；planned 条目应说明等待什么 |

## source

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | enum | `internal`（ours，指向 `experiments/`）或 `external`（第三方仓库） |
| `url` | str/null | 外部仓库 URL；completed external 必须非空 |
| `commit` | str/null | 外部运行的完整 commit SHA；completed 时必须非空 |
| `local_path` | str/null | 本地 clone 路径；本机默认 `/data1/data8tb/liangyuxuan/Comparison method/<Method>/`（也可用条目内 `upstream/`） |
| `experiment` | str/null | internal 时指向 `experiments/...` 原始实验目录 |
| `run_id` | str/null | internal 时指向原始 `record.json` 的 run |
| `license` | str/null | 外部仓库 license / 使用限制 |
| `notes` | str | 来源说明 |

规则：

- `type=internal`：`experiment` 必须存在，且其中 `record.json` 的 `runs` 包含 `run_id`；`commit` 必须与原始实验的 `source_commit` 一致。
- `type=external`：`planned` 时 `url` 可为 `null`；`running/completed` 时 `url` 与 `commit` 必须非空。
- 外部代码不要直接复制进 `model/`；适配脚本放在该条目的 `scripts/`，原始仓库放在 `source.local_path`。

## protocol

`completed` 时必填：

| 字段 | 类型 | 主对比默认 |
|---|---|---|
| `one_class` | bool | `true` |
| `train` | str | 官方 train 正常帧；per-scene merge |
| `test` | str | 每场景 test 全部 concat 后一次 AUC/EER |
| `metrics` | list | `["AUC","EER"]` |
| `seed` | int/null | `42` |
| `epochs` | int/null | 按方法/实验记录 |
| `input_size` | int/null | 256（若上游不同，写清） |
| `notes` | str | 任何协议偏差、输入/后处理差异 |

## results

`completed` 时必填：

| 字段 | 类型 | 说明 |
|---|---|---|
| `macro_auc` | number | 七场景（或声明的场景集）macro AUC |
| `macro_eer` | number | macro EER |
| `per_scene` | object | `{scene: {auc, eer}}` |
| `summary` | str | 相对该条目的 `results/summary.json` |
| `source_results` | str/null | internal 时指向原始实验的 `results/` |

`check_comparisons.py` 会核对 internal 条目的 `macro_auc` / `per_scene` 与原始实验 `results/summary.json` 一致。

## efficiency（可选，轻量化协议 v2）

用于把轻量化测量挂到同一条 `record.json` 上。**没有按 v2 测就留空；不要填猜测值。** `schema_version` 仍为 `1`，该对象是向后兼容的可选扩展；出现时必须包含下列全部键，未知值填 `null` / `[]`，不允许缺键。

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | enum | `measured` / `partial/unverified` / `paper-only` / `blocked`；定义见 [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) §9 |
| `input_hwt` | str | `HxWxT`，例如 `256x256x1`、`256x256x16`；通道数写 Notes |
| `code_commit` | str/null | 实际运行的完整 commit SHA；`measured` 时必须非空 |
| `weights_state` | str | `scratch` / `pretrained` / `DA-trained` / `architecture-init` / 说明字符串 |
| `da_adapted` | bool | 是否为 DA 做了输入/帧数/预处理适配 |
| `auc_source` | str | AUC 来源，例如 `ANDT Table VI (DA-adapted)` 或实验 ID；不能只写“paper” |
| `profiling_config` | str | GPU UUID、精度、batch、warmup/iters/rounds、同步方式 |
| `inference_params_m` | number/null | 推理图参数（含冻结教师，不含 training-only D） |
| `training_trainable_params_m` | number/null | one-class 训练更新的全部参数（含 D，不含冻结教师） |
| `training_only_params_m` | number/null | 只训练使用的参数（D）；应是 training trainable 的子集 |
| `frozen_inference_params_m` | number/null | 推理参数中的冻结部分（如教师），仅作透明拆分 |
| `flops_g_per_output_frame` | number/null | 一个有效输出分数的 GFLOPs；`1 MAC=2 FLOPs` |
| `flops_window_g` | number/null | 一个窗口的 GFLOPs；单帧方法等于 per-output 值 |
| `flops_verified` | bool | 是否通过算子计数器 + profiler + 手工公式交叉检查 |
| `flops_counter` | str/null | 计数器/工具版本与覆盖说明 |
| `flops_ops_covered` | list[str] | 已覆盖算子，如 `["Conv2d","ConvTranspose2d","Linear","MatMul","Attention-QK","Attention-AV"]` |
| `flops_ops_missing` | list[str] | 漏计/无法核验的算子；非空时不能标 `measured` |
| `latency_bs1_p50_ms_per_output_frame` | number/null | BS1 P50，按输出分数摊销 |
| `latency_bs1_mean_ms_per_output_frame` | number/null | BS1 Mean；`FPS@BS1` 用它反算 |
| `latency_bs1_p95_ms_per_output_frame` | number/null | BS1 P95 |
| `fps_bs1` | number/null | `1000 / mean_latency_ms`（输出分数/秒） |
| `fps_bs16` | number/null | `16·K·500 / median(round_time_s)`（输出分数/秒） |
| `input_frames_t` | int/null | 窗口输入帧数 `T` |
| `effective_output_scores_k` | int/null | 每个窗口的有效输出分数数 `K`，不得默认等于 `T` |
| `window_stride` | int/null | 窗口滑动步长（原始帧） |
| `window_latency_ms` | number/null | 一个窗口的 BS1 时延 |
| `fps_independent_window` | number/null | 独立窗口吞吐（输出分数/秒） |
| `fps_stream` | number/null | 真实视频流吞吐（原始帧/秒，按 `stride`） |
| `score_coverage` | number/null | `K/stride`；小于 1 表示有帧没有分数 |
| `benchmark_source` | str/null | 新结果 JSON 路径，例如 `comparison/da/benchmark_v2_ours.json` |
| `notes` | str | 口径偏差、漏计说明、与 AUC 配置的差异 |

校验规则（`check_comparisons.py` 会执行）：

- `status=measured` 时：`flops_verified=true`、`flops_ops_missing=[]`、`code_commit` 为完整 SHA、参数量三列非空、`input_hwt`/`weights_state`/`auc_source`/`profiling_config` 非空；`input_frames_t`/`effective_output_scores_k`/`window_stride` 必须非空；`T>1` 时 `window_latency_ms`/`flops_window_g`/`fps_independent_window`/`fps_stream`/`score_coverage` 也必须非空。
- `training_only_params_m <= training_trainable_params_m`；`frozen_inference_params_m <= inference_params_m`。
- `benchmark_source` 非空时，路径必须存在。
- 旧 `total_params_m` / `trainable_params_m` 是 v1 字段，**不要写进 efficiency**；新测量用三列新字段。旧 benchmark JSON 原样保留。

## 目录最小集

- `completed` 条目：`README.md`、`record.json`、`results/summary.json`、`results/summary.md`；
- `planned` 条目：`record.json` 即可；
- 外部方法代码/适配器：`scripts/`；本地 fork：`source.local_path`（按 license 决定是否入 Git）。
