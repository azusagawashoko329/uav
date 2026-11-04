# comparison record schema (v1)

每个对比条目是一个目录 `comparisons/<dataset>/<dataset>-<method>-<variant>/`，其中 `record.json` 是唯一机器入口。JSON Schema 见 [schema.json](schema.json)；`check_comparisons.py` 做不依赖第三方库的等价校验。

## 必填字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | int | 固定 `1` |
| `id` | str | 必须等于目录名，且等于 `<dataset>-<method>-<variant>` |
| `dataset` | enum | `da` / `uit` |
| `method` | str | 小写 slug，如 `ours` / `madvit` / `lightvit-ad` |
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
| `local_path` | str/null | 本地 clone 路径（建议 `upstream/`） |
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

## 目录最小集

- `completed` 条目：`README.md`、`record.json`、`results/summary.json`、`results/summary.md`；
- `planned` 条目：`record.json` 即可；
- 外部方法代码/适配器：`scripts/`；本地 fork：`source.local_path`（按 license 决定是否入 Git）。
