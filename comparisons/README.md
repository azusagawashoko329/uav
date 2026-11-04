# 方法对比工作区（comparisons）

本目录是**方法对比的展示/编排层**，与 `experiments/` 的原始实验记录分开：

- `experiments/<dataset>/<family>/...` 仍是原始实验的唯一记录（README / config / record / artifacts / logs / results）；
- `comparisons/` 只回答“在统一 One-Class 协议下比较哪些方法/变体”，条目指向原始实验或外部方法仓库；
- **不移动、不删除**原始实验目录；原始实验的 record / artifacts / checkpoint 仍以 `experiments/` 为准。

## 布局

```
comparisons/
  README.md                 # 本文件
  SCHEMA.md                 # record.json 字段规范（人类可读）
  schema.json               # record.json 的 JSON Schema（机器可读）
  check_comparisons.py      # 校验 record + CATALOG 一致性
  build_catalog.py          # 由 record 生成 CATALOG.csv / INDEX.md
  CATALOG.csv               # 机器可读索引
  INDEX.md                  # 人类可读索引
  da/
    da-ours-<variant>/      # 我的方法，六个 DA 外观蒸馏变体
      README.md
      record.json
      results/summary.{json,md}
    da-<method>-<variant>/  # 外部方法；URL 到位后建目录
      record.json
      upstream/             # 外部仓库本地 fork（可选，按 license 决定是否入 Git）
      scripts/              # 适配脚本
      results/
  uit/
    README.md               # UIT 对比空间；条目按同一 schema 添加
```

## 命名

- 目录名：`<dataset>-<method>-<variant>`，例如 `da-ours-rd_s4_t12_gl192_h512`、`da-madvit-full`。
- `dataset`：`da` 或 `uit`。
- `method`：小写 slug，如 `ours`、`madvit`、`astt`、`lightvit-ad`。
- `variant`：方法内变体；我的方法直接用 `experiments/VARIANTS.md` 的 canonical ID（`rd_s12_t4_h512` 等）；单变体方法用 `default`。
- 详细字段见 [SCHEMA.md](SCHEMA.md)；机器校验以 [schema.json](schema.json) 和 `check_comparisons.py` 为准。

## 状态

`planned` / `running` / `completed` / `failed` / `cancelled` / `blocked` / `unknown`。

- `planned`：已登记方法/变体，等待 URL、代码或运行环境；
- `completed`：有可核对的结果摘要（`results/summary.json`），且 `results.macro_auc` 存在；
- 外部方法的 `source.url` / `source.commit` / `source.local_path` 在状态进入 `running/completed` 前必须填写。

## 统一协议

主对比默认要求：

- one-class：训练只用官方 train 正常帧；
- 每个场景独立模型，训练帧来自 `sequence*/train/*/*.jpg`；
- 测试：该场景所有 test 视频逐帧分数 concat 后计算一次 global AUC/EER；
- 记录 `seed`、`epochs`、`input_size`、指标与任何协议偏差；
- 外部方法如果原生协议不同（例如需要光流、输入分辨率不同），必须在 `protocol.notes` 写清，不能静默改口径。

## 添加外部方法

1. 用户提供上游 URL 后，clone 到该条目的 `source.local_path`（建议 `upstream/`），记录 `source.url`、`source.commit`、`source.license`；
2. 在该条目写 `scripts/` 适配器，把上游训练/评测接到 DA 的 per-scene 官方 train/test 上；
3. 运行并产出 `results/summary.json`（至少含 macro AUC/EER 与 per-scene AUC/EER）；
4. 更新 `record.json`，把 `status` 改为 `completed`，填 `results`；
5. 运行：

```bash
python comparisons/build_catalog.py
python comparisons/check_comparisons.py
```

## Ours

六个 DA 变体的对比条目在 `comparisons/da/da-ours-rd_*`：

- V0 `rd_s12_t4_h512`
- V1 `rd_s4_t12_h512`
- V2 `rd_s4_t12_gl768_h512`
- V3 `rd_s4_t12_gl192_h512`
- V4 `rd_s4_t12_gl192_h128`
- V5 `rd_s4_t12_gl192_h128_qr2`

它们指向 `experiments/da/appearance/20261103_variant_matrix_*` 的原始 30 epoch / seed42 实验；对比条目只复制 `results/summary.{json,md}`，完整逐帧分数与 checkpoint 仍以原始实验目录为准。数据来源与泄漏审计见 [20261103_variant_matrix_data_audit.md](../experiments/da/appearance/20261103_variant_matrix_data_audit.md)。

## 当前 DA 对比矩阵

`comparisons/INDEX.md` 会列出：

- `ours` 六个变体（completed）；
- 用户表格中的外部方法（planned，等待 URL）：adVAE、GANomaly、MemAE、MKD、ANDT、Future Frame、ST Dissoc.、MLEP、ASTT、DAD-FSM、HSTforU、MADViT（Full / Multi-layer / Last-layer）、LightViT-AD。

外部方法的年份/星号/颜色按用户表格登记；`record.json` 暂未编码颜色分类，需要时再加 `category` 字段。
