# 方法对比工作区（comparison）

本目录是**方法对比的展示/编排层**，与 `experiments/` 的原始实验记录分开：

- `experiments/<dataset>/<family>/...` 仍是原始实验的唯一记录（README / config / record / artifacts / logs / results）；
- `comparison/` 只回答“在统一 One-Class 协议下比较哪些方法/变体”，条目指向原始实验或外部方法仓库；
- **不移动、不删除**原始实验目录；原始实验的 record / artifacts / checkpoint 仍以 `experiments/` 为准。

## 布局

```
comparison/
  README.md                 # 本文件
  SCHEMA.md                 # record.json 字段规范（人类可读）
  schema.json               # record.json 的 JSON Schema（机器可读）
  check_comparisons.py      # 校验 record + CATALOG 一致性
  build_catalog.py          # 由 record 生成 CATALOG.csv / INDEX.md
  CATALOG.csv               # 机器可读索引
  INDEX.md                  # 人类可读索引
  da/
    ours/
      <variant>/            # 我的方法，六个 DA 外观蒸馏变体
        README.md
        record.json
        results/summary.{json,md}
        benchmark_v2.json   # 该变体的 v2 轻量化结果
      benchmark_v2_ours.json # 六变体聚合
      logs/                 # 分批运行日志
      legacy_v1/            # 冻结的 v1 结果
    <Method>/
      <variant>/            # 外部方法；目录名保留官方大小写/标点
        record.json
        benchmark_v2.json   # 该方法的 v2 轻量化结果
        benchmark_v2.log
        legacy_v1/          # 冻结的 v1 结果
        upstream/           # 可选；默认 clone 到 /data1/data8tb/liangyuxuan/Comparison method/<Method>/
        scripts/            # 适配脚本
        results/
  uit/
    <Method>/
      <variant>/            # UIT 对比空间；条目按同一 schema 添加
```

## 命名

- 目录路径：`comparison/<dataset>/<Method>/<variant>/`，例如 `comparison/da/ours/rd_s4_t12_gl192_h512`、`comparison/da/adVAE/default`、`comparison/da/MADViT/Full`。
- `record.json` 的 `id` 为 `<dataset>-<Method>-<variant>`（例如 `da-ours-rd_s4_t12_gl192_h512`、`da-adVAE-default`、`da-MADViT-full`），用于 CATALOG/索引唯一标识；目录路径与 `id` 是两种表示。
- `dataset`：`da` 或 `uit`。
- `method`：官方方法名，保留官方大小写和标点，如 `ours`、`adVAE`、`GANomaly`、`Future Frame`、`ST Dissoc.`、`MADViT`、`LightViT-AD`。
- `variant`：方法内变体；我的方法直接用 `experiments/VARIANTS.md` 的 canonical ID（`rd_s12_t4_h512` 等）；单变体方法用 `default`。
- 详细字段见 [SCHEMA.md](SCHEMA.md)；机器校验以 [schema.json](schema.json) 和 `check_comparisons.py` 为准。

## 状态

`planned` / `running` / `completed` / `failed` / `cancelled` / `blocked` / `unknown` 是**条目生命周期**状态，和轻量化测量的 `efficiency.status` 是两套字段：

- `planned`：已登记方法/变体，等待 URL、代码或运行环境；
- `completed`：有可核对的结果摘要（`results/summary.json`），且 `results.macro_auc` 存在；
- 外部方法的 `source.url` / `source.commit` / `source.local_path` 在状态进入 `running/completed` 前必须填写；
- 轻量化状态只用 `measured` / `partial/unverified` / `paper-only` / `blocked`，定义见 [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) §9。

## 统一协议

主对比默认要求：

- one-class：训练只用官方 train 正常帧；
- 每个场景独立模型，训练帧来自 `sequence*/train/*/*.jpg`；
- 测试：该场景所有 test 视频逐帧分数 concat 后计算一次 global AUC/EER；
- 记录 `seed`、`epochs`、`input_size`、指标与任何协议偏差；
- 外部方法如果原生协议不同（例如需要光流、输入分辨率不同），必须在 `protocol.notes` 写清，不能静默改口径。

## 添加外部方法

1. 用户提供上游 URL 后，clone 到本机默认根目录 `/data1/data8tb/liangyuxuan/Comparison method/<Method>/`（`source.local_path` 记录该绝对路径），或放在条目内 `upstream/`（已 gitignore）；记录 `source.url`、`source.commit`、`source.license`；
2. 在该条目写 `scripts/` 适配器，把上游训练/评测接到 DA 的 per-scene 官方 train/test 上；
3. 运行并产出 `results/summary.json`（至少含 macro AUC/EER 与 per-scene AUC/EER）；
4. 更新 `record.json`，把 `status` 改为 `completed`，填 `results`；
5. 运行：

```bash
python comparison/build_catalog.py
python comparison/check_comparisons.py
```

## ours

六个 DA 变体的对比条目在 `comparison/da/ours/rd_*`：

- V0 `rd_s12_t4_h512`
- V1 `rd_s4_t12_h512`
- V2 `rd_s4_t12_gl768_h512`
- V3 `rd_s4_t12_gl192_h512`
- V4 `rd_s4_t12_gl192_h128`
- V5 `rd_s4_t12_gl192_h128_qr2`

它们指向 `experiments/da/appearance/20261103_variant_matrix_*` 的原始 30 epoch / seed42 实验；对比条目只复制 `results/summary.{json,md}`，完整逐帧分数与 checkpoint 仍以原始实验目录为准。数据来源与泄漏审计见 [20261103_variant_matrix_data_audit.md](../experiments/da/appearance/20261103_variant_matrix_data_audit.md)。

## 轻量化对比

- **v2 公平口径**：[BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)：同一张 5880 Ada、FP32、关闭 AMP/TF32；参数量拆 Inference / Training Trainable / Training-only；FLOPs 用算子计数器 + profiler + 手工公式三方交叉，`1 MAC=2 FLOPs`；时序方法记录 `T/K/stride`；BS1 P50 + Mean/P95，BS16 连续吞吐；含教师/时序/光流。
- **DA 总表**：[da/comparison_summary.md](da/comparison_summary.md)。v2 主表暂时留空/标状态，v1 数字只在归档区。
- **v1 归档**：[da/LEGACY_v1.md](da/LEGACY_v1.md) + [da/LEGACY_v1.sha256](da/LEGACY_v1.sha256)。旧 `benchmark_*.json` / `.log` / 旧审计不改。
- **改动记录**：[protocol_change_log.md](protocol_change_log.md)。
- **逐方法重测**：[RETEST_CHECKLIST.md](RETEST_CHECKLIST.md)。
- **脚本规范**：[benchmarks/README.md](benchmarks/README.md)。v2 脚本：`flops_counter.py`、`bench_v2_common.py`、`bench_v2_ours.py`、`bench_v2_external.py`、`bench_v2_andt.py`、`bench_v2_extra.py`、`bench_v2_madvit_variants.py`、`bench_tf1_extra.py`、`bench_tf2_compat.py`（含 `tf2_compat_shims.py`）、`bench_v2_scored.py`、`bench_tf2_scored.py`；旧 `bench_ours.py` / `bench_external.py` 是 v1 legacy；v2 结果写对应方法/变体目录的 `benchmark_v2.json`，不得覆盖同目录 `legacy_v1/` 下的旧文件。
- **v2 验证记录**：[da/v2_validation.md](da/v2_validation.md)。
- **公平性复核（measured vs paper-matched）**：[da/fairness_audit.md](da/fairness_audit.md)。逐方法的官方/适配/未确认判断、评分完整性和需脚注/需重测清单。
- 机器字段：`record.json` 可选 `efficiency` 对象，见 [SCHEMA.md](SCHEMA.md)。

## 当前 DA 对比矩阵

`comparison/INDEX.md` 会列出：

- `ours` 六个变体（completed）；
- 用户表格中的外部方法（planned，等待 URL）：adVAE、GANomaly、MemAE、MKD、ANDT、Future Frame、ST Dissoc.、MLEP、ASTT、DAD-FSM、HSTforU、MADViT（Full / Multi-layer / Last-layer）、LightViT-AD。

外部方法的年份/星号/颜色按用户表格登记；`record.json` 暂未编码颜色分类，需要时再加 `category` 字段。

13 个外部仓库已 clone 到 `/data1/data8tb/liangyuxuan/Comparison method/<Method>/`（目录名保留官方大小写/标点），对应 15 条 `planned` record（MADViT 三个变体共享一个仓库）；`source.url/local_path/commit` 已填，等待 DA 适配与运行。`Comparison method/MANIFEST.csv` 保留每个仓库的 commit / license / clone 状态。
