# DA 六变体矩阵数据与泄漏审计

日期：2026-11-04
范围：Drone-Anomaly 七场景；`20261103_variant_matrix_*` 六个外观蒸馏实验。
结论：在官方数据与当前训练/评测代码下，**未发现训练数据泄漏**。Bike Roundabout 与官方 `Drone-Anomaly.zip` 逐文件 CRC 一致，未做筛选或修改。

## 1. 官方性

- 官方 zip：`/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly.zip`（9.7 GB，顶层目录 `Drone-Anomaly/`）。
- 当前解压目录：`/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly/`。
- Bike Roundabout 对比：zip 内 26,384 个文件，当前解压目录 26,384 个文件；`missing=0`、`extra=0`、`size mismatch=0`、`CRC mismatch=0` → 字节级一致，官方原样。
- 数据集根目录的 `README_claude*.md`、`README_codex.md` 是本地核查笔记，不在 zip 内；七个场景的数据目录未做筛选/重命名/删除。
- 注意：数据集 README 的 Bike 计数（6 test videos / 16,950 test frames / 10,779 positives）是旧版；当前是 7 / 18,427 / 12,022（`test/07` 后加）。旧实验的 bike 结果也使用 18,427 / 12,022，与当前一致，不是新旧协议差异。

## 2. 目录、视频 ID、帧键、label

| 场景 | train 帧 | test 视频 | test 帧 | 正样本 | label 长度==帧数 | 帧键 0..N-1 |
|---|---:|---:|---:|---:|---|---|
| bike | 7,950 | 7 | 18,427 | 12,022 | 是 | 是 |
| crossroads | 15,772 | 5 | 6,244 | 1,687 | 是 | 是 |
| farmland | 9,548 | 1 | 2,387 | 80 | 是 | 是 |
| highway | 9,045 | 3 | 2,820 | 497 | 是 | 是 |
| railway | 1,206 | 1 | 882 | 68 | 是 | 是 |
| solar | 2,848 | 3 | 2,450 | 792 | 是 | 是 |
| vehicle | 5,266 | 2 | 2,633 | 973 | 是 | 是 |

- 训练路径只来自 `sequence*/train/*/*.jpg`；测试路径为 `sequence*/test/*/*.jpg` + 同名 `.npy`。
- 所有 test 视频的 label 长度与帧数严格一致；label 只含 `{0,1}`；帧键连续。
- 无 symlink/hardlink。多个场景的 train/test 使用相同视频 ID 和相同帧键编号（如 `train/01/0..N-1` 与 `test/01/0..N-1`），这是官方目录结构；内容检查见下。

## 3. 泄漏检查

### 3.1 字节级

- 七个场景 train/test 之间 **0 个完全相同的 JPEG**。
- 同一 `(sequence, video, key)` 的 train/test 文件内容不同（0 个 byte-identical）。
- 无 symlink/hardlink。

### 3.2 感知近重复（dHash）

| 场景 | test 帧中 exact dHash 命中 train | same-video `|Δkey|≤5` 且 Hamming≤4 | 其中正样本 | 同键 dHash 中位数 | 偏移检验 |
|---|---:|---:|---:|---:|---|
| bike | 12,356 | 6,935 | 3,510 | 3 | offset −10..+10 基本平坦，无 0 低谷 |
| crossroads | 0 | 0 | 0 | 28 | 平坦 |
| farmland | 0 | 0 | 0 | 32 | 平坦 |
| highway | 2 | 0 | 0 | 28 | 平坦 |
| railway | 0 | 0 | 0 | 30 | 平坦 |
| solar | 0 | 0 | 0 | 32 | 平坦 |
| vehicle | 0 | 0 | 0 | 27 | 平坦 |

Bike 的低 dHash 距离**不是同段/相邻帧泄漏**：

- 同键 train/test 的 dHash 距离在 offset −10..+10 上几乎不变（mean ≈ 1.9–5.2），没有 offset=0 低谷；
- 负对照 `train/01 vs test/02`（不同视频）的距离更低（≈1.1），说明这是**静态机位场景相似**；
- 数据集 README 也注明 Bike 是静止镜头；
- 其他六个场景同视频、相邻键的 near4 全部为 0，说明 dHash 在非静态场景有判别力。

### 3.3 标签污染

- 训练 loss/优化不读任何 label；label 只在评测函数中读取。
- Bike 的“同键 test 标签为正”的 train 帧为 4,257 / 7,950 = 53.5%，低于 test 正样本 base rate 12,022 / 18,427 = 65.2%；这符合“静态场景相似 + test 正样本占比高”的预期，不构成训练集被异常帧污染的证据。
- 无 train/test byte-identical 帧；无 train 帧直接来自 test 目录。

### 3.4 代码路径

| 入口 | 训练数据 | test label | checkpoint 选择 |
|---|---|---|---|
| `scripts/da_variant_runner.py`（新矩阵） | 只 glob `sequence*/train/*/*.jpg` | 只在 `evaluate()` | 每场景 final；无 best/early |
| `train.py` / `eval.py` | `FrameDataset` 只读 train | `eval.py` 评测时读 | final |
| structural / width192 / ocbe / seed43 / student12 runners | 只 glob train | 只在 `evaluate()` | final；无 best/early |

- `dataset_manifest` 读取 test 文件大小和 label hash 仅用于 `split_fingerprint`，不参与训练。
- 行为/融合分支已于 2026-11-04 舍弃，不在当前模型/主对比中；历史 fusion 的 test-rank 是**无标签 transductive 后处理**，不是训练泄漏。若重跑 behavior/fusion，需单独核对 `labels[-n:]` 与检测丢帧的对齐。

## 4. 去 Bike 敏感性

| ID | 七场景 macro AUC | 去 Bike 六场景 macro AUC |
|---|---:|---:|
| V2 | 85.2147 | 83.1337 |
| V1 | 85.1314 | 83.0294 |
| V0 | 84.7916 | 82.5816 |
| V5 | 84.6233 | 82.4247 |
| V3 | 84.2137 | 81.9495 |
| V4 | 83.9925 | 81.6985 |

排序为 V2 > V1 > V0 > V5 > V3 > V4（V5/V3 相对旧 top-k 评分交换）。即使采用最保守的“排除 Bike”口径，主结论也不由 Bike 驱动。

## 5. 结论与限制

- **未发现数据泄漏**：官方数据未改；train/test 无 byte-identical 帧；训练只读 train；label 只在 eval；无 test-based checkpoint 选择。
- **Bike 的 dHash 近重复是静态场景相似**，不是同段泄漏；偏移检验和跨视频负对照都支持这一点。
- **残留限制**：数据集不提供时间戳；train/test 共用视频 ID 和帧编号，文件/感知检查不能绝对证明时间上不重叠。本审计依赖官方 split 的“train 只含正常帧”声明；我们没有改动 split。
- **建议**：主表保留官方七场景口径，同时在附录写明 Bike 静态机位 caveat；若审稿要求更保守，用上表“去 Bike 六场景”作为敏感性分析。不要超出已做检查声称“零泄漏”。
