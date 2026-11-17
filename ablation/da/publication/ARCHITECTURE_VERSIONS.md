# Publication过程消融结构版本存档

B、C、D、F1、F2作为**过程消融版本**长期保留，不列为正式论文模型。正式模型仍为PT-RD、HAP-RD、HAP-RD-Lite；开发期V0–V5编号和定义保持原样。

本存档固定的是结构和训练/评测定义，不是训练完成状态。各seed的实时运行状态、指标、权重哈希和原始复用证据继续由对应实验的`record.json`、`artifacts.json`管理。

| 存档版本ID | 展示名称 | Student | taps / 独立头 | Student width | PDFB hidden | PODP | α | 可训练参数 |
|---|---|---|---|---:|---:|---|---:|---:|
| `pub_b_20261117` | B-ablation | 4标准Global，GELU ratio4 | 4 / 4 | 768 | 512 | 无 | 四tap平均 | 32,376,128 |
| `pub_c_20261117` | C-ablation | 4标准Global，GELU ratio4 | 12 / 12 | 768 | 512 | 无 | 0 | 40,258,880 |
| `pub_d_20261117` | D-ablation | 4标准Global，GELU ratio4 | 12 / 12 | 768 | 512 | 无 | 0.5 | 40,258,880 |
| `pub_f1_20261117` | F1 | LLLG + gate ratio2 | 12 / 12 | 192 | 512 | 无 | 0.5 | 7,718,336 |
| `pub_f2_20261117` | F2 | LLLG + gate ratio2 | 12 / 12 | 192 | 128 | 无 | 0.5 | 4,154,432 |

## 固定运算及对照关系

- B：PT-RD的12个学生block减为4个，保留teacher taps `[5,7,9,11]`，4个独立重建头按11/9/7/5反向对齐；从头训练，不剪枝旧权重热启动。
- C：B扩为teacher taps 0..11、12个独立头；四stage依次重建9–11、6–8、3–5、0–2。每stage三个头各自预测，不把一个预测重复配给三层。α=0等价于12层平均。
- D：与C同结构，仅将α改为0.5，使用LAGA；与旧V1结构和Loss一致，已有三seed结果复制复用。
- F1：从HAP-RD仅降低学生内部宽度768→192；FFN、LLLG顺序、12头、瓶颈hidden512和Loss不变。
- F2：从F1仅降低瓶颈hidden512→128，仍无PODP；F2→HAP-RD-Lite才加入共享可学习Rank-2 Q。
- F1/F2与旧V3/V4只具有相同宽度：旧模型为GLLL，存档版本为LLLG，因此不能作为旧V3/V4的简单别名。

统一冻结DINOv3 ViT-B/16、tap_norm=True、code_dim64、输入256×256、30 epoch、batch64、AdamW+OneCycleLR。所有版本先算逐patch cosine distance、取最大的5个patch距离平均。B对4个tap平均；其他版本按四组三层做`(1-α)*mean+α*max`再组平均。训练对同一match分数增加batch mean，测试保留逐样本score。

## 精确源码与配置

- 本轮运行源码提交：`aa0f057d834efc9ec30ba1a120b93d9d42aa7028`。
- 新模型入口：`model/publication_rd.py`中的`build_publication_model`。
- 新训练入口：`scripts/da_publication_scene.py`。
- match入口：`model/publication_rd.py`中的`publication_match`。
- [机器可读版本存档](ARCHITECTURE_VERSIONS.json)：保存五个版本的完整模型字段、固定训练参数、match配置、重建头路由、三个seed的实验入口及config SHA-256，同时记录源码文件SHA-256。
- [运行前核验](publication_preflight.json)：保存实际模型配置、forward/backward、路由检查和复用权重等价性证据。

这些版本ID后续保持不变。若修改结构、Loss或路由，增加新版本，不覆盖本版；seed与运行次数不写入结构版本ID，仍由独立实验ID和run_id区分。复制权重的原始训练提交以各实验provenance记录为准，不替换成本轮源码提交。

## 三seed实验入口

| 版本 | seed42 | seed43 | seed44 |
|---|---|---|---|
| B | [记录](20261117_b_ablation_s42/record.json) | [记录](20261117_b_ablation_s43/record.json) | [记录](20261117_b_ablation_s44/record.json) |
| C | [记录](20261117_c_ablation_s42/record.json) | [记录](20261117_c_ablation_s43/record.json) | [记录](20261117_c_ablation_s44/record.json) |
| D | [记录](20261117_d_ablation_s42/record.json) | [记录](20261117_d_ablation_s43/record.json) | [记录](20261117_d_ablation_s44/record.json) |
| F1 | [记录](20261117_f1_s42/record.json) | [记录](20261117_f1_s43/record.json) | [记录](20261117_f1_s44/record.json) |
| F2 | [记录](20261117_f2_s42/record.json) | [记录](20261117_f2_s43/record.json) | [记录](20261117_f2_s44/record.json) |
