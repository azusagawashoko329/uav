# DA publication 消融链

本轮论文实验直接存放在本目录，各配置独立seed42/43/44目录；原始GL/Loss等实验不移动。正式模型名以[命名规范](../../../docs/PUBLICATION_VARIANTS.md)为准。

过程消融B/C/D/F1/F2的固定版本及机器可读配置见[结构版本存档](ARCHITECTURE_VERSIONS.md)，不列为正式模型。

## 完成结果（2026-11-19）

24 个实验、168 个场景结果已齐全；119 个新任务完成，49 个结果复用，无失败。

- [三种子汇总与逐步消融分析](summary.md)
- [总体矩阵 CSV](results/matrix.csv)、[场景长表](results/scene_metrics.csv)、[同 seed 配对差值](results/paired_contrasts.csv)
- [全量核验报告](results/validation.json)、[汇总 JSON](summary.json)

## 实验数量与复用来源

| 配置 | 学生结构 | teacher taps / heads | Student width | PDFB hidden | PODP | α | 参数量 | 复用seed | 新训练数 |
|---|---|---|---:|---:|---|---:|---:|---|---:|
| PT-RD | 12 × 标准Global，GELU MLP ratio4 | 4 / 4 | 768 | 512 | 无 | 0（四tap平均） | 89,079,104 | 42 | 2 |
| B-ablation | 4 × 标准Global，GELU MLP ratio4 | 4 / 4 | 768 | 512 | 无 | 0（四tap平均） | 32,376,128 | — | 3 |
| C-ablation | 4 × 标准Global，GELU MLP ratio4 | 12 / 12 | 768 | 512 | 无 | 0 | 40,258,880 | — | 3 |
| D-ablation | 4 × 标准Global，GELU MLP ratio4 | 12 / 12 | 768 | 512 | 无 | 0.5 | 40,258,880 | 42/43/44 | 0 |
| HAP-RD | LLLG + gate ratio2 | 12 / 12 | 768 | 512 | 无 | 0.5 | 30,248,000 | 42/43/44 | 0 |
| F1 | LLLG + gate ratio2 | 12 / 12 | 192 | 512 | 无 | 0.5 | 7,718,336 | — | 3 |
| F2 | LLLG + gate ratio2 | 12 / 12 | 192 | 128 | 无 | 0.5 | 4,154,432 | — | 3 |
| HAP-RD-Lite | LLLG + gate ratio2 | 12 / 12 | 192 | 128 | 共享可学习2×3 Q | 0.5 | 3,761,222 | — | 3 |

共24个实验，7个复制复用、17个从头训练；七场景对应49个复用结果、119个新训练任务，整条链最终应有168个场景结果。

- PT-RD/seed42：`experiments/da/appearance/20261103_variant_matrix_v0_rd_s12_t4_h512_s42`。
- D/seed42–44：`ablation/da/GL/20261106_gggg_gelumlp_s42/43/44`。seed42来自V1原矩阵，另两个seed是GL消融中的GGGG+GELU ratio4，模型和训练匹配评分与V1一致。
- HAP-RD/seed42–44：`ablation/da/GL/20261106_lllg_gate_s42/43/44`。

复制原始场景JSON/CSV、日志及本地checkpoint，原始配置和record保存于各目录`provenance/`。保留原source_commit、run_id和实验ID，不伪装成本次新训练；新入口能严格加载复制权重的证据见`publication_preflight.json`。

## 控制变量及图示纠正

- PT-RD → B：仅删除未直接配对重建头的学生block。B从头训练4-block架构，不从PT-RD剪枝checkpoint热启动；保留4 taps `[5,7,9,11]`，stage依次重建11/9/7/5。
- B → C：teacher taps扩为0..11、瓶颈concat随之由3072变为9216、重建头扩为12个。每stage有3个独立头，依次重建9–11 / 6–8 / 3–5 / 0–2；不是一个预测复用三次。该步同时改变teacher输入覆盖与头数，不能描述成仅增加重建头数量。
- C → D：只改α=0到0.5。等大小的四个三层组在α=0时等价于12层平均。D使用LAGA，C不使用max项。
- D → HAP-RD：标准GGGG+GELU变为LLLG+gate，该步同时改变mixer与FFN；细分因素见已有GL消融。
- HAP-RD → F1 → F2 → Lite：依次仅降低student width、降低PDFB hidden、加入PODP。Q四组共用、按seed随机正交初始化，前向Gram-Schmidt保持行正交，code_dim始终64。
- 图中B的`Linear9216`和`1/12`是残留标注，本轮按4 taps的`3072→512→64`、四tap平均实现。C/D无Q，concat为9216而非6144。

## 统一训练与match评测

仅DA；每个场景独立seed初始化；冻结DINOv3 ViT-B/16，输入256×256，实际256个patch，tap_norm=True，teacher维度768，code_dim=64。30 epoch、batch64、workers8、AdamW(max_lr=5e-4, weight_decay=0.05) + OneCycleLR，确定性cuDNN，train drop_last=True。无行为分支。

每对重建/教师特征先逐patch计算`1-cosine`，取最大的5个距离平均。4-tap模型取tap均值；12-tap模型按四组三层做`(1-α)*mean+α*max`后组平均。训练取同一逐样本score的batch mean，测试直接保留score；新入口`publication_match`不调用历史variant runner。测试batch32，读原`sequence*/test/<video>.npy`二值标签，1为异常，分数越大越异常，不反转方向。场景内concat后计算AUC×100及离散ROC EER，七场景等权宏平均。

## 运行入口、核验与调度

- 新模型构造与路由：`model/publication_rd.py`。
- 每实验新建`scripts/run_scene.py` → 新写`scripts/da_publication_scene.py`，不导入历史训练runner。
- `scripts/preflight_da_publication.py`验证全部24份配置、实际forward/backward和重建头stage路由；核验冻结教师实际patch形状；复制的49个scene重算CSV AUC/EER、哈希并严格加载新模型，与历史decoder前向作等价性比对。
- `scripts/da_publication_scheduler.py`读取提交后的manifest，以GPU UUID固定两张物理卡。按训练帧数优先的全局队列，一卡一进程，完成立即补位；按用户要求不做外部占用等待。
- Detached启动：关闭stdin、stdout/stderr写文件、`start_new_session=True`，并忽略SIGHUP；scheduler锁避免重复调度。
- 每scene只有一个正式训练日志`logs/<scene>.log`，恢复时追加。每epoch保存模型、optimizer、scheduler和Python/NumPy/Torch/CUDA/DataLoader RNG，可从完整epoch恢复；final checkpoint后评测失败可从epoch30重新评测。
- 运行source_commit与代码SHA-256固定；每个子进程启动前检查已核验代码和配置字节未变。模型实际guard、分割指纹和环境写入日志/结果/checkpoint。小型结果与日志提交Git，权重仅本地忽略文件并登记哈希。

未完成的训练不标completed；失败日志保留。已有结果不因新命名而重新打分或改写。

## LGGD without LAGA 补充消融（2026-11-19）

别名 d-new-ablation，展示名 **HAP-RD w/o LAGA**，非正式模型版本 `pub_lggd_without_laga_20261119`。在 HAP-RD 的 LLLG+gate、width768、PDFB hidden512、无PODP下，仅将训练与match测试的α从0.5改为0。新跑三个seed、21个DA场景任务，与本目录HAP-RD同seed直接比较；原C/D结果保留。新目录按当前实验规范建在 `experiments/da/aggregation/`：

- [seed42](../../../experiments/da/aggregation/20261119_laga_lggd_a0_s42/README.md)
- [seed43](../../../experiments/da/aggregation/20261119_laga_lggd_a0_s43/README.md)
- [seed44](../../../experiments/da/aggregation/20261119_laga_lggd_a0_s44/README.md)

该补跑检验 LAGA 的效果是否依赖 decoder 模板，不预设一定改善；α同时影响训练和测试读出。
