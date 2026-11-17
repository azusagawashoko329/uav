# 论文模型与模块命名（2026-11-17 定稿）

模型与模块定义已由用户确认：基础模型名称为 **PT-RD**（RT-RD为笔误）；保留每stage三个独立重建头及反向对齐；仅Lite采用共享可学习Q的PODP。PT-RD的OCBE明确指V0的四tap瓶颈实例。本文定稿仅确立命名和配置，不改变历史V0–V5、现有模型默认参数或历史实验记录，不启动补跑。用户三张示意图的修订清单见[图示核对](PUBLICATION_FIGURE_REVIEW.md)。

## 三个论文模型

| 论文名称 | 英文全称 | 用途 | 建议机器 ID | 与开发变体的关系 |
|---|---|---|---|---|
| PT-RD | Patch-aware Transformer Reverse Distillation | 基础参考模型 | `pt_rd` | 与 V0 配置一致 |
| HAP-RD | Hierarchically Aligned Patch-aware Reverse Distillation | 完整方法 | `hap_rd` | V2 的宽度和瓶颈，学生顺序改为 LLLG；不是历史 GLLL V2 的简单别名 |
| HAP-RD-Lite | Lightweight HAP-RD | 轻量化方法 | `hap_rd_lite` | V5 的宽度、瓶颈和可学习 Rank-2 Q，学生顺序改为 LLLG；不是历史 GLLL V5 的简单别名 |

旧 `rd_*` ID、V0–V5 编号和对应 checkpoint 含义保持不变。论文名称描述架构；具体结果仍需 dataset、experiment ID、seed、run_id、source_commit 定位。相同参数量不表示 checkpoint 或训练结果可互换。

已再次确认三模型的对应关系：

- **PT-RD = V0 的完整配置**：12个标准学生block、4个teacher taps、4个重建头；student width=768、瓶颈hidden=512，无Rank-2。
- **HAP-RD = V2 将学生顺序改为LLLG + gate**：student width=768、瓶颈hidden=512，无Rank-2。
- **HAP-RD-Lite = V5 将学生顺序改为LLLG + gate**：student width=192、瓶颈hidden=128，包含共享可学习Rank-2 Q。

因此，从HAP-RD到Lite同时增加PODP并降低两处宽度（student 768→192、瓶颈hidden 512→128）；它不是单独增加Rank-2的单因素对照。若从V5描述HAP-RD，则除了GLLL→LLLG和移除Rank-2，还必须恢复student width=768和瓶颈hidden=512。

## 配置表

下表按用户确认的配置与当前实现核对。

| 配置项 | PT-RD | HAP-RD | HAP-RD-Lite |
|---|---|---|---|
| Teacher | 冻结 DINOv3 ViT-B/16 | 同左 | 同左 |
| Teacher taps（零起始） | `[5,7,9,11]` | `[0,...,11]` | `[0,...,11]` |
| Student blocks | 12 | 4 | 4 |
| Student decoder | Standard Transformer | LGGD | LGGD |
| Block order | 12 × Global | LLLG | LLLG |
| Local mixer | 无 | 3×3 depthwise conv + Linear | 同左 |
| Global mixer | 8-head MHSA | 8-head MHSA | 8-head MHSA |
| FFN | GELU MLP，hidden ratio=4 | Gated FFN，hidden ratio=2 | Gated FFN，hidden ratio=2 |
| Gated FFN 运算 | — | `Linear(SiLU(gate) * value)` | 同左 |
| Student width | 768 | 768 | 192 |
| Teacher/reconstruction width | 768 | 768 | 768 |
| Reconstruction heads | 4 | 12 | 12 |
| 重建头/教师关系 | 每个头对应1个tap | 每stage 3个独立头，各对应1个tap | 同左 |
| Bottleneck 论文名称 | OCBE | PDFB | PDFB + PODP |
| 当前瓶颈实现 | `OCBE` | `OCBE` | `LearnableDepthRank2OCBE` |
| PODP / Q | 无 | 无 | 一个可学习2×3 Q，四组三层教师特征共用 |
| Q 初始化/约束 | — | — | 按seed随机正交初始化；前向Gram-Schmidt行正交化 |
| Fusion hidden dim | 512 | 512 | 128 |
| Bottleneck code dim | 64 | 64 | 64 |
| Per-tap norm | True | True | True |
| Patch discrepancy | Top-5 patch cosine distance mean | 同左 | 同左 |
| Layer aggregation | 4-tap mean | 四组三层，组内0.5 mean + 0.5 max，再组平均 | 同左 |
| 论文Loss名称 | Top-k Reconstruction Loss | PLDD | PLDD |
| 可训练参数（不含冻结教师） | 89,079,104 | 30,248,000 | 3,761,222 |

参数量由当前源码构造上述三个配置并计数核对；这不是新的训练或性能验证。输入256×256时patch grid为16×16，256个patch；Lite的192维仅指学生内部宽度，不改变教师768维重建目标。

## 模块名称及边界

| 名称 | 英文全称/正式术语 | 对应计算与当前实现 |
|---|---|---|
| LGGD | Local-to-Global Gated Decoder | 整个学生解码器：code投影 → GLMB → GLMB → GLMB → GGMB，及重建头；`FourBlockGlobalLocalDecoder(block_order='LLLG', ffn_type='gate')` |
| GLMB | Gated Local Mixing Block | LayerNorm、3×3 DWConv、Linear与残差，再LayerNorm、Gated FFN与残差；`LocalMixerBlock(ffn_type='gate')` |
| GGMB | Gated Global Mixing Block | LayerNorm、MHSA与残差，再LayerNorm、Gated FFN与残差；`GlobalMixerBlock(ffn_type='gate')` |
| PDFB | Patch-wise Depth-Fusion Bottleneck | 逐patch跨teacher taps融合与code压缩；HAP-RD无Q，Lite含PODP。PT-RD沿用V0的四tap OCBE实例，HAP-RD采用12-tap PDFB，不能只凭OCBE代码类名混用两者 |
| PODP | Patch-wise Orthogonal Depth Projection | Lite中按每个patch、每个通道，对相邻三层教师特征用Q投影为两个深度模式，不混合空间patch；Q在各patch、通道和四组之间共享 |
| Bottleneck | Bottleneck | PDFB中实际压缩至code_dim=64的部分；不额外引入新的结构缩写 |
| PLDD | Patch–Layer Discrepancy Distillation | 完整12-tap蒸馏目标，含下面两步；训练取逐样本分数的batch mean，测试保留同一分数 |
| — | Discrepancy-Guided Patch Distillation | 每个teacher/reconstruction pair按patch算`1-cosine`，取最大5个距离的平均 |
| — | Layer-Aware Group Aggregation | 对四个相邻三层组分别计算`(1-alpha)*mean + alpha*max`，再四组平均；alpha=0.5 |

PDFB 的普通路径：每个tap独立LayerNorm → 同patch跨层concat → Linear(hidden) → GELU → Linear(code_dim=64)。Lite路径在LayerNorm后增加PODP，将12个深度特征压成四组×两个模式，再concat并使用hidden=128的压缩路径。这里的depth是教师层轴，不是空间patch轴，也不是学生深度。

## 已确认的重建对齐

现有实现每个stage输出三个独立重建头，12个头依次对应12个教师层；不能写成“每个重建头输出对应三层”。已确认沿用现有反向对齐：

| HAP-RD / Lite stage | Block | 独立重建头目标teacher taps |
|---|---|---|
| 1 | GLMB（L） | 9、10、11 |
| 2 | GLMB（L） | 6、7、8 |
| 3 | GLMB（L） | 3、4、5 |
| 4 | GGMB（G） | 0、1、2 |

按此定义，最后的G对应浅层教师组，前三stage的重建头不会经过最后的G；LGGD表示学生计算顺序为local-to-global。若要让最后的G对应最深教师组，或让其影响全部12个重建输出，需要明确修改对齐/出头位置，属于另一个架构变化。

PT-RD沿用V0：在第3/6/9/12个block后各出一个头，依次对齐teacher taps 11/9/7/5。

## 已有证据与补跑边界

DA / GL多seed消融中，宽度768的LLLG+gate与GLLL+gate参数量均为30,248,000。七场景宏平均AUC：

| seed | GLLL+gate | LLLG+gate | LLLG − GLLL（百分点） |
|---|---:|---:|---:|
| 42 | 85.2147 | 85.3461 | +0.1314 |
| 43 | 85.6322 | 85.8928 | +0.2607 |
| 44 | 85.2080 | 85.7268 | +0.5188 |

三seed宏平均AUC分别为85.3516与85.6552；EER分别为0.196469与0.192889。三个seed的LLLG均高于GLLL，但不是每个场景都更高，也不代表Lite或UIT已经验证同样收益。

- [DA GL三seed汇总](../ablation/da/GL/multiseed_summary.json)及`20261106_lllg_gate_s42/43/44`：已有HAP-RD候选架构的DA证据，历史目录和名称继续保留。
- 历史V5采用GLLL，不可直接改名认定为LLLG的HAP-RD-Lite。
- 现有DA Loss消融采用GLLL母模板，不可改写为LGGD/LLLG的Loss消融。
- 不因命名变化覆盖旧实验、改写旧指标，或把旧源码提交替换成本次文档提交。

## 瓶颈实例与名称边界

三个模型都按patch独立做深度融合，差别不能只用旧类名OCBE表达：

| 论文模型 | 瓶颈名称 | 输入特征数 | 压缩路径（保留batch与patch轴） |
|---|---|---:|---|
| PT-RD | V0 OCBE | 4 taps | `(B,P,4,768) → concat → (B,P,3072) → 512 → 64` |
| HAP-RD | PDFB（无PODP） | 12 taps | `(B,P,12,768) → concat → (B,P,9216) → 512 → 64` |
| HAP-RD-Lite | PDFB + PODP | 12 taps → 8 modes | `(B,P,4,3,768) → shared Q → (B,P,4,2,768) → concat → (B,P,6144) → 128 → 64` |

三者均在融合前使用每tap独立LayerNorm；两个Linear之间使用GELU。P=256。Lite的PODP只影响瓶颈输入，不减少蒸馏目标：损失仍对齐原始12个教师tap，12个重建头每个输出768维。

本轮不重命名Python类，不将旧GLLL默认构造器直接改成LLLG，也不将旧实验结果重新命名为新架构结果。正式模型代码入口和补跑可在此定义基础上另行实施。


## DA 完整消融链补跑（2026-11-17）

新入口为 `model/publication_rd.py`，运行计划见 [ablation/da/publication](../ablation/da/publication/README.md)。24个seed实验中7个复制已有结果、17个需从头训练，共119个新场景任务。PT-RD补seed43/44；B、C、F1、F2、HAP-RD-Lite各补三seed。D复用GGGG+GELU的旧V1等价结果，HAP-RD复用LLLG+gate三seed；新实验均使用新写的scene脚本和显式模型注册表，不改变历史变体默认入口。


## 过程消融版本存档

B、C、D、F1、F2作为非正式论文模型的过程消融版本保存，完整定义和固定源码指针见[结构版本存档](../ablation/da/publication/ARCHITECTURE_VERSIONS.md)。正式模型仍为本文件的三个模型，不因存档增加新的正式模型名称。
