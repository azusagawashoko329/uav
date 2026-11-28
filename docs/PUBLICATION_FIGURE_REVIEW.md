# 论文三模型图示核对（2026-11-17）

核对对象为用户本轮提供的三张示意图。以[已确认模型定义](PUBLICATION_VARIANTS.md)和当前代码运算为依据。本文给出修改要求，不编辑原图，不代表模型新训练结果。

## 1. PT-RD 图

基础模型名称为 **PT-RD**，图中RT-RD已由用户确认是笔误。PT-RD沿用V0，学生12个标准Transformer block，但教师只提取4个tap，且只有4个重建头。

| 当前图示 | 应修改为 |
|---|---|
| 标题 RT-RD | PT-RD |
| 12层教师特征都参与瓶颈与损失 | 仅选代码零起始tap `[5,7,9,11]`；若用论文一起始编号则为第6/8/10/12层 |
| 瓶颈 `(B,12,768)`、concat=9216 | 四tap：逐patch `(B,P,4,768)`、concat为`(B,P,3072)`；MLP为`3072 → 512 → 64` |
| 学生12个block后都显示重建特征 | 12个block保留，在第3/6/9/12个block后设置4个Linear重建头；依次重建代码teacher tap 11/9/7/5 |
| `1/12`、12项求和 | 对4个tap的Top-5误差取平均；若教师特征示意保留12层，需明确标出只有4层被选中 |
| 误差图中的FLATTEN | 明确为每patch的768维向量间cosine distance；不把整张`256×768`特征图展平为一个向量 |

PT-RD的OCBE是V0的`n_taps=4, fuse_hidden=512, code_dim=64`实例；HAP-RD虽然复用相同Python类，但输入12个tap，瓶颈第一层维度和参数量都不同。

PT-RD误差定义：

\[
 d_{b,t,p}=1-\cos(F^T_{b,t,p,:},\hat F^S_{b,t,p,:}),\qquad
 e_{b,t}=\frac15\sum_{p\in\operatorname{Top5}(d_{b,t,:})}d_{b,t,p}.
\]
\[
 s_b^{PT}=\frac14\sum_{t\in\{5,7,9,11\}}e_{b,t},\qquad
 L^{PT}=\frac1B\sum_b s_b^{PT}.
\]

若使用`L_t`已包含batch mean，最后只需四tap平均，不能再重复除以B。Top-5选择的是最大的距离，不是最大的cosine相似度。

## 2. HAP-RD 图

| 当前图示 | 应修改为 |
|---|---|
| 四个学生block都写MHSA + GELU | LGGD按数据流为GLMB → GLMB → GLMB → GGMB；均使用Gated FFN（hidden ratio=2） |
| 普通12-tap瓶颈的concat写6144，但下方Linear写9216→512 | concat应为9216；无PODP时不产生6144维。完整路径`9216 → 512 → 64` |
| 一个stage特征直接与3个教师层连线 | 一个stage先经过3个独立Linear头，再分别与对应的3个教师层计算误差；画成分支或明确标注“3 independent reconstruction heads” |

图中学生计算从右向左时，应依次为右侧L、L、L、左侧G。图中teacher从左向右为浅到深，stage从右向左为1到4，最后的G对齐最浅组；此反向对应关系可保留。

## 3. HAP-RD-Lite 图

Lite图从右向左为L→L→L→G，**顺序与已确认LLLG一致**。以下标注需要补充或修改：

- `SiLU FFN ×4`容易被理解成MLP hidden ratio=4或每个block重复4个FFN；改为 **Gated FFN，hidden ratio=2**。当前运算为输入投影`d→4d`，拆成value/gate各`2d`，再`SiLU(gate)⊙value`和输出投影`2d→d`。输入投影输出4d不表示hidden ratio=4。
- Local mixer应包含`3×3 DWConv + Linear`；全图或图注说明pre-LayerNorm与残差。若使用模块名GLMB/GGMB并在子图完整定义，可以在主图简写。
- Lite学生stage特征是`(B,P,192)`；每stage三个独立Linear头均为`192→768`，输出才是`(B,P,768)`。图中若把`H_i^S`标为768维，需要明确它是重建头输出而非stage隐藏特征，并有三份独立重建输出。
- PODP图中Q与三层特征之间建议用普通矩阵乘法，避免`⊗`被理解成Kronecker积。对固定的样本b、patch p和组g：

\[
 X_{b,p,g}\in\mathbb R^{3\times768},\quad
 Q_g\in\mathbb R^{2\times3},\quad
 Y_{b,p,g}=Q_gX_{b,p,g}\in\mathbb R^{2\times768},\quad Q_gQ_g^\top=I_2.
\]

当前定义使用四个独立的可学习矩阵 `Q_g`，每个按 seed 随机正交初始化，并在前向中分别用 Gram-Schmidt 保持两行正交。每个 `Q_g` 只在对应组的各patch和通道之间共享，四组之间不共享。无需也不能要求 `Q_g^T Q_g=I_3`。每tap先LayerNorm，再PODP。旧共享-Q图示若继续用于历史结果，需明确标注为 shared-Q historical run。

- PODP后四组×两模式=8个特征，`8×768=6144`、`6144→128→64`的瓶颈维度正确。12个原始教师层仍是蒸馏目标，不能把损失改成8个模式的对齐。
- 统一标题大小写为 **HAP-RD-Lite**。

## 4. 三图共同规范

### 保留空间patch轴

右侧瓶颈都逐patch工作，图中`(B,12,768)`、`(B,9216)`、`(B,64)`若保留，需要明确注明“For each patch p”。若写全tensor形状，则应为：

| 模型 | 深度concat或投影后concat | Code | Code到学生投影 |
|---|---|---|---|
| PT-RD | `(B,256,3072)` | `(B,256,64)` | `64→768` |
| HAP-RD | `(B,256,9216)` | `(B,256,64)` | `64→768` |
| HAP-RD-Lite | `(B,256,6144)` | `(B,256,64)` | `64→192` |

不能画成先把空间patch聚成一个`(B,64)`向量，再无说明地展开为`(B,256,64)`。

### 特殊token与LayerNorm

256×256图像经patch embedding得到256个patch token，加1个CLS和4个register token后是261个token。每个被提取的教师block输出都移除这5个prefix token，再使用patch特征。原图“CLS token register token”建议明确写“Add 1 CLS + 4 registers”，在teacher tap出口写“Remove 5 prefix tokens”。

所有瓶颈使用每tap LayerNorm；可在主图中画出，也可在图注明示。HAP-RD/Lite解码器的LayerNorm、残差与GLMB中的Linear若省略，必须在模块子图或图注完整定义。

### HAP-RD / Lite 的PLDD公式

逐样本、逐教师层先算Top-5 patch cosine distance mean，再在每组的三个标量之间做mean/max，最后四组平均：

\[
 e_{b,g,r}=\frac15\sum_{p\in\operatorname{Top5}(d_{b,g,r,:})}d_{b,g,r,p},
\]
\[
 s_b=\frac14\sum_{g=1}^{4}
 \left[\frac{1-\alpha}{3}\sum_{r=1}^{3}e_{b,g,r}
       +\alpha\max_{r\in\{1,2,3\}}e_{b,g,r}\right],\quad \alpha=0.5,
\]
\[
 L_{\mathrm{PLDD}}=\frac1B\sum_{b=1}^{B}s_b.
\]

用户两张HAP图的最终`1/(4B)`聚合形式与此一致，前提是`e`中还未先做batch mean。测试直接使用逐样本`s_b`，与训练目标严格match，分数越大越异常。

需避免把三个独立重建头画成同一个重建张量复用三次；对于同一个stage隐藏特征，三个头的参数和输出独立。

## 核对结论

已经没有需要另行定义的头数、对齐、Q范围歧义。当前三张图尚不能直接作为已定稿架构图：应先纠正PT-RD的4/12 taps与头数冲突、HAP-RD的模块和concat维度、Lite的FFN/宽度/独立重建头标注，以及三图的逐patch维度表达。本文描述的是图示修订，不要求改变已经确认的架构。
