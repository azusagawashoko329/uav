# 随机正交初始化可学习 Q：GL192＋OCBE hidden128＋rank2，seed42

## 研究问题与唯一变化

相对[固定 Mean+Trend 初始化的可学习 Q](../20261013_ocbe_depthrank2_learnableorth_h128_s42/README.md)，仅改变共享 2×3 深度矩阵的**初始值**：从 seed42 的独立 CPU 生成器抽样高斯 2×3 矩阵，经 Gram-Schmidt 得到随机正交二维基。它与固定 Mean+Trend 的初始子空间主夹角为 **70.307°**，并非在固定基附近扰动。四组三层教师 tap 共用同一 Q，七个场景独立模型使用相同的 seed42 初始 Q。

局部生成器不消耗全局随机数。预运行检查确认：除 Q 外，OCBE 和学生网络的每个初始权重与固定初始化对照逐元素相同；两组均为 **3,761,222** 个可训练外观参数。Q 的六个原始系数仍可训练，每次前向仍用 Gram-Schmidt 得到有效正交矩阵，不加正交惩罚项。将比较最终 Q 与随机初值、固定 Mean+Trend、固定初始化后学得 Q 的**行子空间主夹角**，以区分真正保留方向变化和行基旋转。

## 不变的训练与评测

- 冻结 DINOv3 ViT-B/16 的 12 个教师 patch tap；四个 Global-Local 学生 block，内部维度 192，反向三对一对齐，12 个重建头。
- 每组三个教师 tap 经共享 rank2 深度聚合；逐 tap 仿射 LayerNorm、`6144→128→64` OCBE 融合不变。group-balanced top-5 cosine loss、alpha=0.5 不变。
- Drone-Anomaly 七场景各自正常帧训练、各自测试。seed42，15 epoch，batch64，256×256，AdamW＋OneCycleLR，max_lr=5e-4，weight_decay=0.05；两张 GPU 动态领取场景。
- 测试分数为每帧 12 个重建头的 top-5 patch cosine distance 平均，越大越异常；用配对 `.npy` 标签计算逐场景 ROC AUC×100，七场景做宏平均。评测时逐帧核对测试键、标签和划分指纹。

## 比较与限制

主要对照是固定初始化后可学习 Q 的同 seed42 结果；另报告固定 Q 的 rank2 结果。预运行等权重检查支持“只改 Q 起点”，但训练非凸且仅抽取一个随机初值，任何 AUC 差异均需谨慎解释。Q 与下游融合层会共同调整，正常帧蒸馏损失不能代替异常 AUC。原始/有效 Q 的逐 epoch 轨迹、正交误差和最终 AUC 均单独保存。
