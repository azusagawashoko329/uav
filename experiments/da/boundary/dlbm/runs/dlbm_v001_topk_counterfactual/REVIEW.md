# DLBM_v001_topk_counterfactual — failed experiment review

日期：2026-07-14  
协议：seed42 / 每场景独立 / 复用 10-epoch all-12 Probe / 不训练 Final

## 结果

| scene | v001 boundary | episode boundaries | evaluator AUC | 人工 Gate 粗边界（仅局外 sanity） |
|---|---:|---|---:|---:|
| bike | L4 | L3/L10/L10/L4 | 0.9901 | L8 |
| crossroads | L11 | L0/L9/L6/L11 | 1.0000 | L4 |
| farmland | L11 | L2/L10/L11/L5 | 0.9896 | L8 |
| highway | L11 | L6/L10/L8/L11 | 0.9920 | L0 |
| railway | L8 | L10/L7/L8/L8 | 1.0000 | L8 |
| solar | L0 | L10/L0/L9/L11 | 0.9770 | L8 |
| vehicle | L8 | L9/L8/L5/L9 | 1.0000 | L8 |

仅 railway、vehicle 与粗边界一致；highway/crossroads 两个关键区分场景失败。多数场景的 episode
边界高度波动，不能用平均后偶然得到的硬边界掩盖。

## 有效结论

- Top-5 平均输入、梯度计算、累计边界均完成前向/反向 smoke，无 NaN/Inf，峰值显存 1.59 GiB。
- 非饱和场景中，一阶 gain 与真实 one-step 的代表边界符号一致率最高 100%，Pearson 接近 1；
  说明反事实一阶公式的实现正确。
- within-frame CutPaste 不跨 batch item 或 sub-video。

## 失败原因

1. evaluator held-out pseudo AUC 达到 0.977–1.000，部分 episode 的 BCE 风险已经饱和；bike/highway
   的 gain 低至 1e-9，真实 one-step 风险在 float32 下完全不变。
2. gain 的 episode 标准差普遍大于均值，边界由抽到的少量 query 帧主导。
3. 快参数只覆盖 all-layer Probe 附近的 `OCBE + proj_in + heads` 局部更新，不能等价表示 fresh
   suffix OCBE/Student 的训练动力学。
4. 因此，训练 gain predictor 只会拟合一个高方差、与粗边界不一致的目标。

## 决策

该版本标记为失败实验：保留代码、日志、evaluator 和结果用于追溯，但不训练 predictor、不启动
Final。下一版应先解决“边界动作必须改变真实 suffix 表征/优化路径”，优先验证可从 all-layer
OCBE 精确切片得到 fixed-suffix OCBE 的 boundary supernet，而不是通过调整 evaluator epoch 或阈值
修补本版本。
