# DA / GL 消融矩阵总结

本矩阵以 V2 `rd_s4_t12_gl768_h512` 为母模板，固定 DINOv3 ViT-B/16、12 个 teacher taps、OCBE hidden 512、code_dim 64、student width 768、4 stages、12 个 reconstruction heads、seed42、30 epoch、batch64 和 group-balanced top-5 cosine loss（alpha=0.5）。只改变 student 的四块顺序和 FFN 类型。

测试分数使用与训练一致的 `matched_score(k=5, alpha=0.5)`；表中 AUC/EER 是七个 DA 场景分别计算后再做等权宏平均。

## 总体矩阵

| block order | FFN | trainable params | macro AUC (%) | macro EER | 状态 |
|---|---:|---:|---:|---:|---|
| LLLL | gate | 28,483,904 | 85.5609 | 0.2006 | 新跑完成 |
| LLLG | gelumlp | 34,966,592 | 85.5263 | 0.1922 | 新跑完成 |
| GLLL | gelumlp | 34,966,592 | 85.5066 | 0.1971 | 新跑完成 |
| LLLG | gate | 30,248,000 | 85.3461 | 0.1945 | 新跑完成 |
| GGGG | gate | 35,540,288 | 85.3199 | 0.1953 | 新跑完成 |
| GLLL | gate | 30,248,000 | 85.2147 | 0.1965 | 迁移完成 |
| GGGG | gelumlp | 40,258,880 | 85.1314 | 0.2031 | 迁移完成 |
| LLLL | gelumlp | 33,202,496 | 84.7780 | 0.2041 | 新跑完成 |

## 逐场景 AUC (%)

| scene | GGGG gate | GGGG gelumlp | LLLL gate | LLLL gelumlp | GLLL gate | GLLL gelumlp | LLLG gate | LLLG gelumlp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bike | 97.76 | 97.74 | 97.70 | 97.65 | 97.70 | 97.70 | 97.67 | 97.67 |
| crossroads | 60.87 | 60.18 | 60.76 | 61.44 | 61.55 | 62.54 | 59.22 | 60.19 |
| farmland | 99.46 | 99.49 | 99.42 | 99.50 | 99.46 | 99.55 | 99.45 | 99.52 |
| highway | 94.77 | 94.82 | 93.73 | 93.04 | 94.94 | 94.00 | 94.40 | 94.16 |
| railway | 95.69 | 95.61 | 95.24 | 94.73 | 95.48 | 94.43 | 95.59 | 95.12 |
| solar | 88.66 | 89.42 | 89.44 | 88.99 | 89.72 | 89.65 | 89.62 | 89.60 |
| vehicle | 60.03 | 58.67 | 62.62 | 58.10 | 57.65 | 60.68 | 61.47 | 62.42 |

## 逐场景 EER

| scene | GGGG gate | GGGG gelumlp | LLLL gate | LLLL gelumlp | GLLL gate | GLLL gelumlp | LLLG gate | LLLG gelumlp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bike | 0.0447 | 0.0448 | 0.0461 | 0.0472 | 0.0459 | 0.0448 | 0.0457 | 0.0456 |
| crossroads | 0.4152 | 0.4400 | 0.4349 | 0.4224 | 0.4332 | 0.4281 | 0.4279 | 0.4233 |
| farmland | 0.0516 | 0.0537 | 0.0542 | 0.0472 | 0.0550 | 0.0524 | 0.0542 | 0.0451 |
| highway | 0.1541 | 0.1567 | 0.1623 | 0.1890 | 0.1593 | 0.1666 | 0.1666 | 0.1627 |
| railway | 0.1069 | 0.1069 | 0.1044 | 0.1044 | 0.1069 | 0.1044 | 0.1007 | 0.1081 |
| solar | 0.1960 | 0.1803 | 0.1791 | 0.1840 | 0.1683 | 0.1803 | 0.1767 | 0.1791 |
| vehicle | 0.3988 | 0.4392 | 0.4229 | 0.4343 | 0.4066 | 0.4030 | 0.3898 | 0.3813 |

## 矩阵分析

- **宏平均 AUC 最高**：`LLLL + gate`，85.5609%。**宏平均 EER 最低**：`LLLG + gelumlp`，0.1922。两个指标的最优组合不同。
- V2 母模板 `GLLL + gate` 的宏平均 AUC/EER 为 85.2147% / 0.1965。在 gate 条件下，`LLLL` 相对 V2 的 AUC 变化为 +0.3463 个百分点，但 EER 变化为 +0.0041。
- FFN 效应依赖 block 顺序：GELU MLP 相对 gate 在 `GLLL` 上 AUC +0.2919 个百分点，在 `LLLG` 上 +0.1802；但在 `LLLL` 上为 -0.7829，在 `GGGG` 上为 -0.1885。因此不能把 FFN 类型的效果脱离 block 顺序解释。
- 场景差异主要集中在 Crossroads 和 Vehicle；Bike、Farmland、Railway 等场景的 AUC 较高且不同配置间差距较小，宏平均排名会受到低分场景的相对变化影响。
- 结果是单 seed42、每场景独立训练的描述性比较；没有多 seed 方差或显著性检验，不能据此宣称全局结构优劣。

## 运行与限制

- 六个新单元通过两张 GPU 动态队列完成；恢复运行跳过已有完整场景，未创建 retry 目录。
- 两个迁移单元为已有 DA V2/V1 matched-score 结果：`GLLL + gate` 与 `GGGG + gelumlp`。
- 所有结果均保留每场景 JSON、逐帧 CSV、checkpoint 哈希和 scheduler 日志；大权重留在被 `.gitignore` 忽略的 `ckpt/`。
