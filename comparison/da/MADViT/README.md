# MADViT 变体核对与实现

本地代码 `/data1/data8tb/liangyuxuan/Comparison method/MADViT/`（HEAD `a1d1732`）**没有** Full / Multi-layer / Last-layer 三种可切换实现。

## 代码实际实现

`code/train_val_VIT_Distillation.py` 的 `multi_layer_distillation_loss` 实际实现：

```python
feature_loss = ...  # 12 层 cosine，逐层 mean
token_loss = ...    # CLS/DIST KL，batchmean
return token_loss   # ← feature_loss 被计算但未返回！
```

因此 code default 的“Full”实际只返回 token loss；虽然 forward 里仍计算了 feature loss，但评分不是论文的 feature+token。

- `Teacher.forward` 对全部 12 个 block 都 append skip connection；
- `Student.forward` 同样对全部 12 个 block append；
- 评估时调用同一个 `multi_layer_distillation_loss`；
- 没有 `full/multi/last` 参数、没有层选择开关、没有三个 checkpoint 目录。

## 论文定义（Sec. 5.4）

论文 PDF 的 ablation 对三个变体给了明确定义：

| 变体 | 论文定义 | 实现 |
|---|---|---|
| Full（code default） | 全层 feature loss + final CLS/DIST token loss | **代码实际 return token_loss；feature loss 被计算但丢弃，不是论文 Full** |
| Multi-layer | 只做全层 feature loss，不做 token loss | adapted scoring：12 层 cosine，无 KL |
| Last-layer | 只监督 final CLS/DIST token（token loss only） | adapted scoring：只做 KL，无 feature loss |

三个变体共享**同一个 teacher+student 前向图**，只改评分函数。因此：

- `inference_params` / `training_trainable` / `FLOPs` 相同；
- latency 会因 scoring 略有差异；
- Full 是代码原路径但**不是论文 Full**（见上）；Multi-layer / Last-layer 是按论文定义重建的 scoring。三者都应标 **reconstructed/unconfirmed**，不能把 Full 当 paper Full。

## 测量结果

| 变体 | FLOPs | P50 | FPS@BS1 | FPS@BS16 | 文件 |
|---|---:|---:|---:|---:|---|
| Full（code default，token-only） | 38.533G | 11.525ms | 83.9 | —（标量） | `Full/benchmark_v2.json` |
| Multi-layer（reconstructed） | 38.533G | 11.130ms | 88.9 | —（标量） | `Multi-layer/benchmark_v2.json` |
| Last-layer（reconstructed） | 38.533G | 9.467ms | 103.5 | —（标量） | `Last-layer/benchmark_v2.json` |

> 三者共享同一 teacher+student 前向图，所以 FLOPs/params 相同；P50 差异只来自 scoring 函数。当前 scoring 返回 batch 标量，BS16 FPS 不能按 16 个样本计，故留 `—`；需要逐样本 scoring 重测。
