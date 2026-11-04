# MADViT 变体核对与实现

本地代码 `/data1/data8tb/liangyuxuan/Comparison method/MADViT/`（HEAD `a1d1732`）**没有** Full / Multi-layer / Last-layer 三种可切换实现。

## 代码实际实现

`code/train_val_VIT_Distillation.py` 只有一条评分路径：

```python
def multi_layer_distillation_loss(teacher_feats, teacher_token, student_feats, student_token):
    # 对所有 12 个 teacher/student skip connection 做 cosine feature loss
    # 再加 CLS/DIST token 的 KL loss
```

- `Teacher.forward` 对全部 12 个 block 都 append skip connection；
- `Student.forward` 同样对全部 12 个 block append；
- 评估时调用同一个 `multi_layer_distillation_loss`；
- 没有 `full/multi/last` 参数、没有层选择开关、没有三个 checkpoint 目录。

## 论文定义（Sec. 5.4）

论文 PDF 的 ablation 对三个变体给了明确定义：

| 变体 | 论文定义 | 实现 |
|---|---|---|
| Full | 全层 feature loss + final CLS/DIST token loss | 代码默认 `multi_layer_distillation_loss` |
| Multi-layer | 只做全层 feature loss，不做 token loss | adapted scoring：12 层 cosine，无 KL |
| Last-layer | 只监督 final CLS/DIST token（token loss only） | adapted scoring：只做 KL，无 feature loss |

三个变体共享**同一个 teacher+student 前向图**，只改评分函数。因此：

- `inference_params` / `training_trainable` / `FLOPs` 相同；
- latency 会因 scoring 略有差异；
- Full 是代码原路径，Multi-layer / Last-layer 是按论文定义重建的 scoring，标 **adapted**。

## 测量结果

| 变体 | FLOPs | P50 | FPS@BS1 | FPS@BS16 | 文件 |
|---|---:|---:|---:|---:|---|
| Full | 38.533G | 11.525ms | 83.9 | 444.0 | `Full/benchmark_v2.json` |
| Multi-layer | 38.533G | 11.130ms | 88.9 | 432.2 | `Multi-layer/benchmark_v2.json` |
| Last-layer | 38.533G | 9.467ms | 103.5 | 430.2 | `Last-layer/benchmark_v2.json` |

> 三者共享同一 teacher+student 前向图，所以 FLOPs/params 相同；P50/FPS 的差异只来自 scoring 函数。所有结果都按 v2 协议（50 warmup + 3×500）测量。
