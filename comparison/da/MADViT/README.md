# MADViT 变体核对

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

因此代码默认路径最接近表格里的 **Full**（全层 feature + token），但不能把它当成 Multi-layer / Last-layer 的实现。

## 三个表格标签怎么处理

| 表格标签 | AUC 来源 | 效率 |
|---|---|---|
| MADViT Full | 用户表/论文 | 代码默认路径实测：38.533G / 11.744ms / 79.8 FPS@BS1；结果在 `Full/benchmark_v2.json` |
| MADViT Multi-layer | 用户表/论文 | **代码未实现**；不把 Full 的数字复用到这一行，效率留空 |
| MADViT Last-layer | 用户表/论文 | **代码未实现**；效率留空 |

如果后续要补 Multi-layer / Last-layer 的效率，需要先明确论文对这两者的精确定义（用哪些层、是否保留 token loss），再写 adapted scoring；不能仅改一个不存在的 flag。
