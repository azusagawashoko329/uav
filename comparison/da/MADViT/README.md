# MADViT：repo-default 与 paper-Full 评分

本地代码 `/data1/data8tb/liangyuxuan/Comparison method/MADViT/`（HEAD `a1d1732`）没有 Full / Multi-layer / Last-layer 的开关；`multi_layer_distillation_loss` 计算了 `feature_loss`，但最终 `return token_loss`。

因此本目录现在区分三条线：

| 名称 | 含义 | 状态 |
|---|---|---|
| **Full（repo-default）** | 原仓库代码路径：token-loss only，batch 标量 | 保留原测量，`reference-only`，不是论文 Full |
| **Full-paper** | 按论文 Sec. 5.4 修正：每层 feature loss + final CLS/DIST token KL，逐样本 `[B]` | 正式效率测量，`paper_matched=true`（图/输入/评分；权重仍 random init） |
| **Multi-layer / Last-layer** | 按论文 ablation 重建的 scoring，batch 标量 | `reconstructed`，仅参考，**不作为正式复现结果** |

## 1. 代码实际实现（repo-default Full）

`code/train_val_VIT_Distillation.py` 的 `multi_layer_distillation_loss`：

```python
feature_loss = ...  # 12 层 cosine，逐层 mean
token_loss = ...    # CLS/DIST KL，batchmean
return token_loss   # ← feature_loss 被计算但未返回
```

- `Teacher.forward` 对全部 12 个 block append skip connection；
- `Student.forward` 同样对全部 12 个 block append；
- 评估时调用同一个 `multi_layer_distillation_loss`；
- 没有 `full/multi/last` 参数或三个 checkpoint。

所以 repo-default “Full” 的评分实际是 token-only，不能与论文 Full AUC 直接配对。原 `Full/benchmark_v2.json` **保留不动**，只增加 `repo-default / reference-only` 元数据。

## 2. 论文 Sec. 5.4 的 Full

论文定义：

```text
L = L_feat + L_token

L_feat = (1/L) * sum_l [ 1 - CosSim(F_t[l][:,2:], F_s[l][:,2:]) ]   # patch tokens
L_token = (T^2/2) * sum_{i in {CLS,DIST}} KL(LogSoftmax_T(t_s^i) || Softmax_T(t_t^i))
```

实现 `Full-paper` 时：

- feature loss 在 **patch 维** 上先算 per-patch cosine（`dim=-1`），再对 patch 取 mean，得到 `[B]`；最后对 12 层取平均；
- token loss 用 `reduction='none'` 按 class 维求和，得到 `[B]`；`T=4.0`；
- 两项相加，返回 **per-sample `[B]`**，不再对 batch 求平均；
- input `224×224×1`，teacher/student 仍是官方 DeiT-base-distilled / DeiT-tiny-distilled 图。

脚本：`comparison/benchmarks/bench_v2_madvit_variants.py --modes Full-paper`。

## 3. 测量结果

| 变体 | 评分 | FLOPs | P50 | FPS@BS1 | FPS@BS16 | 输出 | 文件 |
|---|---|---:|---:|---:|---:|---|---|
| Full（repo-default） | token-only | 38.533G | 11.525ms | 83.9 | —（batch 标量） | `[]` | `Full/benchmark_v2.json` |
| **Full-paper** | feature+token，per-sample | **38.533G** | **11.066ms** | **90.1** | **457.6** | `[1]`/`[16]` | `Full-paper/benchmark_v2.json` |
| Multi-layer（reconstructed） | feature only | 38.533G | 11.130ms | 88.9 | —（batch 标量） | `[]` | `Multi-layer/benchmark_v2.json` |
| Last-layer（reconstructed） | token only | 38.533G | 9.467ms | 103.5 | —（batch 标量） | `[]` | `Last-layer/benchmark_v2.json` |

- Full-paper 的 FLOPs 与 repo-default 相同（38.533G）：feature/token 评分都是 elementwise/reduction，不计入 MAC-based FLOPs；`flops_verified=true`。
- Full-paper 的 BS1/BS16 已按 50 warmup + 3×500 重测；`BS16=457.6` 使用每批 16 个 per-sample 分数，分子与输出一致。
- repo-default 与 Multi/Last 的旧 BS16 是 batch 标量，per-sample FPS 无效，主表记 `—`；原原始耗时保留在各自 JSON 的 `fps_bs16_raw_batch_aggregate`。
- 权重仍是 random init（官方 checkpoint 未随 repo 提供）；FLOPs/latency 与权重无关，但 AUC 不能用这些随机权重配对。`paper_matched=true` 指图/输入/评分，权重配置另见注。

## 4. Multi-layer / Last-layer 的处理

按论文 Sec. 5.4 的定义，两者分别是：

- Multi-layer：只做全层 feature loss，不做 token loss；
- Last-layer：只监督 final CLS/DIST token。

当前 `Multi-layer/`、`Last-layer/` 是按这两个定义重建的 scoring，共享同一前向图，且返回 batch 标量。它们是**轻量化参考**，不是正式复现结果；没有为了它们重训三套权重。若论文正文要并列展示三个变体的 AUC 与效率，才需要按各自训练目标完整重训并重新评测。

## 5. 结论

- 论文 Full 的评分缺口已补：`Full-paper` 是全层 feature + final token、逐样本 `[B]`；它才是可与论文 Full 讨论配对的效率行。
- 原仓库 `Full` 保留为 repo-default，不能与论文 Full AUC 配对。
- Multi/Last 暂不进入正式复现结论，只保留作 scoring 参考。
