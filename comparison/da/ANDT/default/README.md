# ANDT (parameters counted; latency/FLOPs blocked)

- 论文：Jin et al., *Anomaly Detection in Aerial Videos With Transformers*, IEEE TGRS 2022（arXiv:2209.13363）；Drone-Anomaly 数据集与 ANDT baseline。
- 本地代码：`/data1/data8tb/liangyuxuan/Comparison method/ANDT/`（HEAD `f2e8c4f70123`；`record.json.source.local_path` 指向这里）。
- 对比条目目录只放记录，不复制上游代码。

## 参数（按代码结构）

- `b16` 配置：`num_layers=2`、`heads=6`、`patch=16`、`emb_dim=768`。
- `num_frames=6`（论文 ablation 最佳）、`mlp=4096`：**70.347M** total/trainable。
- `num_frames=6`、`mlp=1536`（config 默认）：**62.478M**。
- `num_frames=1`、`mlp=4096`：**69.364M**。

## 为什么 FLOPs/latency 标 blocked

发布代码 `ANDT/ANDT/model.py` 的 `forward` 是未整理变体：

- 里面有 `pdb.set_trace()`，直接跑会中断；
- forward 里 `.cuda()` 硬编码；
- decoder 里 `x.view(x.size(0), 256, 16, 16)` 只对应 **1 帧**的 256 个 patch；`num_frames>1` 时 `feat` 是 `1 + 256*num_frames` 个 token，`view` 形状不匹配。论文的 6 帧预测模型需要确认/修补 decoder 后才能测 FLOPs/latency。

因此参数量可按代码结构数（见上），但时延/FLOPs 不能把这份 1 帧变体冒充论文 6 帧模型。

## 如果接受 adapted 版本

最小改动（不重训、不改权重）：

1. 去掉 `pdb.set_trace()`；
2. forward 末尾从 `output = self.decoder(feat)` 改为 `output = self.decoder(self.de_dense(feat[:, 0]))`（`de_dense` 是代码里已定义但被注释掉的 768→256×16×16 投影）；
3. 输入 `(B, 6, 3, 256, 256)`，decoder 输出预测的下一帧；
4. 按 `BENCHMARK_PROTOCOL.md` 测 BS1/BS16/FLOPs，主表标 `adapted`。

这不是作者发布的完整 6 帧模型（`de_dense` 路径被注释，论文 decoder 细节也可能不同），所以不能冒充 official；是否接受 adapted 数字需要确认。
