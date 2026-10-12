# 全局-局部学生 block 结构实验

## 研究问题

在 4 个学生 block、12 个教师重建头的反向蒸馏模型中，用一个全局 patch mixer 和三个局部空间 mixer 替代四个镜像 Transformer block，是否能在保持教师输出语义对齐的同时降低 block 内部计算和显存，并保持七场景异常检测 AUC？

## 唯一结构变化

学生输入、OCBE、学生内部维度、12 个重建头和训练损失均保持不变。学生 block 改为：

```text
stage 0: LayerNorm -> global MHSA -> residual -> LayerNorm -> gated FFN -> residual
stage 1: LayerNorm -> 3x3 depthwise conv on 16x16 patch grid
          -> pointwise linear -> residual -> LayerNorm -> gated FFN -> residual
stage 2: same local mixer
stage 3: same local mixer
```

当前 decoder 的反向对齐把 stage 0 映射到最深教师组，因此全局 mixer 放在 stage 0。局部 mixer 不改变 token 数量和 patch 顺序，只在相邻 patch 上做空间混合。gated FFN 使用 SiLU 门控，隐藏比率为 2；这是 block 内部算子变化，不是入口维度变化。

## 固定设置

- 冻结 DINOv3 ViT-B/16，12 层 patch 输出；OCBE 输入 12 层，code_dim=64。
- 学生内部维度 768、4 stages、8 attention heads、12 个独立 768→768 重建头。
- 训练只使用正常帧；group-balanced top-5 cosine loss，alpha=0.5。
- 每个场景独立初始化 OCBE、学生和优化器；seed=42，15 epoch，batch=64，输入 256x256。
- 测试分数为 12 个重建头 top-5 patch cosine distance 的平均，按场景合并帧计算 ROC AUC×100，再取七场景宏平均。

## 调度

`scripts/dynamic_scheduler.py` 使用两个 GPU worker。初始分别探测显存，启动一个场景；任一进程结束后立即在同一 GPU 启动队列中的下一个场景，轮询间隔 2 秒，不按固定时长轮转。调度器事件、每场景日志和结果均保存到本目录。

## 限制

这是单种子结构实验；AUC 结果不能代表多种子稳健性。GPU benchmark 和无人机端实时性不等价，结果中需要另行测量端到端 teacher+OCBE+student 延迟。
