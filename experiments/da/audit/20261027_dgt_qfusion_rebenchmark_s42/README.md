# DGT及后续融合版本：统一Q推理实现后重测速度

## 研究问题与范围

此前共享可学习Q的Gram–Schmidt和逐元素投影引入额外启动开销，Dense128＋Rank2已经验证融合后明显改善。DGT及其后续版本重写了forward，尚未接入该优化，因此旧FPS/Latency不能作为优化后结构的最终效率结论。本审计重测已存在的7种融合模型，并以随机可学习Rank2和无Rank2 Dense128为对照，不生成新训练模型或新AUC。

7种既有模型：DGT路径图、NoGraph、CrossAttnPool、Bilinear、Slot16、ResidualDepth(group-balanced)、ResidualDepth(score-aligned)。模型来源、记录/权重/最终Q的精确SHA见config。6个后续版本加载各自七场景完整checkpoint；原始DGT及DenseRank2旧训练缺完整checkpoint，仅恢复最终Q，其余权重使用严格配对的seed42初始化。不能把后两者称为完整训练模型复测。

## 唯一优化与等价性

Q矩阵、四组三层教师、组内value/gate、所有跨组融合、学生、loss均不改。只在显式准备的CUDA FP32无梯度推理路径，缓存GS(Q_raw)，把四组投影融合为一个kernel。DGT按[group,batch,patch,2*channel]写输出，使每组进入原Linear的内存连续，不改组内Linear的调用形状与计算顺序。乘加显式round-to-nearest并保持旧顺序，PTX禁止FP32 FMA。训练和开启梯度的eval保留原运算。

DGT变体默认仍记录训练诊断；部署审计统一显式关闭这些无关CPU统计，原始、缓存循环、融合三个条件都关闭，口径与旧审计clean_copy一致。另检查诊断开关与旧clean_copy输出相同。不会把关闭日志的收益冒充Q融合收益。

## 测量协议

两张卡各一个独立进程，分别出表，单进程只访问指定CUDA卡。5880绑定CPU20–23，A6000绑定CPU0–3；FP32、TF32/AMP关闭，CPU4线程。主表batch1，256patch、教师宽度768的GPU合成特征，七场景模型/最终Q分别测试。统计范围为OCBE＋GL192学生，不包含教师、图像处理、传输和评分；OCBE单独计时。

四条件同轮：无Rank2 Dense128控制、未缓存Q原循环、缓存Q原循环、缓存Q融合。每函数预热30次，4轮各100次，使用4条件Williams平衡顺序，使每条件在每个位置出现一次、每个有向相邻对一次。Latency为逐次CUDA同步墙钟p50，另存p95和各轮中位数；FPS=1000/Latency。主结果先平均七场景p50再换算FPS。

附表仅bike完整checkpoint/最终Q：batch16的Latency及连续队列吞吐（4轮各30）；batch1含评分计时另列，避免混入主表。七个Q不表示七次独立硬件采样，不锁CPU/GPU频率。

所有计时结束后才用独立进程Profiler核对各架构的kernel数、投影kernel和PTX。计时进程不启动Profiler/FlopCounter，计算量沿用旧主要线性代数口径，参数和权重大小重新核对。检查旧原始诊断forward、旧clean_copy、当前关闭诊断路径，以及原循环/缓存/融合的投影、code、12个学生输出、异常分数；要求逐位一致。另核对小尺寸/非连续输入、DGT组连续性、训练与eval梯度回退、state_dict键和缓存失效。

## 运行与归档

运行前提交源码、README、config/record和scripts，记录40位source_commit，再运行DMAD Python的scripts/run_audit.py。该脚本管理两卡独立运行、日志与状态；两卡计时都结束才启动Profiler，完成后生成summary.md/CSV和artifact SHA。旧实验、原速度表及负结果保留。新报告分别显示同轮原实现与优化实现，历史AUC只作引用。

部署：载入checkpoint→to(device).eval()→set_diagnostics_enabled(False)（提供该接口的DGT变体）→cache_depth_basis_for_inference(fuse_projection=True)→torch.inference_mode()前向。默认训练行为与旧checkpoint键不改。

## 复测结果

九个版本均完成七场景复测，所有 batch=1/16 的 code、学生重建和异常分数差异均为 0。主表以 OCBE＋学生为响应链路；`score` 表与旧审计的 core_score 口径对应。不同版本由动态调度分配到两张卡，绝对值只在同一版本的三种 Q 路径内比较。

| 版本 | 原循环 FPS | 缓存Q原循环 FPS | 融合Q FPS | 融合相对原循环 |
|---|---:|---:|---:|---:|
| GL192＋Dense128 | 649.1 | 648.8 | 648.8 | -0.05% |
| Rank2随机可学习Q | 470.3 | 498.6 | **618.3** | **+31.45%** |
| DGT固定图 | 312.3 | 326.4 | **396.5** | **+26.96%** |
| DGT-NoGraph | 409.4 | 429.3 | **521.4** | **+27.37%** |
| DGT-CrossAttnPool | 270.8 | 282.4 | **341.7** | **+26.20%** |
| DGT-Bilinear | 333.0 | 346.3 | **401.7** | **+20.65%** |
| Slot16 | 417.6 | 438.7 | **530.8** | **+27.10%** |
| ResidualDepth（原loss） | 356.8 | 372.1 | **436.3** | **+22.28%** |
| ResidualDepth（score-aligned） | 358.6 | 373.8 | **439.1** | **+22.44%** |

包含异常分数的融合 FPS 为：Rank2 320.6、DGT 236.7、NoGraph 294.6、CrossAttnPool 199.9、Bilinear 252.4、Slot16 300.0、ResidualDepth 原 loss 266.2、score-aligned 267.6。完整逐场景、batch16、Latency、p95 和评分表见 `results/summary.md` 与 `results/measurements.csv`。

这组数字解释了旧表为什么需要重做：Rank2 的原始投影路径在每帧重复触发大量小 CUDA kernel；缓存 Q 只能去掉正交化，融合路径才去掉投影拆分的启动和临时张量开销。DGT 后续版本保留了同一 Q 投影，因此旧 FPS/Latency 同样被拖低。融合后 DGT 各版本的响应链路提高约 20.65%–27.37%；CrossAttnPool 的绝对速度仍最低，是其交叉注意力结构本身的成本。Dense128 没有 Q 投影，融合前后基本不变。

第一轮动态调度在所有 worker 已完成后，于报告 CSV 展平阶段因重复写入 `batch` 字段失败；失败日志保留在 `logs/scheduler.log`，修正后的报告运行记录在 `logs/report_retry2.log`。没有使用第一轮错误报告，也没有丢弃任何 worker 测量。
