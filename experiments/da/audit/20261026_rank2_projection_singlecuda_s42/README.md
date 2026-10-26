# Rank2融合投影：单CUDA进程与固定CPU亲和性复核

目标仍为GL192＋Dense128＋共享可学习随机正交初始化Q、seed42。融合GPU运算与预编译launcher完全复用已验证实现，不改训练、loss、Q或参数。launchfix同轮对照显示原Rank2比Dense128慢约24%，融合后FPS差约3%；但三种条件绝对FPS均变化，因此追加受控环境复核，保留全部旧结果。

## 数学、数据与计时

Y[b,p,g,r,c]=Σ(k=0..2) Q[r,k]X[b,p,g,k,c]，四组三层教师、两方向、通道768、patch256；每个输出与旧循环保持相同FP32乘加顺序，PTX要求无FMA。3,761,222训练参数与1.957199872主要GFLOPs不变；Q另含0.00786432G逐元素运算。

恢复七场景epoch15 raw_Q，输入为GPU合成教师特征。旧训练没有保存完整OCBE/student checkpoint，因此其他权重同构重建，原循环和融合严格同权重同输入。AUC不重新评测。Dense128无Rank2、缓存Q旧循环、缓存Q融合三个条件在同一进程配对；RTX5880 Ada、FP32、TF32/AMP关闭、batch1、预热30、六轮各100次并轮换全部六种顺序。子网仅OCBE＋student，不计教师或异常评分。逐次CUDA同步墙钟p50作Latency；FPS=1000/Latency；七场景p50平均后换算主表。

## 本轮环境控制

只在计时进程访问CUDA ordinal0；跨设备/非默认stream/launcher cache-hit已在launchfix审计验证。通过taskset固定CPU20–23（5880的NUMA节点1），实际亲和性和GPU拓扑写入environment。核心原始审计复用scripts及其精确SHA写入产物链。全部计时完成后单独子进程Profiler，避免同进程Profiler影响正式Latency。

由于CUDA上下文准备与CPU亲和性同时改变，不能将跨轮绝对值变化归因其中任一项。主要论断仅用本轮三种条件的配对比值。不锁CPU/GPU时钟，七个Q并非七份硬件独立样本。

## 运行与验证

运行前提交本目录和脚本，工作区干净后执行：

```bash
taskset -c 20-23 /home/lyx/miniconda3/envs/DMAD/bin/python -u experiments/da/audit/20261026_rank2_projection_singlecuda_s42/scripts/benchmark.py --gpu 0 --mode timing
```

日志重定向logs/timing.log；计时脚本自动生成独立Profiler日志、JSON/CSV、PTX和summary.md。验证七场景投影、code、12个学生输出、异常分数逐位相等，以及非连续/非整块输入、训练与梯度回退、checkpoint键、缓存失效。权重和AUC不新生成。

启用优化：载入权重后`ocbe.to(device).eval().cache_depth_basis_for_inference(fuse_projection=True)`，在`torch.inference_mode()`下前向。需CUDA FP32＋Triton；默认接口保持兼容，训练实现不改。

## 观察：最终同轮结果

| 条件 | OCBE ms | OCBE＋student ms | FPS | OCBE CUDA kernel数 |
|---|---:|---:|---:|---:|
| dense128 | 0.2925 | 1.5491 | 645.5 | 17 |
| cached_loop | 0.7141 | 1.9850 | 503.8 | 57 |
| fused | 0.3247 | 1.6029 | 623.9 | 18 |

融合比缓存Q的原循环FPS提高23.83%，Latency降低19.25%。相对同轮无Rank2对照，FPS差距由21.96%缩到3.36%，净延迟差仅0.0538ms。逐场景明细与全部子步骤见[summary.md](results/summary.md)、[summary.csv](results/summary.csv)。

## 解释：问题在执行拆分

Q的确只有六个共享系数，作用在四组×256patch×768通道的三元向量上，数学运算约0.00786432GFLOPs，很小。但原Python表达式每个方向写成三次逐元素乘法与两次逐元素加法；四组、两方向共24次乘法和16次加法。Profiler证实这40个CUDA kernel外还存在48次CPU张量select，反复分派并读写临时结果。它没有以一个整体小矩阵投影执行。此前预缓存Q只去掉Gram–Schmidt，未消除这些开销。

本次将四组所有patch/通道的投影合成一个Triton kernel，flat concat打包输入，并复用已编译launcher。OCBE总kernel数57→18；无Rank2为17。隔离投影步骤约0.4019→0.0361ms，进一步支持主要瓶颈是小算子调度与临时张量，而非Q算术量。组件耗时不可直接相加替代pipeline测量。尚余一次投影kernel与相应数据读写/启动开销，因此轻微慢于Dense128有实现上的解释。当前不主张Rank2增加计算量就必然导致约22%的FPS下降。

## 数值、训练与指标边界

七场景的投影、OCBE code、12个student重建与异常分数最大绝对差均为0；小尺寸/非连续输入、梯度开启eval与缓存生命周期验证通过。跨设备、非默认stream及launcher命中由launchfix独立记录支持。FP32显式乘加与旧循环顺序一致，PTX没有FMA。

仍为3,761,222可训练参数（约3.7612M）和1.957199872主要GFLOPs；未改Q、loss或训练结构。优化仅显式启用的CUDA FP32无梯度推理使用，需要本机已有Triton。原模型与训练路径保留，可加载相同state_dict。

本轮无Rank2参考645.5FPS而旧记录约630，说明绝对测速有环境差异；**不能用本轮623.9直接除以前一轮491.8计算提速**。只用当前三条件同轮配对。未重新训练、未保存新权重、未重测AUC；旧完整checkpoint缺失的问题不由这次审计补造。
