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
