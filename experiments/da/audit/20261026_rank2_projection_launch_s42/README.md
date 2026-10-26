# Rank2投影融合：缩短启动路径的单帧复核

目标：GL192＋Dense128＋Rank2，共享可学习Q、随机正交初始化、seed42。已有缓存Q版本约491.8FPS，Dense128约630FPS。检验差距是否来自四组三层投影被拆成24次乘法、16次加法和拼接，而非Q数学计算本身。

## 数学与唯一变化

对归一化教师特征X[b,p,g,k,c]，Y[b,p,g,r,c]=Σ_k Q[r,k]X[b,p,g,k,c]，g=0..3、k=0..2、r=0..1。原循环为每方向分别执行三次乘法及左结合两次加法。融合实现先按等价[tap,channel]顺序concat十二层，再由一个GPU kernel为每个输出元素读取三层并按同样顺序计算，直接写出[group,direction,channel]的布局。

显式使用FP32 round-to-nearest乘法/加法，禁止将乘加收缩成FMA；检查生成PTX及逐位输出。默认训练及缓存接口保留原循环，只在显式启用融合的CUDA FP32无梯度推理路径使用新kernel。Q、学生、tap归一化、MLP权重、loss及旧checkpoint键不变。依赖本机已有Triton 2.1，不安装新依赖。

## 运行与对照

同轮条件：Dense128参考、缓存有效Q后的原循环、同一个Q及权重的融合实现。RTX 5880 Ada、FP32、TF32/AMP关闭、本轮只测batch1。每函数预热30次，六轮各100次，三个条件按全部六种排列轮换。单次CUDA同步墙钟p50/p95为延迟；FPS=1000×batch/Latency；连续队列吞吐另计。

分开测量pack/投影、MLP、OCBE、纯子网、子网＋top-k评分，以及公共的逐tap归一化和学生。主结论以纯OCBE＋学生前向为准，不包含教师或行为分支。组件中位耗时不能相加代替pipeline实测。

全部正式计时结束后，再启动独立子进程采Profiler，记录算子次数、CPU self time、CUDA kernel次数与时长，归档PTX。计时进程不启动Profiler，避免上轮同进程profiling影响绝对时延的问题。

## 数据与验证

恢复七场景epoch15 raw_Q，文件SHA256见config。该旧训练未保存完整OCBE/student checkpoint，其他权重按固定seed同构重建，两个Rank2条件严格共用同一权重与输入。此处审计运行开销，不重新验证AUC；历史82.9599仍来自原训练实验。

验证七场景本轮只测batch1的原/融合投影、OCBE、12个重建输出和最终分数，要求逐位一致。另验证非连续输入、非整块小尺寸及不同分组数、梯度开启的eval、恢复训练、载入权重与精度切换的缓存失效。训练参数应保持3,761,222，主要矩阵FLOPs仍1.957199872G；投影数学本身另有0.00786432G逐元素运算，不在此前主要FLOPs统计里。

使用：加载权重→to(device).eval()→cache_depth_basis_for_inference(fuse_projection=True)→inference_mode下前向。只缓存Q而保持旧循环可调用不带参数的cache_depth_basis_for_inference()。设备/精度改变、重新训练或加载另一场景权重后需重新准备。

## 限制

桌面GPU和合成特征、无图像预处理/H2D/教师、无固定时钟/CPU绑核。七个场景是七个最终Q，不是七个硬件独立样本。融合实现仅支持CUDA FP32，默认训练及其他设备不使用它；能否接近Dense128、是否保留数值一致，由本次实测决定。

运行前提交源码和计划，DMAD Python执行scripts/benchmark.py --gpu 0 --mode timing。它在完成计时后自动启动--mode profile子进程，结果均归档在本目录。


## 本轮执行优化

上一轮融合GPU kernel时间已与Dense128接近，剩余主机分派明显。本轮在完全相同GPU数学上进一步优化：flat concat替代stack；按(device,total,channels,groups)缓存CompiledKernel[grid]返回的launcher，只在首次形状调用执行JIT编译/特化分派。缓存里没有Tensor指针或Q值，每帧传入当前Tensor；跨设备时进入对应CUDA context并保留调用方当前设备，使用当前stream。

聚焦batch1的七场景最终Q与六种顺序配对，避免无关批量吞吐重复。额外测试launcher cache命中、CUDA0/1及非默认stream。训练、Q、loss、参数、checkpoint键和浮点乘加顺序不改。计时结束后仍在独立子进程采Profiler。

## 失败记录

在launcher cache-hit功能检查中失败：Triton 2.1 CompiledKernel launcher只接收运行时Tensor参数，误将四个编译期常量一并传入，触发参数数量TypeError。未进入正式性能测量，没有有效FPS结果。本次失败保留；修正后实验另建20261026_rank2_projection_launchfix_s42。
