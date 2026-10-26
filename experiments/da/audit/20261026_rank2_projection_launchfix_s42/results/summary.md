# Rank2投影融合：定位并修复执行开销

GL192＋Dense128＋共享可学习随机正交初始化Q，seed42。恢复七场景epoch15 raw_Q，比较缓存后原逐元素循环与保持FP32乘加顺序的融合投影。Dense128为同轮参考。训练结构、Q值、loss和所有原state_dict键均不改。

## 计时口径

RTX 5880 Ada、FP32、TF32/AMP关闭、256×256、本轮仅batch1。每函数预热30次、六轮各100次，三个条件以全部六种顺序轮换。Latency为每次CUDA同步的墙钟p50；FPS=1000×batch/Latency。连续队列FPS另存CSV/JSON。Profiler在全部计时结束后的独立子进程执行。

旧完整checkpoint未保存，本次恢复各场景最终Q，其他权重使用完全配对的同构初始化。检查函数等价性与执行开销，不重新评测AUC。

## 单帧主结果：纯OCBE＋学生

按七场景p50取算术均值，FPS按该均值换算；七个Q不表示七份硬件独立样本。

| 条件 | OCBE ms | 子网 ms | FPS | CUDA kernel/OCBE |
|---|---:|---:|---:|---:|
| dense128 | 0.3991 | 2.0862 | 479.3 | 17 |
| cached_loop | 1.0559 | 2.7523 | 363.3 | 57 |
| fused | 0.4511 | 2.1486 | 465.4 | 18 |

## 逐场景单帧配对

| 场景 | Dense ms | 原Rank2 ms | 融合Rank2 ms | 原FPS | 融合FPS |
|---|---:|---:|---:|---:|---:|
| bike | 2.0487 | 2.7399 | 2.1094 | 365.0 | 474.1 |
| crossroads | 2.0879 | 2.7442 | 2.1490 | 364.4 | 465.3 |
| farmland | 2.0946 | 2.7582 | 2.1556 | 362.6 | 463.9 |
| highway | 2.0908 | 2.7533 | 2.1541 | 363.2 | 464.2 |
| railway | 2.0932 | 2.7579 | 2.1566 | 362.6 | 463.7 |
| solar | 2.0939 | 2.7560 | 2.1589 | 362.8 | 463.2 |
| vehicle | 2.0946 | 2.7566 | 2.1570 | 362.8 | 463.6 |

## 子步骤隔离

以下是各阶段单独调用的耗时，不能直接相加替代pipeline实测。Dense128的pack_or_project仅执行concat，Rank2两种实现执行Q投影；它们输出宽度不同，后续MLP随之缩小。

| 步骤 | Dense128 ms | 原Rank2 ms | 融合Rank2 ms |
|---|---:|---:|---:|
| pack_or_project | 0.0251 | 0.6294 | 0.0495 |
| fuse_mlp | 0.0912 | 0.0861 | 0.0863 |
| ocbe | 0.3991 | 1.0559 | 0.4511 |
| subnet | 2.0862 | 2.7523 | 2.1486 |
| subnet_with_score | 4.2855 | 4.9347 | 4.3396 |

## 验证、解释与使用

原投影包含24次逐元素乘法、16次逐元素加法及一次拼接；融合版先按[tap,channel]顺序concat十二层，再用一个kernel计算四组所有patch/通道并直接写出[group,direction,channel]顺序。使用显式round-to-nearest乘法/加法，保持原先(乘积0＋乘积1)＋乘积2的顺序。PTX单独归档并检查无FP32 FMA。

验证包括7组完整尺寸投影、OCBE、12个学生输出及异常分数，要求逐位一致；还包括非整块尺寸、非连续输入、梯度开启的eval、训练恢复、加载权重和精度转换的缓存失效。误差/状态见verification.json与measurements.json。

启用：加载权重 → to(device).eval() → cache_depth_basis_for_inference(fuse_projection=True) → inference_mode下前向。默认缓存接口仍为原循环以保持兼容；显式融合只支持CUDA FP32并依赖Triton，训练及梯度开启路径继续使用原实现。设备/精度转换、重新训练或载入新权重后重新准备。

Rank2训练参数仍为3,761,222，主要矩阵算术仍为1.957199872 GFLOPs。Q投影本身另有约0.00786432 GFLOPs逐元素运算（batch1），此前主要FLOPs口径未计它；融合减少执行次数和中间张量，不减少这部分数学运算。

桌面GPU合成特征、未锁定时钟或绑核、不包含教师/预处理/H2D；不是完整模型FPS。历史AUC 82.9599未重算，不能将本次初始化网络测速视作新AUC结果。

## 缩短主机启动路径的复核

上一轮融合投影已使OCBE kernel数从57减到18，CUDA kernel时长总和约97us，与Dense128约100us接近，但墙钟子网仍慢约0.095ms。本轮仅优化融合执行的启动和打包：用等价flat concat替代stack，按(device,total,channels,groups)缓存已编译kernel的launcher，避免每帧重复JIT签名构造与设备分派。缓存不保存Tensor指针或Q值，始终传入当前权重对应的Q。

本轮聚焦batch1，六种顺序轮换的配对协议、七场景最终Q、参考模型、输出与梯度检查均复用上轮；没有重新训练或改Q/学生/loss。额外核对CUDA0/1切换、launcher cache命中及非默认stream，不改变调用方的当前设备。Profiler仍由计时结束后的独立子进程采集。
