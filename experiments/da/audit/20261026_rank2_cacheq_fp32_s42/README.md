# Rank2推理缓存Q：隔离重复正交化的开销

目标模型：GL192＋Dense128＋Rank2，共享可学习Q，随机正交初始化，seed42。唯一变化是推理前计算一次有效Q=GS(raw_Q_final)，后续前向直接读取缓存。四组压缩、concat、MLP、学生、top-k评分及训练损失都不修改，不进行重新训练。

## 数学与缓存生命周期

训练每一步更新raw_Q，前向仍实时Gram–Schmidt保持可微。推理权重固定时，Q_eff不依赖输入，因此缓存与每帧重新计算在数学上相同。缓存不写入state_dict、不删除原六个训练参数；恢复训练、加载权重、转换设备/精度后失效。梯度开启的eval继续计算可微Q。每个场景使用自己的epoch15 Q，不能跨场景混用。

使用方法：加载权重、to(device/dtype)、eval()后调用`ocbe.cache_depth_basis_for_inference()`；推理包在`torch.inference_mode()`或`torch.no_grad()`中。清除缓存可调用`ocbe.clear_inference_depth_basis()`。

## 数据与实验边界

旧随机Q实验的训练脚本没有保存完整OCBE/student checkpoint，但七场景逐epoch raw/effective Q已完整保存。本次恢复各场景epoch15 raw_Q，其余权重按seed42重建；两条件用完全相同权重和输入。因此本次检验结构开销和函数一致性，不重新验证历史AUC 82.9599，不生成或上传新的训练权重。

输入使用GPU驻留的十二层[B,261,768]特征切去五个prefix，匹配原教师patch的形状和stride；不读取训练/测试集，不优化AUC。七场景表示七个最终Q的一致性检查，不表示七个硬件独立样本。

## 测量协议

RTX 5880 Ada、FP32、TF32/AMP关闭，CPU四线程；batch=1为主、batch=16为辅。每函数预热30次，六轮各100次，每轮交替AB/BA。主Latency为逐次CUDA同步的墙钟中位数，另记p95；单帧FPS=1000/Latency。连续队列吞吐另测。基线与缓存条件本轮配对，不能只和旧2.157ms/463.7FPS跨轮比较。

范围分别为OCBE、OCBE＋学生纯前向、OCBE＋学生＋评分；不运行教师和行为代码。另测Dense128与固定Mean＋Trend作为现有结构参考，不修改其实现。Profiler仅用于核对GS算子和GPU kernel数量，不用其耗时作为主Latency。

## 验证与限制

验证七场景batch1/16的Q、OCBE、十二个学生输出和异常分数。另验证训练梯度、恢复训练、checkpoint兼容、设备/精度转换和参数更新后的缓存失效。训练参数应保持3,761,222；主要算术FLOPs仍为1.957199872G，沿用此前1MAC=2FLOPs且不计逐元素/归一化/评分的口径。

桌面GPU测速不能替代无人机硬件部署；无图像解码、预处理、H2D或教师前向。未锁定GPU时钟或绑核，配对结果用于评估本次修改。缓存只去掉GS，原四组逐元素加权与拼接仍有开销，是否更快以及快多少由实测决定。

运行：DMAD Python执行`scripts/benchmark.py --gpu 0`。先提交计划和源码，再启动；结果和产物按协议归档。
