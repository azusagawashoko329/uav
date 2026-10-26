# 轻量化性能统一审计：FP32、256×256、seed42

本目录是一项跨现有模型的性能审计，不产生新训练模型。比较 16 个已登记架构/配置：12-block 历史基线、原始4-block及两个adapter、GL768/GL192、Dense128、固定/可学习Rank2，以及DGT各融合版本。

原约89M基线是 **12个原始学生block＋4tap OCBE**，当前源码重建的OCBE＋学生精确参数为89,079,104；原始4-block＋12重建头＋Dense512为40,258,880。两者不能混称为4block89M。历史配置、record和源码指针均在config的variants中登记；不能证实的原运行commit保留null。

## 统一口径

- 分别报告OCBE＋学生和加上85,641,216参数冻结教师的完整模型。教师在实际推理中仍需要运行；按用户最新定义，蒸馏外观分支本身即整个模型，行为分支已取消。
- 输入[batch,3,256,256]，FP32，TF32/AMP关闭；batch=1/16，eval与inference_mode。两张GPU分别独立测全套，设备用PyTorch实际名称记录，不能假定nvidia-smi序号等于CUDA序号。
- 主Latency是每次调用后CUDA同步的墙钟p50/p95，预热30次，每轮100次、3轮。串行FPS=1000×batch/Latency；另测不在每帧同步的队列吞吐FPS。batch16吞吐不冒充单帧延迟。
- 输出单列OCBE、学生、OCBE＋学生、加top-k分数、教师以及完整模型。完整模型实测，不能用各模块median相加代替。
- FLOPs以一次乘加=2FLOPs，报告主要线性代数算子：矩阵乘法、卷积以及attention的QK/AV；由PyTorch原生FlopCounter统计，以独立公式交叉验证所有学生和教师。LayerNorm/激活/逐元素Rank2/softmax/topk/bias/数据搬运不在该数值内，但实际延迟包含所有推理算子。另给GMAC=GFLOPs/2，避免混用。
- 部分DGT训练实现每次前向都会把诊断量拷回CPU。主表副本仅删去诊断的统计和主机拷贝；先验证与原实现输出一致，原始含诊断耗时保留附表。共享训练代码不修改。
- Model Size同时给FP32 tensor payload与实际torch.save(state_dict)序列化长度（内存中测量），同时记录完整模型大小；单位MiB=2^20字节，另提供MB=10^6字节。排除优化器与训练日志，不导出假训练权重。
- 有现存checkpoint的版本严格加载bike_final.pth并记录SHA256；老版本缺权重时使用同构初始化，明确标注。架构无稀疏/动态路由，不重新测试AUC；历史AUC不被此次时延测量覆盖。

## 限制

该测试是GPU驻留输入的桌面GPU推理性能，不含读盘、解码、resize、H2D；不是无人机硬件时延。不同场景结构相同，因此统一按模型结构测量，不把七场景重复测量当成七个独立速度样本。旧库版本可能不同，结果是本次统一环境下的公平重测，不与旧latency JSON直接拼接。

运行前提交计划与源码。命令为DMAD Python执行scripts/benchmark.py --gpu 0和--gpu 1；GPU分开写入results/gpu0.json与gpu1.json，最后运行scripts/summarize.py。

用户在测量过程中明确模型范围为蒸馏外观分支；已统一full的命名。这仅调整报告定义，计时计算图和配置未改变，无需重跑。
