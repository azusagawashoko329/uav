# HAP-RD w/o LAGA / DA / seed44

过程消融别名：d-new-ablation、LGGD without LAGA。固定版本：pub_lggd_without_laga_20261119；不是新的正式模型。

与同seed的 [HAP-RD](../../../../ablation/da/publication/20261117_hap_rd_s44/config.json) 比较。仅改变训练与match测试中的 α：0.5→0；七个场景均从头训练。四个等大小三层组在α=0时等价于12层平均。结果是否改善是待检验假设，不预设与其他decoder模板一致。

冻结DINOv3 ViT-B/16，teacher taps0..11，LLLG+gate ratio2、student width768、12个独立重建头、PDFB hidden512、code_dim64、tap_norm=True，无PODP。可训练参数30,248,000。四stage分别用三个独立头重建teacher9–11、6–8、3–5、0–2。

仅DA；七场景独立训练，30 epoch、batch64、workers8、256×256输入（256 patches）；AdamW(max_lr=5e-4, weight_decay=0.05)+OneCycleLR，deterministic cuDNN、train drop_last。Top-5 patch cosine distance→组内mean→四组mean。训练取同一ablation_match的batch mean，测试保留每帧score。分数越大越异常；sequence*/test/<video>.npy为标签，1=异常；场景内concat计算AUC×100及离散ROC EER，再七场景等权宏平均。

专用入口：本目录scripts/run_scene.py→scripts/da_laga_scene.py→model/lggd_no_laga.py。启动前核验实际模型、梯度、头路由及α=0评分等价性；独立preflight代码哈希固定。初次运行先提交计划，record中的source_commit由调度器绑定实际干净HEAD，历史模型和原结果不覆盖。

动态双卡按训练帧数优先的一条跨seed队列；每卡一个本轮任务，完成立即补位，不做外部占用门控。调度器以独立session运行、stdin关闭，忽略SIGHUP。每场景单一logs/<scene>.log；每epoch保存模型、优化器、调度器和全部RNG，恢复从已完成epoch继续，已有完成结果跳过，失败日志保留。权重由Git忽略，产物索引记录哈希。

[版本配置](architecture_version.json)，[运行记录](record.json)。三seed运行计划及preflight位于seed42目录。

状态：completed。21个场景任务全部完成；本目录七场景完整训练与测试结束。[完成后的三种子结果分析](../20261119_laga_lggd_a0_s42/results/laga_comparison.md)，[该seed结果](results/summary.md)。
