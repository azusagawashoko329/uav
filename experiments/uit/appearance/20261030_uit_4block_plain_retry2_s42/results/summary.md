# 4block + Dense512：UIT-ADrone，seed42，30 epoch

AUC **81.7450%**，EER **25.9195%**，OCBE + Student 参数 **40.258880M**。

完整 30 轮，逐帧指标复算、标签对齐、权重哈希、严格加载均通过。训练日志耗时 3.172 小时，不能作为推理速度对比。

[五版本完整对比](../../20261030_uit_gl192_dense128_rank2_retry2_s42/results/five_variant_summary.md)；[核验记录](verification.json)；[逐轮耗时](training_timing.csv)。

最终权重：`ckpt/uit_final.pth`（本地保存，不入 Git）。
