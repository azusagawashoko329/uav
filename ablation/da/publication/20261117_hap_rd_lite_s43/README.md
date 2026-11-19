# HAP-RD-Lite / DA / seed43

唯一配置：{"label": "HAP-RD-Lite", "depth": 4, "taps": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "width": 192, "hidden": 128, "order": "LLLG", "ffn": "gate", "rank2": true, "alpha": 0.5, "params": 3761222}。

数据为DA七场景，每场景独立从头初始化并训练30 epoch，batch64，256×256，AdamW(max_lr=5e-4, weight_decay=0.05) + OneCycleLR；teacher冻结，tap_norm=True，code_dim=64。

逐patch计算1-cosine并取最大5个距离的均值。4-tap模型对四tap平均；12-tap模型对四个相邻三层组计算(1-alpha)*mean + alpha*max，再四组平均。训练仅增加batch mean；测试直接保留同一match逐帧分数，越大越异常。标签为sequence*/test/<video>.npy，1为异常，场景内concat计算AUC×100和离散ROC EER，最后七场景宏平均。每stage的三个头分别对应三个教师tap，不复用同一个头输出。

运行入口为本目录新建scripts/run_scene.py → scripts/da_publication_scene.py → model/publication_rd.py；不调用历史variant/GL/Loss runner。每场景单独logs/<scene>.log，断点重启追加；每epoch保存优化器、调度器和RNG状态。

状态：completed；源码计划提交与 preflight 后，以记录中的精确 source_commit 完成运行。

B-ablation是从头训练的4-block架构，不是剪枝旧权重后的直接测试。C的alpha=0对四个等大小组取mean，数学上等于12层平均。默认超参数保持一致；不根据test标签选择模型或score方向。

完成后的[单种子结果](results/summary.md)、[三种子矩阵与分析](../summary.md)及[全量核验](../results/validation.json)。
