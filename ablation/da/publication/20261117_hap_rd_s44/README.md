# HAP-RD / DA / seed44

唯一配置：{"label": "HAP-RD", "depth": 4, "taps": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "width": 768, "hidden": 512, "order": "LLLG", "ffn": "gate", "rank2": false, "alpha": 0.5, "params": 30248000}。

数据为DA七场景，每场景独立从头初始化并训练30 epoch，batch64，256×256，AdamW(max_lr=5e-4, weight_decay=0.05) + OneCycleLR；teacher冻结，tap_norm=True，code_dim=64。

逐patch计算1-cosine并取最大5个距离的均值。4-tap模型对四tap平均；12-tap模型对四个相邻三层组计算(1-alpha)*mean + alpha*max，再四组平均。训练仅增加batch mean；测试直接保留同一match逐帧分数，越大越异常。标签为sequence*/test/<video>.npy，1为异常，场景内concat计算AUC×100和离散ROC EER，最后七场景宏平均。每stage的三个头分别对应三个教师tap，不复用同一个头输出。

运行入口为本目录新建scripts/run_scene.py → scripts/da_publication_scene.py → model/publication_rd.py；不调用历史variant/GL/Loss runner。每场景单独logs/<scene>.log，断点重启追加；每epoch保存优化器、调度器和RNG状态。

复制自 ablation/da/GL/20261106_lllg_gate_s44；原始结果与权重不改写，原配置、record和产物索引保留于provenance/，完成性和模型等价性须通过publication_preflight。

B-ablation是从头训练的4-block架构，不是剪枝旧权重后的直接测试。C的alpha=0对四个等大小组取mean，数学上等于12层平均。默认超参数保持一致；不根据test标签选择模型或score方向。
