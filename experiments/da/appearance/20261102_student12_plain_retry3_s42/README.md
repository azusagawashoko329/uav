# 20261102_student12_plain_s42

DA 七场景、seed42、30 epoch。唯一变量是学生结构：StudentDecoder depth12 width768。教师 12 taps、Dense512 OCBE、top-k loss、宽度768和其余训练设置保持一致。每场景独立训练并输出 AUC/EER。

本次 retry2 修正为历史 baseline 的四个 taps `[5,7,9,11]`，Plain12 与 GL12 完全同 taps。

本次 retry3 固定历史 baseline 对齐：OCBE n_taps=4，学生 12 blocks 在位置 [2,5,8,11] 输出 4 个重建 head。
