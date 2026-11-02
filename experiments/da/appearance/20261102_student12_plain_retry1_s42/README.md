# 20261102_student12_plain_s42

DA 七场景、seed42、30 epoch。唯一变量是学生结构：StudentDecoder depth12 width768。教师 12 taps、Dense512 OCBE、top-k loss、宽度768和其余训练设置保持一致。每场景独立训练并输出 AUC/EER。
