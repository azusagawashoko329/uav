# channel adapter 结果

七场景 frame-level ROC-AUC 算术平均：**84.1148**（seed 42）。

| scene | AUC | trainable parameters |
|---|---:|---:|
| highway | 96.0690 | 40,708,448 |
| farmland | 99.2393 | 40,708,448 |
| bike | 97.7025 | 40,708,448 |
| solar | 88.9270 | 40,708,448 |
| vehicle | 54.7743 | 40,708,448 |
| crossroads | 59.2854 | 40,708,448 |
| railway | 92.8060 | 40,708,448 |

逐帧分数和复核见同目录的 `*_scores.csv`；三变体对照见 [`../../20261009_student_adapter_results_s42.md`](../../20261009_student_adapter_results_s42.md)。
该结果为单 seed，不能据此宣称稳定收益。
