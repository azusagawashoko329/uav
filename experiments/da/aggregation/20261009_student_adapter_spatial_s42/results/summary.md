# spatial adapter 结果

七场景 frame-level ROC-AUC 算术平均：**83.9376**（seed 42）。

| scene | AUC | trainable parameters |
|---|---:|---:|
| highway | 95.0708 | 40,711,328 |
| farmland | 99.1547 | 40,711,328 |
| bike | 97.7102 | 40,711,328 |
| solar | 88.9266 | 40,711,328 |
| vehicle | 53.3208 | 40,711,328 |
| crossroads | 60.5900 | 40,711,328 |
| railway | 92.7898 | 40,711,328 |

逐帧分数和复核见同目录的 `*_scores.csv`；三变体对照见 [`../../20261009_student_adapter_results_s42.md`](../../20261009_student_adapter_results_s42.md)。
该结果为单 seed，不能据此宣称稳定收益。
