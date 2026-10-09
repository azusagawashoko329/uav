# none adapter 结果

七场景 frame-level ROC-AUC 算术平均：**83.9505**（seed 42）。

| scene | AUC | trainable parameters |
|---|---:|---:|
| highway | 95.5409 | 40,258,880 |
| farmland | 99.2376 | 40,258,880 |
| bike | 97.7067 | 40,258,880 |
| solar | 89.0936 | 40,258,880 |
| vehicle | 53.9801 | 40,258,880 |
| crossroads | 59.4296 | 40,258,880 |
| railway | 92.6651 | 40,258,880 |

逐帧分数和复核见同目录的 `*_scores.csv`；三变体对照见 [`../../20261009_student_adapter_results_s42.md`](../../20261009_student_adapter_results_s42.md)。
该结果为单 seed，不能据此宣称稳定收益。
