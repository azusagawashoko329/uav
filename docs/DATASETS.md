# 两个数据集及实验标识

项目实验只使用下表两种数据集。数据本体在仓库外，不提交 Git。以下路径是本机观测位置，换环境时用运行配置记录真实根目录。

| ID | 数据集 | 本机路径 | 划分与指标 |
|---|---|---|---|
| `da` | Drone-Anomaly | `/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly` | 七场景，场景下 `sequence*/train/*` 与 `sequence*/test/*`；每场景合并各 test 子视频逐帧分数计算 AUC，再列七场景算术均值。具体运行须记录标签读取及聚合口径。 |
| `uit` | UIT-ADrone | `/data1/data8tb/liangyuxuan/dataset/UIT-ADrone` | `train/frames` 与 `test/frames`，标签在 `test/test_frame_mask`；整个 test 的逐帧分数汇总为单个 AUC/EER。 |

Drone-Anomaly 的七个场景键：`bike`→`Bike Roundabout`，`crossroads`→`Crossroads`，`farmland`→`Farmland Inspection`，`highway`→`Highway`，`railway`→`Railway Inspection`，`solar`→`Solar Panel Inspection`，`vehicle`→`Vehicle Roundabout`。场景键是路径/文件名约定，不代表相同场景可共享权重。UIT 没有这七场景分桶，不能把两个数据集结果直接求“八场景均值”。

旧笔记 `note/方法对比总表.md` 记录 UIT 官方划分的 train 视频帧会出现在 test 全片中，以及这一协议对绝对 AUC 的影响。任何 UIT 新实验必须记录所用划分、是否去重、标签处理和评测口径，不能默认与 Drone-Anomaly 的不相交划分等价。此处只记录数据布局与报告要求，不宣称重新审计了官方原始数据。
