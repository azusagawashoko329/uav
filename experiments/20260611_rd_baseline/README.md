# 20260611_rd_baseline

第一个反向蒸馏基线。**一句话**:冻结 DINOv3 教师 → OCBE 瓶颈 → transformer decoder 学生重建教师 tap 特征,异常 = Topk 逐 patch 重建差异。

## 配置
见 `config.json`。关键:taps=[5,7,9,11], code_dim=64, depth=12, k=5, img=256, epochs=15。
**训练配方(对齐 MADViT 成熟配方)**:AdamW + weight_decay=5e-2 + OneCycleLR(max_lr=5e-4) + bs=64 + workers=8。
(bs64 峰值显存仅 ~13GB/48GB。)

## 目录约定(所有实验照此)
```
experiments/<date>_<name>/
├── config.json     # 本批实验的全部超参(唯一权威)
├── README.md       # 说明 + 状态
├── ckpt/           # 每序列一个权重: <scene>_<seq>.pth
├── logs/           # 每序列一份训练日志: <scene>_<seq>.log
├── run.log         # 启动器总日志(各 job 起止)
└── results.csv     # eval 回填: scene,seq,auc,...(训练阶段为空)
```
**换超参 = 新建一个 `<date>_<name>/` 文件夹,不要往旧的里塞。**

## 本批跑的 7 个场景(per-scene,合并各场景所有 seq —— 与 MADViT 一致)
| scene | 合并 seq | train 帧 |
|---|---|---|
| crossroads | s1+s2+s3 | 15772 |
| farmland | s1 | 9548 |
| highway | s1+s2+s3 | 9045 |
| bike | s1 | 7950 |
| vehicle | s1+s2 | 5266 |
| solar | s1+s2 | 2848 |
| railway | s1 | 1206 |

**调度**:动态共享队列,两卡谁空谁领下一个场景(flock 原子),重活先排;只要有任务绝不空卡。

## 复跑
```
bash run_experiment.sh 20260611_rd_baseline
```

## 状态
- [ ] 训练(进行中,见 run.log / logs/)
- [ ] eval(待 eval 脚本)
