# 20260612_motion_probe1 — 运动信号 go/no-go 探针

> 此为迁移前的历史说明，旧路径/命令可能失效。当前状态见 [RECORD.md](RECORD.md)，旧新路径见 [迁移表](../../../../docs/MIGRATION_MAP.csv)。

**问题**: 冻结单帧 DINOv3 特征里有没有"可学的"时序/运动信号? (你 MADViT 那次用免训练逐点算术证了"逐点无关",但那不等于"无可学结构"。)

**做法**: masked-target 时空 transformer, 给 W=4 帧预测第 5 帧的 patch 特征(冻结 DINOv3 block7), 只在正常帧训练; 异常分 = 逐帧 top-5 余弦预测误差。per-scene 合并训练 + global-concat 出 AUC, 与外观分支同协议直接可比。

**判读**:
- crossroads/vehicle 顶过外观天花板(~60) -> 单帧特征有运动信号 -> 运动分支可共享 DINOv3 骨干(方案A)
- 趴在 ~50-60 -> 没有 -> 论证"必须上视频教师"(方案B), 成为 ablation
- railway 对照(静态白粉外观异常): 运动预测器应抓不到 -> 验证探针抓的是运动而非外观

脚本: ../../probe_motion_v1.py | 产物: run.log, results.csv
