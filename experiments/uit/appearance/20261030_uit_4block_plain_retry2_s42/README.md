# 4block plain StudentDecoder depth4 + Dense512，UIT-ADrone，seed42

## 研究问题

在 UIT-ADrone 上重新跑关键轻量化版本，确认 Drone-Anomaly 上的参数与 AUC 变化是否保持。

## 固定设置

- DINOv3 ViT-B/16 冻结，12 个 teacher patch taps，256×256 输入；
- UIT 官方 train/test 布局，不去重 test 中重复出现的 train 帧；
- seed42、30 epoch、batch64、AdamW、OneCycleLR、top-5 group-balanced cosine loss；
- 五个版本共用同一套读取、标签、评测和动态双卡调度；
- 仅运行外观反向蒸馏分支，不运行行为分支。

## 评测

所有 51 个 test 视频逐帧分数拼接后计算单个 AUC 和 EER。

## 限制

UIT 官方协议存在 train/test 帧重叠，结果只在该协议内与历史 MADViT/DABG 口径比较；单种子，不能替代多种子稳定性结论。

## retry2 说明

retry1 在训练过程中外部中断且没有最终结果；本目录是同一固定配置的独立重跑，使用脱离会话的后台调度，避免终端会话结束影响训练。
