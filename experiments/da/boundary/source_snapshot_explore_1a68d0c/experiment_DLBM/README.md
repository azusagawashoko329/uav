# DLBM：Probe → Boundary → Fresh Final

当前目录只实现外观分支的边界训练功能，不接入融合，也不使用测试异常标签。

## 数据流

1. **Probe**：每个场景用 `L0..L11` 和 `tap_align=absolute` 独立训练 10 个 epoch；各场景权重可被后续边界实验复用。
2. **Boundary**：每个场景独立学习一个连续后缀边界 `k∈[0,11]`；所有版本只读复用同一组
   10-epoch Probe，不跨场景共享 DLBM 权重。
3. **Fresh Final**：读取场景边界，仅用连续后缀 `[k,...,11]`，以独立 seed 从头初始化并训练
   OCBE/Student。Probe 权重不会传给 final。

边界 `k` 始终表示 `[k,...,11]`，不允许十二个互不相关的独立 Gate。

## 已归档的失败版本

`DLBM_v001_topk_counterfactual` 使用训练一致的逐帧逐层 top-5 平均距离、共享序列 evaluator，
以及 normal-only suffix 更新与 held-out normal/pseudo 风险之间的一阶梯度对齐。公式和真实
one-step 在非饱和场景中一致，但七场景粗边界 sanity 未通过，且 evaluator 对 CutPaste 饱和后
梯度低至 float32 不可分辨量级。因此该版本仅保留为失败实验，不进入 predictor 或 Final。

旧 `model/dlbm.py` / `train_boundary.py` 是更早的共享七场景 v1，也不符合当前“每场景独立模型”
定义，不应作为新实验入口。

`DLBM_v002_exact_suffix_gain` 实现了可以继续复用的 exact suffix supernet：all-12 OCBE 第一层
按边界切片后与独立 fixed-suffix OCBE/Student 数值等价。但其静态 Normal/Pseudo pairwise 风险
Oracle 在七场景集中为 L5/L6，DLBM 在 30 epoch 下也不优于逐场景常数均值基线，因此该目标与
predictor 均标记失败；不能把 ep30 偶然出现的 L4/L5 当作有效场景差异。

`DLBM_v003_micro_adapt_target_probe` 进一步排除了“候选 suffix 没有实际训练”的解释：从同一个
Probe10 起点构造 L0/L4/L8/L11 四个严格等价的 fixed-suffix 模型，各自在正常训练集上微调 5 个
epoch，再在固定 held-out normal/pseudo 集上评价同一 pairwise 风险。最终 6/7 场景选择 L8，且
highway/crossroads 不能复现外部人工 Gate 的粗边界。该结果说明延长 suffix 训练仍只强化统一的
深层偏置；当前风险定义本身不具备所需场景区分度，不能通过继续增加 DLBM epoch 修复。

`DLBM_v004_mask_localization_target_probe` 不再使用 Normal-Pseudo 标量差作为增益答案。四个 coarse
suffix 先完成相同的 normal-only 微调，再由一个跨 boundary 共享的单层序列编码器直接读取逐 patch、
逐层误差图，学习定位 CutPaste 的已知空间支持。held-out patch AUC 只用于验证候选 opening target
是否具有场景区分度；人工 Gate 粗边界仍只在实验结束后作局外 sanity check。该版本是 target probe，
验证通过前不拟合 DLBM predictor，也不进入 Final。七场景 20 epoch 结果为六个 L0、一个 L4，仅
highway 与外部粗边界一致；5/10 epoch 同样主要选择 L0。该 target 学到的是浅层局部空间定位优势，
而非 opening 的异常泛化增益，因此也已标记失败。

`DLBM_v005_gdas_bilevel_rank` 删除额外的差异编码器。每个场景只学习 L0/L4/L8/L11 上的一个离散
边界分布；每个 episode 从不变的 Probe10 参数出发，在 support normal 上执行一次完整可微 RD virtual
update，再以 query 的两个 within-frame pseudo 视图之间的面积排序 RankNet loss，通过 exact
hyper-gradient 更新 hard-Gumbel boundary logits。Probe 权重不被二阶段覆盖，Normal-Pseudo delta、
mask localizer、人工 Gate 标签和显式 gain 公式均不进入训练。

七场景 50 epoch 结果为六个 L8、solar=L4，与人工粗边界匹配 4/7。固定动作审查证明策略与其
post-update RankNet oracle 7/7 一致，但七场景的 pre/post oracle 也 7/7 完全相同；一步 inner update
对所选目标的改善仅约 `-7.5e-7 .. 5.8e-5`。因此 GDAS/hyper-gradient 实现有效且训练充分，但当前
outer loss 主要选择 Probe 初始静态 pseudo 排序，并没有识别 opening 的训练增益；该 target 不进入
Final。

## Probe 运行

七场景正式 Probe 已按 seed42、10 epoch、all-12、absolute alignment 完成：

```bash
bash experiment_DLBM/launch_probe10.sh
```

`DLBM_v001_topk_counterfactual` 的复现实验入口为 `launch_boundary_v001.sh`。它只运行 Boundary
gain probe，不训练 Final；该版本已判定失败，不建议重复运行，除非做明确消融。

当前开发产物使用一个可复用 Probe 根目录；DLBM 版本不得重复训练 Probe：

```text
experiment_DLBM/run/
├── probe/
│   ├── ckpt/<scene>.pth
│   ├── logs/<scene>.log
│   ├── status/<scene>.done
│   └── COMPLETE
└── boundary/
    ├── DLBM_v001/
    ├── DLBM_v002/
    └── ...
```

七场景 10-epoch Probe 后台入口为 `launch_probe10.sh`；它只运行正常 RD，不启动 DLBM 或 final。
后续每个 `DLBM_vxxx` 只读 `run/probe/ckpt/`。

快速机械 smoke 可给三个阶段分别传入 `--*_max_steps 1`，并缩小 batch 和图像尺寸。正式实验不应
复用 smoke 目录。

建议实验名使用 `_smoke_*`、`small_*`、`full_*`、`ablation_*`、`control_*`、`failed_*` 和
`candidate_*` 前缀区分用途。`runs/` 已被 Git 忽略；无保留价值的 smoke 产物应在检查后删除。
