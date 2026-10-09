# 本实验脚本 —— 蒸馏对齐结构探索

这一组脚本是 2026-09 那轮"相邻三层怎么聚合成一个蒸馏目标"的探索产物, **不属于正式管线**,
不改 `model/` 与 `train.py`; 它们复用 `model/` 的组件, 自己搭评估。

所有脚本都能从仓库根直接跑:

```bash
python experiments/da/aggregation/20260928_agg_explore/scripts/<脚本>.py --gpu 0 ...
```

(脚本内部会把自己所在目录的上一级加进 `sys.path`, 所以 `from model import ...` 正常。)

## 两类脚本

**免训练探针** —— 只读冻结教师, 在正常帧上量特征几何, 用来在投入训练前问清前提。

| 脚本 | 回答什么 |
|---|---|
| `probe_group_agg.py` | 组内三层有多像? 均值还能反推出多少原层(保真度)? 逐层归一化承不承重? 合并后目标糊了多少? |
| `probe_group_gap.py` | 往图里贴一块造伪异常, 逐层量特征动了多少(异常位移); 合并之后这个位移还剩多少 |
| `probe_group_alt.py` | 换几种固定合并方式(按敏感度加权 / 层间差), 量同样的事 |
| `probe_group_auto.py` | 给每种合并加一项"还能不能反推回原层"的检查 —— 这一项揪出了"层间差/内部残差其实是在扔内容" |

**轻训练探针** —— 只训一个小聚合器(教师冻结), 分钟级。

| 脚本 | 回答什么 |
|---|---|
| `probe_agg_learn.py` | 用一个"自己的 loss"(正常相邻帧别动 + 伪异常要动)训聚合器。**文件头写了 v1 为什么塌、v2 怎么修的** |
| `probe_agg_kinds.py` | 固定住那个 loss, 横向比 5 种聚合结构(取平均 / concat+MLP / 跨层注意力 / 注意力+MLP / 空间卷积) |

**端到端** —— 真训学生, 用 `eval.py` 同一口径算 AUC。

| 脚本 | 回答什么 |
|---|---|
| `probe_e2e.py` | 单场景短训, **只换"学生要还原什么"**, 其余全同。变体表见文件头 |
| `probe_e2e_learned.py` | 两段式: 聚合器先用它自己的 loss 训好、冻结, 再训学生 |

## 跑满 7 场景

```bash
bash experiments/da/aggregation/20260928_agg_explore/scripts/run_full_scenes.sh <variant> --seed 42     # 双卡共享队列, 重的场景优先
python experiments/da/aggregation/20260928_agg_explore/scripts/summarize.py                             # 汇总成 results.csv + 表格
```

产物落在 `experiments/da/aggregation/20260928_agg_explore/`(raw 结果 + 逐场景日志), 那里的 README 有结论表。

## 变体速查(`probe_e2e.py`)

| variant | 目标怎么来 | OCBE 吃什么 | 学生 |
|---|---|---|---|
| `base12` | 4 个单层 `[5,7,9,11]` | 这 4 层 | **12 层**(复刻基线) |
| `old4tap` | 4 个单层 `[5,7,9,11]` | 这 4 层 | 4 层 |
| `rep4_ocbe4` | 4 个单层 `[2,5,8,11]`(三三分组每组取代表) | 这 4 层 | 4 层 |
| `rep4` | 4 个单层 `[2,5,8,11]` | 全 12 层 | 4 层 |
| `mean4_ocbe4` | 4 个组平均 | 这 4 个平均(同源) | 4 层 |
| `mean4` | 4 个组平均 | 全 12 层 | 4 层 |
| `rawmean4` | 4 个组平均(不逐层归一化) | 全 12 层 | 4 层 |
| `raw12` | 12 个单层全当目标 | 全 12 层 | 4 层 |

对齐约定(全部变体一致): **最浅的目标由学生最后一个 block 还原**, 逐层反着来。

## 已知的坑(写下来免得重踩)

- `probe_e2e.py` 的 `forward_variant` **不能加 `@torch.no_grad()`** —— 它同时服务训练和评测,
  加上会让 loss 没有梯度(踩过一次, 报 "element 0 of tensors does not require grad")。
- 变体里 `n_taps > depth` 会静默截断(`StudentDecoder` 的老问题), 所以自定义的
  `MultiHeadDecoder` 允许一个 block 挂多个头, 并在 `recon_at` 上显式指定分配。
- `probe_agg_learn.py` / `probe_agg_kinds.py` 里的聚合器输出都带一层 LN, 所以"起点等于均值"
  严格说是"起点等于 LN(均值)" —— 与表格里 `mean` 那一行(不归一化的均值)有细微差别。
