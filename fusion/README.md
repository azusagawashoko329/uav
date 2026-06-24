# 融合(fusion)

把**外观教师**(DINOv3-RD,`model/`+`train.py`+`eval.py`)与**行为分支**(`behavior/`,检测+5维描述子+kNN-OOD)
两条**相互独立**的分支,在**分数层面**做晚融合(late fusion)。不碰两分支的模型/训练,只读它们落盘的逐帧分数
(`experiments/fusion/scores/{app,beh}_<scene>.npz`)。本文档记录 2026-06 这一长串融合实验的**全部结果与机制**
(含大量 negative results——它们和正结果同样重要)。

---

## TL;DR — 最终方案

**每分支各自 test-rank → 路由(行为只参与"拥堵类"场景)→ mean 融合。** 7 场景均值 **84.67(纯外观)→ 88.06**,
增益几乎全来自 vehicle(外观短板 60.71 被行为补成 84.36)。

- **标准化**:每分支用 **test 自身分布的 rank**(平均秩,平局取中位)映射到 [0,1]。
- **融合算子**:**mean**(不是 max,见 §3)。
- **路由**:行为分支**只在拥堵类场景(vehicle、crossroads-s1)参与**;其余纯外观。这是**唯一**能让融合 > 纯外观的做法——
  所有"无先验盲融合"实测都 ≤ 纯外观(见 §5、§8)。

> 一句话教训:**无监督下,"融合要拿到增益"必须靠一个最轻的路由先验;盲融合(任何出分方式)都打不过纯外观。**

---

## 1. 两分支分数尺度诊断(`dump_*_scores.py` 打印的 anom/normal)

| 场景 | 分支 | normal mean | anom mean | **anom/normal** | 形状 |
|---|---|---|---|---|---|
| crossroads | 外观 RD-topk | 0.173 | 0.181 | 1.043 | 窄正态(std 0.018) |
| crossroads | 行为 kNN | 0.095 | 0.094 | **0.986** | 宽 |
| vehicle | 外观 RD-topk | 0.216 | 0.224 | 1.036 | 窄正态(std 0.047) |
| vehicle | 行为 kNN | 0.197 | 0.719 | **3.653** | 重尾(max 7.9) |

**注释**:① 两分支绝对量级其实接近(都 0.1~0.7),真正差异是**分布形状**(外观窄正态 / 行为重尾)→ 统一尺度要用
**百分位 rank**(把两者都压成 [0,1] 均匀),不能线性缩放、也不宜 z-score(行为重尾会让 z-score 尾巴主导)。
② **crossroads 在 concat 口径下两分支都不可分**(行为真信号在 s1,被 s2/s3 稀释到 anom/normal≈1)→ crossroads 的融合
价值只在 per-seq;**vehicle 是 concat 上唯一干净赢点**(行为 3.653)。

## 2. 标准化:必须用 test 自身 rank,**绝不能用 train-CDF**(关键 negative result)

`verify_percentile.py` 实测,外观分数做百分位标准化:

| 场景 | 原始 AUC | **test 自身 rank** | **train-CDF(用 train 分布做锚)** |
|---|---|---|---|
| bike | 98.08 | **98.08** | 57.69 |
| railway | 95.92 | **95.92** | 64.07 |
| solar | 86.11 | **86.11** | 63.39 |
| farmland | 99.47 | **99.47** | 78.59 |
| highway | 92.69 | **92.69** | 72.21 |
| vehicle | 60.71 | **60.71** | 50.00 |
| crossroads | 59.68 | **59.68** | 50.31 |
| **均值** | **84.67** | **84.67(Δ≈0)** | **62.32(−22.3)** |

**注释**:test 自身 rank 是**保序变换**→ 单场景 AUC **一字不变**。而 **train-CDF 全场景暴跌**,因为 **RD 外观分数有
强烈 train→test 漂移**:test 帧(连**正常**的)分数都整体超出 train 正常范围(bike train 上限仅 0.057,test 正常帧
84% 比它还高)→ 用 train 分布当 CDF 锚,test 几乎全部落在右尾外、被饱和到 1.0、正常异常糊一起。**结论:标准化锚取
test 自身,不取 train**(代价:transductive,但离线 frame-level AUC 评测本就是整段 test 一起算,合规)。详见 memory
`uav-vad-fusion-score-normalization`。

## 3. 融合算子:mean 不是 max(负相关下 max 被害)

两分支 test-rank 后是**负相关**的(crossroads corr=−0.25,vehicle corr=−0.55)——它们抓**不同的**异常帧(互补)。vehicle 实测:

| | 外观单分支 | 行为单分支 | **max 融合** | **mean 融合** |
|---|---|---|---|---|
| vehicle AUC | 60.71 | 71.79 | **70.59**(比行为还低!) | **84.36** |

**注释**:负相关下,**max 把两分支正常帧的单边假阳性取并集**→正常帧被顶高→AUC 掉;**mean 把正常帧的单边假阳性
中和、把异常帧的互补信号叠加**→分离度暴涨。所以融合算子用 **mean**。(早期 README 写 max,已纠正。)

## 4. 平均秩(rank 必须处理平局,否则是 bug)

rank 实现必须用**平均秩**(平局取中位名次),不能用 `argsort`(对相同分数按位置硬排)。无车场景(farmland/solar/
railway)行为 kNN 分数 99% 是平局(无车帧距离=0),`argsort` 把这一堆相同的 0 排成伪随机名次、凭空注入噪声:

| 全场景盲融合(mean) | farmland | solar | railway | **均值** |
|---|---|---|---|---|
| argsort rank(错) | 60.65 | 59.93 | 59.09 | **70.09** |
| **平均秩(对)** | **98.90** | **84.37** | **95.38** | **83.32** |

**注释**:平均秩让常数维→常数 0.5→`mean = 0.5·p_app + 0.25` 是单调变换→AUC 不变(do-no-harm)。这也印证了
一个核心命题(见 §8):**融合无损的真条件是"行为输出常数",不是"行为 AUC≈50"。**

## 5. 路由:盲融合是灾难,必须路由(均值对比)

全场景"行为也参与"的盲融合(平均秩 mean) vs 仅 vehicle 融合:

| 方案 | 均值 | vs 纯外观 |
|---|---|---|
| 纯外观(baseline) | 84.67 | — |
| 盲融合(全 7 场景 mean,平均秩) | 83.32 | −1.35 |
| 盲融合(全 7 场景 mean,**argsort**) | 70.09 | −14.58 💥 |
| **路由(仅 vehicle 融合)** | **88.06** | **+3.37** ✅ |

**注释**:7 场景里 6 个是"外观强 / 行为是噪声或反相关",mean 把行为的噪声平均进强外观→稀释→拉低。**只有 vehicle
是行为真有信号的拥堵场景**,所以只让它参与。路由依据是**异常的物理类型**(拥堵=行为类 vs 新物体/外观类),是
看任何 test 标签**之前**就能声明的先验,不是窥探 AUC。

## 6. 检测质量置信权重(用户提出的无监督路由替代,部分有效)

**动机**:行为分支严重依赖前置检测;检测烂(目标闪烁)→ 描述子全是噪声 → 行为不可信。用**帧间检测数波动**度量检测质量:

| 场景 | 均值车数 | 归一化抖动 jitter | 大跳变(≥3) | 高频翻转率 |
|---|---|---|---|---|
| **bike** | 7.0 | **0.256** | **25.9%** | **45.4%**(目标闪烁) |
| crossroads | 9.8 | 0.101 | 9.6% | 29.6% |
| vehicle | 3.7 | 0.059 | 0.1% | 7.8% |
| highway | 4.3 | 0.057 | 2.3% | 7.6% |
| farmland/solar/railway | ~0 | ~0 | — | — |

**置信权重**:用 **train 帧**算每场景 `jitter = mean(|Δn|)/mean(n)`(场景级、不碰 test);
`q = min(1, j0/jitter)`(j0=0.1=帧间车数变化<10% 算稳);`w_beh = 0.5·q`。融合 `(1−w)·p_app + w·p_beh`。

| 场景 | jitter | w_beh | 纯外观 | 检测质量加权融合 |
|---|---|---|---|---|
| bike | 0.290 | 0.17 | 98.08 | **96.70**(自动降权回升) |
| vehicle | 0.079 | 0.50 | 60.71 | **84.45**(满权重) |
| highway | 0.061 | 0.50 | 92.69 | **76.52**(仍崩,救不了) |
| 均值(highway 排除行为) | | | 84.67 | **87.44** |

**注释**:✅ 检测质量加权**自动**把"检测烂的 bike"降权(无需人为排除),又不误伤"检测稳的 vehicle"。
❌ 但它**救不了 highway**——highway 检测稳(jitter 0.061),q=1、不降权,而 highway-s2 的病是"描述子语义"
不是"检测质量"(见 §8)。**检测质量管"检测可不可信",管不了"描述子有没有判别力"。** bike/vehicle 都满车流,
区别在异常类型(bike=新车类→外观;vehicle=拥堵→行为),所以**路由只能按异常类型,不能按车流密度**。

## 7. 逐场景"三态"(per-seq,把分支在每个 seq 的角色看清)

每个分支在每个 seq 是 **负责 / 静默 / 反相关拖后腿** 三态之一(test-rank 后看正常 vs 异常帧的均值):

- **行为负责**(异常>正常):vehicle-s1(83.0)、vehicle-s2(82.3)。
- **行为静默**(正常≈异常≈0.50 一条线):farmland、solar、railway(无车)、highway-s3(0.36/0.36)。
- **行为反相关/噪声(拖后腿)**:highway-s2(AUC 12,正常 0.88>异常 0.77)、bike(检测噪声)、highway-s1 略噪。
- **外观也会反相关**:vehicle-s1 外观 AUC=35.6(正常 0.85>异常 0.80,外观在这反而拖后腿,但 s2 帮忙,concat 净正)。
- **crossroads**:两分支在 concat 上都≈chance(用户判定两分支都负责不了)。

> 关键:融合的真敌人是**反相关**(highway-s2 行为、vehicle-s1 外观),不是"无信号"——反相关比无信号更糟。

## 8. highway-s2 深挖:为什么救不了(一连串 negative result + 根本墙)

highway-s2 异常 = 近碰撞;行为 AUC=12(强反相关)。逐层排除:

1. **不是检测质量**:s2 的 train 检测 jitter 才 0.06~0.10(稳);q≈1 救不了。
2. **不是 seq 混合**:per-seq(只 s2 train)更糟(6.6<12),merged 没问题。
3. **是描述子 + 联合距离的"投票权"问题**。维度分解(各维单独 AUC):
   `d_min 84.7(正!) d_mean 71.3(正) n_close 11.7(反) count 24.5(反) max_local 50`。
   异常帧**车少**(count 9.7 vs 正常 23.6)→ 数量/密度维反相关;而真异常信号"两车贴极近"被 **d_min 抓到(84.7)**。
4. **点云状态**:n_close 原始范围 [0,215]、std 26.9(因 train 混了 s1 稀疏/s2 密集/s3 空)。z-score 后联合 kNN
   距离的各维**平方贡献**:`count 37.7% + max_local 26% + n_close 20% ≈ 84%`,而 **d_min 只 0.4%**——
   **唯一能识别近碰撞的 d_min 在联合距离里被高方差反相关维彻底淹没** → 整体反相关。

**出分方式(聚合)的对比**(highway-s2 / 全场景融合均值):

| 出分方式 | highway-s2 单独 | vehicle 行为 | 全场景**融合**均值 |
|---|---|---|---|
| 联合 5 维 kNN(现状) | 12.0 | 71.9 | 83.32 |
| 各维独立 rank → **mean** | 46.2(静默) | 67.3 | 80.60 |
| 各维独立 rank → **max** | 68.6 | 65.5 | 80.98 |

**注释(真实帧 trace)**——highway-s2 异常帧 `[d_min 0.04,…,n_close 10]` → 各维 rank `d_min=1.00, n_close=0.18`;
正常密集停车帧 `[d_min 0.01,…,n_close 101]` → `d_min=0.38, n_close=1.00`:
- **各维 mean**:异常 d_min 的 1.00 被其他维平均掉 → 0.49;正常 0.44 → **两者都≈0.45,静默**。
- **各维 max**:异常 max=1.00(靠 d_min)、**正常也 max=1.00(靠 n_close)→ 分不开**(累积假阳性)。
- **联合**:被 n_close 主导 → 正常帧距离大、高分 → 反相关。

**两堵无监督的墙**:
- **(a) 静默 ≠ 常数**:各维 mean 让 highway-s2 行为 AUC=46(静默),但它**不是常数**(rank 仍 spread,只是与异常无关的
  噪声)→ mean 融合**照样稀释**外观 → highway 仍掉(76.52→77.16)。只有**真常数**(farmland 行为=0.5)才融合无损。
- **(b) 离群方向**:正常"密集停车"和异常"近碰撞"在描述子空间**都是离群**(一个 n_close 离群、一个 d_min 离群),
  kNN-OOD 的假设"离群=异常"**分不清哪种离群才是异常**——这需要标签。任何出分方式(联合/mean/max)都逃不出。

**最终裁决**:三种出分的全场景盲融合(83.32 / 80.60 / 80.98)**全部低于纯外观 84.67**。
**"highway 不掉分" ⟺ "行为在 highway 权重→0" ⟺ "需识别 highway 该关掉行为"**,而检测质量(检测稳)识别不了、
描述子判别力无监督测不出 → **"highway 不掉分"与"完全无先验"不可兼得**。务实做法:highway 纯外观 92.69
(次高、不掉分)+ 写 limitation。要真救 s2 需要**时序**(区分静止停车 vs 逼近碰撞)或实例分割(近碰撞两车不并框),
都超出"单帧 + 几何描述子"框架。

---

## 文件

| 文件 | 作用 | Env |
|---|---|---|
| `dump_appearance_scores.py` | 导出外观逐帧分数(test;`--train` 额外导 train 分数,仅作漂移诊断,**不做标准化锚**) | base(`HF_HUB_OFFLINE=1`) |
| `dump_behavior_scores.py` | 导出行为逐帧分数(test;`--train` leave-self-out)→ `beh_*.npz` | DMAD |
| `verify_percentile.py` | 验证 test-rank 无损 / train-CDF 因漂移崩(§2) | base/DMAD |
| `fuse.py` | 读两分支 npz → test-rank → (路由) → mean → concat 对比表 | base/DMAD |

分数产物落 `experiments/fusion/scores/`(npz,已 gitignore)。逐场景 per-seq、维度分解、检测抖动、出分方式对比等
诊断脚本本轮以 inline 形式跑过,关键数字已固化在上文。

## 复跑

```
# 1) 落盘两分支逐帧分数(外观需离线缓存教师)
HF_HUB_OFFLINE=1 python fusion/dump_appearance_scores.py --scenes bike,crossroads,farmland,highway,railway,solar,vehicle --train --gpu 0   # base
python fusion/dump_behavior_scores.py --scenes bike,crossroads,farmland,highway,railway,solar,vehicle --gpu 1                              # DMAD
# 2) 标准化无损性自检
python fusion/verify_percentile.py
# 3) 融合(默认全场景 mean;路由/检测质量加权见上文,目前以 inline 验证为主)
python fusion/fuse.py
```

## 状态

- ✅ 标准化(test-rank,§2)、融合算子(mean,§3)、平均秩(§4)、路由必要性(§5)、检测质量置信权重(§6)、
  逐场景三态(§7)、highway-s2 死角(§8)全部跑通并固化。
- ✅ 最终方案:**test-rank + 路由(行为仅拥堵场景)+ mean = 88.06**(+3.37 vs 纯外观 84.67)。
- ⏭️ 把检测质量置信权重(§6)正式写进 `fuse.py`(目前 inline);per-video macro 口径补充;crossroads per-seq 价值的呈现。
- ⛔ highway-s2 / bike 类别异常 = 单帧行为分支的能力边界,记为 limitation(不再投入)。
