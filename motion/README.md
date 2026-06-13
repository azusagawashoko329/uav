# 运动分支(object-centric behavior anomaly)

外观分支(`model/` + `train.py` + `eval.py`,DINOv3 反向蒸馏)在 **vehicle / crossroads** 这类
"异常藏在车怎么动里"的场景上接近瞎猜。运动分支补这个位:**检测车 → 跨帧跟踪 → 在帧窗里建关系
行为 → 与外观晚融合(do-no-harm)**。完全独立于外观:不共享参数/梯度,只共享数据集和评测协议。
运行在 **DMAD conda 环境**(唯一装了 ultralytics 的);外观跑在 base。

## 文件

| 文件 | 作用 |
|---|---|
| `detector.py` | 冻结 VisDrone YOLOv8s,仅车辆类,逐帧给框 |
| `detector.py: track_clip` | 自己检测 + 喂**独立 `BOTSORT`** 跟踪器(绕开 ultralytics 源处理坑,见下) |
| `r1.py` | **R1**:关系描述子 + kNN-OOD 可分性验证(建模前 go/no-go) |
| `visualize.py` + `track_all.sh` | 跟踪可视化预览(每场景一个 mp4 → `tracking_preview/`) |
| `weights/`, `cache/` | YOLO 权重 / 跟踪缓存(均 gitignore) |

## 整体前向(目标设计)

```
帧 ─▶ YOLO车辆检测(冻结) ─▶ BOTSORT跟踪(逐clip重置) ─▶ tracks{id:(t,cx,cy,w,h,cls)} ─滤短轨迹
  ─▶ 因果窗 T=16 ─▶ 每对象token[运动学 ⊕ (L2+)DINOv3场景ctx] ─▶ 时空编码(时间×社交注意力,相对几何偏置)
  ─▶ forecast残差 ─▶ top-k over objects ─▶ 运动分m_t ─▶ 与外观a_t晚融合(z-norm + do-no-harm门控)
```
- **训练**:per-scene 合并正常子视频,只在正常上学联合动态(forecast loss),无异常标签。
- **冻结**:检测器、跟踪器、DINOv3、单应估计;**可训练**:仅时空编码 + forecast 头(每场景小模型)。
- **FP 护栏**:车辆类限定 → farmland/railway/solar/bike 无车 → 静默;关系-非密度 → 免疫合法车流增多。

## 关键工程结论(踩过的坑)

- **跟踪**:`model.track()` 把松散图片序列当独立图像、帧间关联失效(ID 33–61% 瞬移)。改成
  **自检测 + 独立 `BOTSORT.update()`** → 冷启动 0% 瞬移。检测本身很稳(相邻帧中位 5px)。
- **相机运动(逐子视频实测)**:静止 = bike/crossroads-s2,s3/vehicle;**移动** = highway 三个 seq
  (4–20px/帧)、crossroads-s1(3px/帧)、solar/farmland。移动序列里**静止车有 ~52% 是假轨迹**。
- **坐标系分层**:**L1 相对量**(成对间距/接近速度/TTC)= ego 不变,无需补偿/坐标统一/mosaic;
  **L2** 全局单应补偿得绝对速度(亚像素可靠);**L3** 逐窗 mosaic(场景拓展不移动)。**先 L1**。
- **窗口 T≈16**(范围 16–32):行为时长够、补偿漂移可控;因果滑窗→只损每子视频头 T−1 帧(外观兜底)。
- **DINOv3 尺度**:ViT-B/16,D=768,256→16×16 patch;**RoPE 分辨率无关**→可喂 640→40×40 更细场景 ctx 不用重训;场景 ctx 逐 patch L2-norm。

## R1 结果(2026-06-13,关系 kNN-OOD,L1,ego 不变,concat 协议)

| 场景 | 关系AUC(concat) | 逐视频均 | 密度基线 | 外观RD-topk |
|---|---|---|---|---|
| **vehicle** | **68.73** | 72.5 | 63.67 | 60.71 |
| crossroads | 49.88 | 60.6 | 51.83 | 59.69 |
| highway | 50.57 | 44.2 | 49.80 | **92.69** |

逐子视频:crossroads **s1=92.1、s3/01=78.7** 强、**s2=18.2 反向**;vehicle s1=83.9/s2=61.2;
highway s1=57.8、**s2=24.7 反向**、s3=50.0。

**诚实结论(部分验证,非全胜)**:
1. **vehicle 清楚赢**(关系 > 外观 + 密度)——外观瞎掉处关系补上,且比纯车数多 +5(关系 not 密度)。
2. **crossroads 信号在但被 concat 冲洗**(s1/s3 很强,s2 反向 + 跨视频分数偏移)→ 打分一致性问题,学习式 per-scene 模型应能救回。
3. **highway 关系没用,外观碾压**(92.7)→ 符合互补;但 s2 近碰撞**反向**待查。
4. **两个反向(crossroads/s2、highway/s2)是疑点**:异常可能根本不是关系类(需 L2 场景相对),或跟踪/描述子没抓到。**未查清。**
5. **R1 没有学对象-场景关系**(只对象-对象)→ 压线/单车违规这类抓不到,是 L2(DINOv3 场景 ctx)的事。

## 状态 / 下一步

- ✅ 前端(检测+跟踪)、R1 验证。
- ⏳ 查两个反向子视频(异常非关系 vs 没抓到);决定是否进学习式时空模型 / 是否爬 L2。
- 复跑 R1:`/home/lyx/miniconda3/envs/DMAD/bin/python motion/r1.py --scenes crossroads,vehicle,highway`(命中 `cache/` 秒出)。
