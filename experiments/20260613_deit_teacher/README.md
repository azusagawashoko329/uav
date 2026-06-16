# 教师消融:DeiT(MADViT 教师) vs DINOv3(20260613_deit_teacher)

**动机**:组会质疑 RD 均值 84.67 vs MADViT 74.19 的 +10.5,是否只来自把教师从 DeiT 换成自监督的 DINOv3。

**做法**:在 RD 管线里**仅**把教师换成 MADViT 自用的 `deit_base_distilled_patch16_224`(timm pretrained,
224→196 patch),其余 **OCBE / student / top-k 损失 / top-k 读数 / taps=[5,7,9,11] / 配方(AdamW wd5e-2 +
OneCycleLR max5e-4 + bs64 + ep15)/ per-scene 合并训 + global-concat 协议逐字节不变**。DeiT 与 DINOv3 同为
ViT-B/16·768·12层·~86M(**容量对等**),唯一变量 = 预训练(DeiT ImageNet 监督 vs DINOv3 1.7B 自监督)。

**新增文件(零改动已有文件)**:
- `model/teacher_deit.py`:镜像 `DINOv3Teacher` 接口的 DeiT 纯提取器,返回 raw 768 维 block 特征。**不引入**
  MADViT 自带的 `latent_dim=192` 随机冻结投影——那是 MADViT 方法的一部分、非"教师权重",对齐照旧由 OCBE/student 在学生侧做。
- `train_deit.py` / `eval_deit.py`:仅 monkey-patch 教师工厂、其余复用 train.py/eval.py 的薄 wrapper;锁 img=224。

**结果(topk,全局 concat,与 baseline 同口径)**:

| scene | RD-DINOv3(baseline) | RD-DeiT | Δ(DeiT−DINOv3) |
|---|---|---|---|
| highway | 92.69 | 95.12 | +2.43 |
| farmland | 99.47 | 97.06 | −2.41 |
| bike | 98.08 | 96.98 | −1.10 |
| solar | 86.11 | 88.91 | +2.80 |
| vehicle | 60.71 | 62.18 | +1.47 |
| crossroads | 59.68 | 56.24 | −3.44 |
| railway | 95.92 | 94.86 | −1.06 |
| **均值** | **84.67** | **84.48** | **−0.19** |

**三方对照(直接回答质疑)**:

| 教师 | 方法 | mean(7) |
|---|---|---|
| DeiT (deit_base_distilled) | MADViT(其最佳 Last-layer) | 74.19 |
| **DeiT (deit_base_distilled)** | **RD (ours)** | **84.48** |
| DINOv3 ViT-B/16 | RD (ours) | 84.67 |

- **控制教师一致(都用 DeiT)**:方法 MADViT 74.19 → RD **84.48**,**+10.3**(同教师下 railway +44.0、bike +16.5、
  farmland +7.2 碾压,仅 crossroads −4.1)。
- **控制方法一致(都用 RD)**:教师 DeiT → DINOv3,84.48 → 84.67,仅 **+0.19**。
- **结论**:增益**压倒性来自方法**(RD + 逐patch top-k 读数,+10.3),**换教师几乎零贡献(+0.19)**。DINOv3 只是更优默认,不是性能来源。

**插曲**:首轮 solar/bike 撞 huggingface.co 间歇 SSL 断连(timm 每次 `create_model(pretrained=True)` 都发 HEAD
查权重更新),崩在加载教师(没存盘)。权重已缓存,设 `HF_HUB_OFFLINE=1` 离线补训成功收敛。**复跑务必带 `HF_HUB_OFFLINE=1`。**

**复跑**:
```
export HF_HUB_OFFLINE=1
for s in railway solar vehicle crossroads farmland highway bike; do
  python train_deit.py --scene $s --gpu 0 --exp 20260613_deit_teacher
done
python eval_deit.py --exp 20260613_deit_teacher --gpu 0
```
