# ASTT（variant unconfirmed；当前数字不是官方 ASTT 效率）

- 论文：Tran et al., *Transformer-Based Spatio-Temporal Unsupervised Traffic Anomaly Detection in Aerial Videos*, IEEE TCSVT 34(9), 2024, DOI `10.1109/TCSVT.2024.3376399`。
- 官方代码：`/data1/data8tb/liangyuxuan/Comparison method/ASTT/`（HEAD `1d05305a94fb47b33897c7aa085c5351bcbbc8f9`）。
- 当前 benchmark：`benchmark_v2.json` 使用 `model_ste_tte.VisionTransformer`，148.220M 参数，输入 384×384×4。

## 为什么不能当官方 ASTT

1. `train_val_Drone-Anomaly.py` 写的是 `from model import VisionTransformer`，但 repo 中**没有 `model.py`**；只有 `model_original.py`、`model_baseline_ca.py`、`model_ste_tte.py`。
2. 三个候选模型的参数量差异很大（按 DA 脚本 `image_size=384, patch=16, emb=768, mlp=1536, heads=6, layers=2, frames=4` 构建）：

| 文件 | Params | 证据 |
|---|---:|---|
| `model_original.py` | **63.068M** | `train_val_Drone-Anomaly_original.py` 使用 |
| `model_baseline_ca.py` | **65.409M** | 唯一包含 `CrossAttention` 的候选；DA 脚本 save path 含 `self_cross` |
| `model_ste_tte.py` | **148.220M** | `train_val_UIT_ADrone_ste_tte.py` 使用；timm `vit_base_patch32_384` + temporal transformer |

3. 当前 benchmark 用的是 `model_ste_tte`，它对应 **UIT-ADrone STE-TTE** 脚本，不是 DA 脚本能直接证明的模型。`model_ste_tte` 的 `timm vit_base_patch32_384` 是 patch 32；`model_original` 是 patch 16，结构不同。
4. 官方 `train_val_Drone-Anomaly.py` 还把 `model_original` 的 import 注释掉；DA 实际使用的 `model.py` 没有发布。

## 当前数字的准确含义

| 项 | 值 |
|---|---:|
| Input | 384×384×4 |
| Params | 148.220M |
| FLOPs | 109.416G |
| P50 | 19.048ms |
| FPS@BS1 / BS16 | 52.2 / 146.6 |
| 评分 | `F.mse_loss(out, zeros)` 代理，不是官方 target MSE |
| Status | `partial/unverified`（variant unconfirmed） |

这些数字只能说明“`model_ste_tte` 这个 STE-TTE 图在这张 5880 上有多快”，**不能**写进主表当作 DA ASTT 的官方效率，也不能和论文 AUC 直接联合。

## 需要什么才能修正

- 拿到 DA 脚本实际使用的 `model.py`（作者未发布）或论文补充材料中的结构；
- 若 `model.py` = `model_baseline_ca.py`，重测 65.409M 版本；若 = `model_original.py`，重测 63.068M 版本；
- 同时把评分从 `MSE-to-zero` 改成官方 target MSE / PSNR 路径。

在拿到上述证据前，记录保持 `partial/unverified`，详见 [../../fairness_audit.md](../../fairness_audit.md)。
