# Comparison v2 公平性复核

- 审计基线：`2b117ee37ed67b5bb1a7659a7de1162a0b47e557`（分支 `exp/da-variant-matrix-20261103`）。
- 范围：`comparison/da/` 主表及外部方法；只做核实、参数修正和文档修订，未重跑 benchmark，未覆盖任何 `benchmark_v2.json`。
- 目标：判断“效率图 + 输入/评分口径”是否与论文/用户表 AUC 对应的模型配置一致，而不是把“测速完成”直接当成“paper-matched”。

## 0. 先定义两个概念

| 概念 | 定义 | 判定依据 |
|---|---|---|
| `measured` | v2 测速协议完成：同一图、参数量三列、FLOPs 有计数说明、BS1/BS16 原始耗时保存 | protocol v2 的 DoD；**不要求**与论文 AUC 配置一致 |
| `paper-matched` | 被测的效率图、输入尺寸/帧数、预处理、评分路径和权重配置，与产生该 AUC 的模型配置一致 | 官方代码/论文/配置/source commit 的明确证据；仅“看起来像”不算 |

因此存在三种常见状态：`measured + paper-matched`、`measured + not paper-matched`、`measured 但评分/结构不完整`。本次审计只把第一种视为可以直接与 AUC 联合排序的结果；其余必须加脚注或重测。

证据等级：

- **官方结构一致**：能从官方 repo 的模型文件、测试脚本和配置直接对上；
- **适配结构**：官方代码有缺失/不可执行，做了明确的最小适配（必须标 `adapted`）；
- **未确认**：repo 缺少入口/配置/权重，或不同候选文件参数差异很大，无法证明 AUC 对应的就是当前被测模型。

## 1. 结论摘要

| 方法 | 当前效率图 | 官方结构判断 | 评分路径 | 与 AUC 配置匹配 | 建议 |
|---|---|---|---|---|---|
| Ours V0–V5 | 内部 variant + top-k readout | 官方（ours） | 完整（top-k） | 结构匹配；效率用 architecture-init | 可用于主表（权重脚注） |
| GANomaly | G 推理图 | 官方 G | 完整（latent MSE） | 输入 256 未在 repo 确认 | 需脚注 |
| MemAE | 3D AE + memory，T=16 | 官方 | 完整（重建误差） | 通道数 3 未在 repo 确认 | 需脚注 |
| MKD | VGG16 source + cloner B，128×128 | 官方 MVTec 配置 | 完整（taps 6/9/12 cosine+MSE） | **否**：repo 无 DA 配置，当前是 MVTec 128 | 需脚注；若 DA 分辨率不同需重测 |
| ANDT | 6 帧 + `de_dense(feat[:,0])` | **适配** | `MSE-to-zero` 代理，不是 target MSE | **未确认** | 保持 adapted；官方 decoder 需重测 |
| Future Frame | G only，prediction tensor | 官方 G | **Prediction-only**，缺 PSNR | 推理结构匹配；DA `LAM_ADV` 未确认 | 需脚注；score-inclusive 需重测 |
| ST Dissoc. | stage E：static/motion/cluster map | 官方 stage E | **缺最终 `loss_map` 评分** | **否/未确认** | 需脚注；最终评分需重测 |
| MLEP | prediction net | 官方 prednet | **Prediction-only**，缺 PSNR | 推理结构匹配 | 需脚注；score-inclusive 需重测 |
| ASTT | `model_ste_tte`，148.220M | **未确认，且很可能不是 DA 模型** | `MSE-to-zero` 代理 | **否/未确认** | **暂不可用，需重测** |
| DAD-FSM | detector（RGB+seg），224×224×4 | 官方 detector | `MSE-to-zero` 代理，不是 seg target | 分割图是离线预计算 | 需脚注（precomputed seg）；完整在线成本需重测 |
| HSTforU | Avenue/PVT2-b2 | **未确认**（repo 无 DA config） | `MSE-to-zero` 代理，不是 PSNR | **未确认** | 需脚注；DA 配置需重测 |
| MADViT Full | code default all-layer+token | 官方 | 完整 | 是（Full） | 可用于主表 |
| MADViT Multi-layer | 共享前向，只改 scoring | **reconstructed** | 重建 scoring | 论文变体未提供独立权重/开关 | 需脚注（reconstructed） |
| MADViT Last-layer | 共享前向，只改 scoring | **reconstructed** | 重建 scoring | 同上 | 需脚注（reconstructed） |
| LightViT-AD | teacher+6-block student，224 | 官方 config | 完整（token MSE） | 是（drone_anomaly.yaml 224） | 可用于主表（随机权重脚注） |
| adVAE | 无 DA 视频实现 | — | — | — | blocked |
| CVAE 系列 | 无本机实现 | — | — | — | paper-only |

## 2. 逐方法证据

### 2.1 Future Frame（P0）

**官方训练路径**：

- `Codes/train.py` 同时定义 `generator` 和 `discriminator`；`if adversarial:` 分支训练 D，`g_vars`/`d_vars` 分开优化。
- `hyper_params.ini` 的 ped1/ped2/avenue/shanghaitech 四组配置都是 `LAM_ADV = 0.05`，所以官方默认训练图是 **G+D**。
- D 由 `pix2pix.pix2pix_discriminator` 构造；按官方结构计数得到 **6,831,489** 个可训练参数（`conv0..conv4`，共 10 个变量）。G 是 **7,704,195**。
- 因此：Inference = **7.704M**（G only）；Training Trainable = **14.536M**（G+D）；Training-only = **6.831M**（D）。
- **限制**：repo 的 `hyper_params.ini` 没有 Drone-Anomaly 段，DA 训练实际是否保留 `LAM_ADV=0.05` 未确认。本次主表按“官方默认 G+D”写 14.536M，并在脚注注明 DA 配置未确认；如果 DA 训练设 `LAM_ADV=0`，D 不参与。

**官方评分路径**：

- `Codes/inference.py` 先 `test_outputs = generator(...)`，再 `test_psnr_error = psnr_error(test_outputs, test_gt)`；`utils.psnr_error` 对生成的下一帧和 GT 下一帧计算 PSNR。
- 当前 `benchmark_v2.json` / `benchmark_v2_tf2compat.json` 只计时到 generator 输出；没有 PSNR，也没有 GT 输入。因此 latency/FPS 必须读作 **Prediction-only**，不能称为完整 per-anomaly-score 耗时。
- `flops_g_per_output_frame=74.476G` 也是 G 的图；PSNR 的元素级运算未计入。PSNR 很小（O(HW)），但口径不同，不能写成“完整评分 FLOPs”。

**判定**：`Future Frame` 的推理主干是官方 G；训练参数统计已修正为官方默认 G+D；评分路径是 **prediction-only**。官方 TF1.15 记录仍是 `partial/unverified`；TF2.20 runtime-adapted 只有 prediction-only 计时，不是完整 `paper-matched` 评分口径。

### 2.2 MLEP（P0）

- `inference.py` 中 `outputs, features, _ = prednet(...)`，随后 `psnr_tensor = psnr_error(outputs, frame_gts)`；官方评分是预测帧与 GT 下一帧的 PSNR（`utils/util.py:psnr_error`）。
- 当前 benchmark 只到 prediction net 输出，和 Future Frame 一样是 **Prediction-only**；没有 GT、没有 PSNR。
- `train_scripts/train_normal_annotation.py` 只优化 generator（`g_vars` 仅 generator），没有额外训练分支；Training Trainable = 12.561M，Training-only = 0 与代码一致。
- 判定：推理结构官方一致；评分口径 **prediction-only**；TF2.20 runtime-adapted 只有 prediction-only 计时，完整评分需补测。

### 2.3 ANDT（P0）

- `ANDT/model.py` 的 `Decoder.forward` 实际执行：
  - `output = x.view(x.size(0), 256, 16, 16)`；
  - `self.de_dense(x)` 被注释掉；
  - `forward` 里还有 `pdb.set_trace()`。
- 对 6 帧输入，Transformer 输出 `feat` 的元素数远大于 65,536，`view(...,256,16,16)` 不能成立；`de_dense` 是唯一能把 `feat[:,0]`（768）映射到 65,536 的现成模块，但官方 active 代码没有使用它。
- 当前 `comparison/da/ANDT/default/scripts/andt_adapted_model.py` 使用 `de_dense(feat[:,0])` → decoder；这是 **最小适配**，不是作者 active decoder 的证明。
- 官方 `Train_and_eval.py` 的评分是 `loss_func_mse(batch_data[:,num_frames], batch_pred)`（target 下一帧），当前 benchmark 用 `F.mse_loss(out, zeros)`，评分目标也不对。
- `config.py` 的 `--num-frames` 默认是 **1**；论文/用户表使用 6 帧，但 repo 没有 DA 的六帧配置文件，六帧是通过命令行假设的。
- 判定：**适配结构 + 评分代理**，不是官方 ANDT 效率；保持 `adapted`，不能与官方 6 帧 AUC 直接联合。当前 70.347M / 74.490G / 5.015ms 只能作 adapted 参考。

### 2.4 ASTT（P0，重点）

repo 里有三个候选模型，参数量差异巨大（本机按 DA 配置 `image_size=384, patch=16, emb=768, mlp=1536, heads=6, layers=2, frames=4` 构建）：

| 文件 | 参数量 | 用途证据 |
|---|---:|---|
| `model_original.py` | **63.068M** | `train_val_Drone-Anomaly_original.py` 直接 import；最接近“原始 ASTT” |
| `model_baseline_ca.py` | **65.409M** | cross-attention baseline；DA 脚本 save path 出现 `self_cross` |
| `model_ste_tte.py` | **148.220M** | `train_val_UIT_ADrone_ste_tte.py` 使用；timm `vit_base_patch32_384` + temporal transformer |

关键问题：

- `train_val_Drone-Anomaly.py:5` 写的是 `from model import VisionTransformer`，但 repo 中 **没有 `model.py`**；只有 `model_original.py`、`model_baseline_ca.py`、`model_ste_tte.py`。
- `train_val_Drone-Anomaly.py` 还把 `model_original` 的 import 注释掉了；它的 save path 是 `experiments_spatial_temporal_self_cross_Crossroads_finetuned_from_ADrone/`，更像 cross-attention 变体。
- 当前 benchmark 使用 `model_ste_tte` 148.220M；这对应 `train_val_UIT_ADrone_ste_tte.py`，**不是 DA 脚本能直接证明的模型**。
- 输入 384×384×4 与 DA 脚本的 `image_size=384, num_frames=4` 一致；但 `model_ste_tte` 的 `timm vit_base_patch32_384` 是 patch 32，`model_original` 是 patch 16，结构不同。
- 当前 `benchmark_v2.json` 的 `adapted` 还是 `false`，这是不准确的。

**判定**：**未确认，且很可能测错变体**。当前 ASTT 行（148.220M / 109.416G / 19.048ms）不能作为 DA ASTT 的正式效率；如果 AUC 对应原始 ASTT，应重测 `model_original`（63.068M）或确认 DA `model.py` 的实际内容。审计建议：**暂不可用/需重测**，至少先把 `adapted` 改为 `true` 并标 `variant_unconfirmed`。

### 2.5 ST Dissoc.（P0）

官方 eval 路径（`arch/solver/Solver_PredRes.py:185-220`）：

1. `self.model(batch_in, None, ['E'])` 返回 `static_decoder, pred_decoder, loss_cluster_map`；
2. `pred_recon = pred_decoder + static_decoder`；
3. `pred_target = static_in + motion_target`；
4. `loss_pixelwise_cl_re = loss_map(mean((pred_recon-pred_target)^2, dim=1), loss_cluster_map)`；
5. 再用 `reciprocal_metric` 得到分数。

当前 benchmark 只执行 `self.m(x, None, ['E'])`，返回三元组；**没有步骤 2–5**。因此：

- 参数/FLOPs/延迟覆盖 stage E 的 static/motion encoder、decoder、cluster map；
- 但最终异常分数（逐像素重建误差 × cluster map、mean、reciprocal）未包含；
- 不能把它称为“每个异常分数的耗时”，应标 **cluster-map-only / 评分未完成**。

判定：官方结构基本一致；评分不完整；**需脚注 + 最终评分重测**。当前 6.120M / 60.612G / 3.668ms 只能作为 stage-E forward 参考。

### 2.6 DAD-FSM（P0）

- 官方测试 `train_ourmodel_danomaly.py:85-86`：`batch_pred = model(batch_data[:, :4], batch_seg[:, :4])`，`loss = loss_func_mse(seg_data_256[:,4], batch_pred)`；评分目标是 **未来分割图**。
- 当前 `bench_v2_extra.py::_dadfsm` 返回 `F.mse_loss(out, zeros)`，不是官方 target；评分口径不对。
- 分割图来自 `../Drone_Anomaly_seg/...` 目录；DAD-FSM repo 没有生成分割图的网络/脚本，`data_utils.py` 里只有可选的 OpenCV MOG 背景减除（`background_sub=True`），且 DADFSM loader 没有使用它。当前 benchmark 直接喂随机分割图，**只测检测网络**。
- 判定：官方 detector 结构基本一致；**precomputed segmentation input**，不是完整在线推理；评分是代理。需脚注；若要完整在线成本，需另测分割生成（repo 无来源，暂不可用）。

### 2.7 MADViT（P0）

- 论文 Sec. 5.4 的 ablation 把 `Full`、`Multi-layer`、`Last-layer` 描述为 **训练期的蒸馏目标**：Full 是全层 feature+token，Multi-layer 是“distilling at all transformer depths only”，Last-layer 是“supervising only the final [CLS, distill] tokens”。
- 论文同时明确 “Training is performed with a single frame processed”，没有说三种变体改变 teacher/student 结构、输入或推理分支。
- repo 代码只有一条 `multi_layer_distillation_loss` 前向路径；没有 Full/Multi/Last 开关或三套权重。
- 当前 Multi/Last 是 `bench_v2_madvit_variants.py` 的 **scoring-only reconstruction**：同一 teacher/student 前向，只替换 loss/scoring。
- 因此“FLOPs 相同”是**我们重建图的属性**，不是论文独立证明的属性；论文没有给出三套独立 checkpoint/架构。Multi/Last 必须标 **reconstructed**，不能写成官方变体实测；Full 可以标 code default。

判定：Full 官方一致；Multi/Last `reconstructed`；三者共享前向图的 FLOPs 只能作为 reconstructed 参考，需要脚注。

### 2.8 HSTforU（P1）

- repo 的 `models/__init__.py` 默认是 **Ped2 / PVT2-b1**；`HSTforU_Avenue.py` 使用 **PVT2-b2**；`HSTforU_ShanghaiTech.py` 是另一套 decoder/encoder。
- repo 只有 `configs/scripts/{ped2,avenue,shanghaitech}.yaml`，**没有 Drone-Anomaly 的 yaml**；README 虽列出 Drone-Anomaly，但没有给出 DA 的 model 选择。
- 当前 benchmark 直接 import `HSTforU_Avenue.AnomalyDetection`，`is_trained=False`；参数 148.970M 是 Avenue/PVT2-b2 结构。
- `bench_v2_extra.py` 里的 `mmcv.runner.load_checkpoint = lambda ...: None` 是 no-op；当前 `is_trained=False` 所以没有调用，但必须说明“结构-only、无权重”。
- 官方 `test.py` 先 `output = model(frames)`，再对 `target` 计算 PSNR，并用 `anomaly_score` 做逐视频 min-max；当前 `_hstforu` 用 `F.mse_loss(out, zeros)`，既没有 target PSNR，也没有逐视频归一化。

判定：**结构未确认（Avenue/PVT2-b2 是假设）**；评分代理；当前 148.970M / 168.209G / 64.258ms 只能作 Avenue 结构参考，不能自动称为 DA 论文配置。需脚注；DA config 需重测。

### 2.9 MKD（P1）

- README 明确 repo 面向 MNIST/Fashion-MNIST/CIFAR-10/MVTecAD/医学数据；`configs/config.yaml` 当前 `dataset_name: cifar10`、`mvtec_img_size: 128`；没有 DA 配置。
- 当前 benchmark 输入 `128x128x1`，是 MVTec 配置；AUC 来自 DA 用户表/论文。两者不是同一数据集配置。
- 评分本身与官方 `test_functions.py` 一致：cloner/source taps `(6,9,12)`，`1 - cosine + lamda * MSE`；source VGG16 在 benchmark 是 random init，官方默认 pretrained，但参数/FLOPs 与权重无关。

判定：**measured 但 paper-matched=no**；保留参数/FLOPs，但表中必须写 “MVTec 128 config”，不能直接归因于 DA 网络结构。若 DA 使用其他分辨率，需重测。

### 2.10 GANomaly（P1）

- 官方 `lib/model.py:test()` 的异常分数是 `mean((latent_i - latent_o)^2, dim=1)`；当前 `_ganomaly` 完全一致。
- 当前 `NetG(isize=256, nc=3)`；官方 `options.py` 默认 `isize=32`，repo 没有 DA 配置。G 的参数与 isize 基本无关，但 FLOPs/延迟随 256 变化。
- 训练 G+D、推理 G；当前 training-only=D 44.739M，正确。
- 判定：结构和评分完整；输入 256 未在 repo 证实；**需脚注**（config assumption）。

### 2.11 MemAE（P1）

- 官方 `script_testing.py`：`recon_res = model(frames)`，`r = recon_frames - frames`，`sp_error_map = sqrt(sum(r^2, dim=1))`，`mean(sp_error_map)` 得到 clip 分数；当前 `fn` 完全一致。
- 官方 `FrameNum=16`，`K=1, stride=1`；当前 T=16、K=1，正确。
- 官方默认 `ImgChnNum=1`（UCSD_P2_256）；当前 `chnum_in=3`。DA 是彩色，3 通道更合理，但 repo 没有 DA config，不能证明。
- 判定：结构/评分/T/K 完整；通道配置需脚注；**measured**。

### 2.12 LightViT-AD（P1）

- `configs/drone_anomaly.yaml`：`image_size=224`, `patch_size=16`；`models.py` 的 `ViTTeacher`（DeiT-tiny）+ `ViTStudent`（6 blocks）与 benchmark 一致；`anomaly_score` 是 CLS/DIST token MSE，benchmark 调用了它。
- benchmark 用 `pretrained=False`，所以 teacher 权重是随机；params/FLOPs/延迟与权重无关，AUC 需要训练权重。
- 判定：**官方结构/输入/评分一致**；`可用于主表`，加“随机权重/architecture-init”脚注。

### 2.13 Ours V0–V5（P1）

- `bench_v2_ours.py` 的 forward 是 teacher 12 blocks → taps → OCBE/student → `anomaly_readouts(..., k=5)['topk']`；评分读入完整。
- `weights_state=architecture-init`，AUC 来自 `experiments/da/appearance/...` 的 DA-trained 权重；效率/延迟与权重无关，但必须在表中注明“architecture-init，不是 AUC 权重”。
- 判定：内部结构一致；`可用于主表`（权重脚注）。

### 2.14 adVAE / CVAE 系列

- adVAE：官方 repo 是表格/向量版，不是 DA 视频模型；`blocked`。
- CVAE/DwCVAE/DCVAE/SCVAE/CLSTMVAE：只有用户表数字，`paper-only`，不参与本机速度排序。

## 3. 指标可用性总表

| 方法 | Params | FLOPs | Latency/FPS | 与 AUC 联合 | 结论 |
|---|---|---|---|---|---|
| Ours V0–V5 | 可用 | 可用（verified） | 可用 | 结构匹配；权重 architecture-init | **可用于主表**（权重脚注） |
| GANomaly | 可用（G/G+D 分开） | 可用（verified） | 可用 | 输入 256 未证实 | **需脚注** |
| MemAE | 可用 | 可用（verified，K=1） | 可用 | 3 通道未证实 | **需脚注** |
| MKD | 可用（MVTec 配置） | 可用（verified，MVTec） | 可用（MVTec） | DA 配置不匹配 | **需脚注**（MVTec 128） |
| ANDT | 仅 adapted | 仅 adapted | 仅 adapted | decoder/评分不匹配 | **需脚注 + 需重测** |
| Future Frame | 已修正 G+D | Prediction-only | Prediction-only | 推理 G 匹配；评分缺 PSNR | **需脚注**；score-inclusive 需重测 |
| ST Dissoc. | 可用 | stage-E only | stage-E only | 缺最终评分 | **需脚注 + 需重测** |
| MLEP | 可用 | Prediction-only | Prediction-only | 评分缺 PSNR | **需脚注**；score-inclusive 需重测 |
| ASTT | 当前是 STE-TTE 148M | 当前 STE-TTE | 当前 STE-TTE | 变体未确认，可能测错 | **暂不可用 / 需重测** |
| DAD-FSM | detector 可用 | detector 可用 | detector 可用 | 分割图离线预计算；评分代理 | **需脚注**（precomputed seg） |
| HSTforU | Avenue 结构 | Avenue 结构 | Avenue 结构 | DA config 未确认 | **需脚注 + 需重测** |
| MADViT Full | 可用 | 可用（verified） | 可用 | 官方 Full | **可用于主表** |
| MADViT Multi/Last | 共享前向 | 共享前向 | 共享前向 | reconstructed scoring | **需脚注**（reconstructed） |
| LightViT-AD | 可用 | 可用（verified） | 可用 | 官方 config 224 | **可用于主表**（随机权重脚注） |
| adVAE | — | — | — | — | blocked |
| CVAE 系列 | — | — | — | — | paper-only |

图例：`需脚注` = 数字可展示但必须解释配置/评分；`需重测` = 当前图或评分与目标配置不一致；`暂不可用` = 变体未确认，不能当该方法的正式效率。

## 4. 跨方法口径问题

### 4.1 评分完整性

当前 benchmark 里有三类评分：

1. **官方评分**：GANomaly latent MSE、MemAE reconstruction error、MKD cosine+MSE、LightViT token MSE、Ours top-k、MADViT Full code default；
2. **prediction-only**：Future Frame、MLEP（缺 PSNR）；ST Dissoc.（缺最终 loss_map/reciprocal）；
3. **代理评分**：ANDT、ASTT、DAD-FSM、HSTforU 都用 `F.mse_loss(out, zeros)` 或类似零目标来产生标量，而官方分别使用 target frame / target segmentation / PSNR。

代理评分的**计算量**通常与真实 target 同阶（MSE/PSNR 都是 O(output)），所以对 latency 的影响可能很小；但**语义不对**，不能把当前数字写成“官方异常分数耗时”。审计建议：要么把 target 接入 benchmark，要么在表中明确 `prediction-only / proxy scoring`。

### 4.2 输入规格

- Future Frame 256×256×4（12 通道拼接）、MLEP 224×224×4、ANDT 256×256×6、ASTT 384×384×4、DAD-FSM 224×224×4、HSTforU 256×256×4、MKD 128×128×1、GANomaly 256×256×1、MemAE 256×256×16、LightViT 224×224×1。
- 这些输入不是统一规格；FLOPs/latency 的差异不能全部归因于网络结构。尤其 MKD 128、ASTT 384、MemAE T=16、Future Frame 4 帧 12 通道，必须保留输入列。
- GANomaly 的 256、MemAE 的 3 通道、MKD 的 128 都来自我们对 DA/原仓库配置的选择，repo 没有 DA 配置的明确证据；需脚注。

### 4.3 运行框架

- PyTorch 方法：PyTorch 2.8 / CUDA 12.9，5880 Ada，FP32；与官方原版 PyTorch 1.x 的 kernel 可能不同。
- Future Frame/MLEP：官方 TF1.15/CUDA10 在 sm_89 上不可执行；当前数字来自 TF2.20/CUDA12.5 runtime-adapted（†），不是官方 TF1.15。
- 不同框架的 latency 不能直接与论文 GPU 数字比较；论文表应把“本机 5880 + 现代 runtime”作为统一测量环境，并注明框架。

### 4.4 TF1 `float_operation` 与 MAC-based FLOPs 不同

最小验证（TF1.15，CPU graph）：

- `Conv2D` 手算 MAC = `8×8×3×3×3×4 = 6,912`，`2×MAC = 13,824`；
- TF profiler 报 `Conv2D 13,824` **加** `BiasAdd 256`，总计 `14,080`；
- v2 PyTorch 手写 `_conv_out_flops` 只计 `2×MAC`，不含 bias/elementwise。

因此 Future Frame 的 74.476G、MLEP 的 319.513G 是 **TF profiler 口径**，不是统一的 MAC-only FLOPs；不能与 PyTorch 方法的 `flops_g_per_output_frame` 当成完全相同口径。当前 JSON 已标 `flops_verified=false`，审计建议在论文表中给 TF 方法的 FLOPs 加 `TF-profiler` 脚注，或另做 MAC-only 计数后再比较。

### 4.5 权重/预处理

- 所有外部方法基本都是 random init（无官方 checkpoint 或未带权重）；params/FLOPs/延迟与权重数值无关，但 AUC 不能由这些效率数字直接复现。
- GANomaly/MemAE 的输入归一化、MKD 的 VGG 预处理、DAD-FSM 的分割图来源、Future Frame 的 PSNR GT、MLEP 的 GT 下一帧，都没有统一；论文表必须把这些放在 “Input / preprocessing” 脚注。
- Future Frame 的 D 训练参数已修正；D 是否在 DA 配置启用仍未确认。

## 5. 需要修正的代码清单

按优先级排序；本次**未改 benchmark 代码**，避免旧 JSON 与新代码混用。

| 优先级 | 文件/函数 | 需要改什么 | 原因 | 是否需重跑 |
|---|---|---|---|---|
| P0 | `comparison/benchmarks/bench_v2_extra.py::_astt` | 确认 DA 对应模型（`model.py` 缺失；候选 `model_original` 63.068M / `model_baseline_ca` 65.409M / `model_ste_tte` 148.220M）；改用正确模型；评分改成 target MSE | 当前是 STE-TTE，可能不是 DA ASTT | 是 |
| P0 | `bench_v2_extra.py::_st_dissoc` | 取 `static_decoder,pred_decoder,loss_cluster_map`，加 `pred_recon`/`pred_target`/`loss_map`/reciprocal；或明确字段 `cluster_map_only=True` | 当前缺最终评分 | 是（至少 score-inclusive 版） |
| P0 | `bench_v2_extra.py::_dadfsm` | 传入真实 `seg_target`，用官方 MSE；记录 `segmentation_source=precomputed`；如需在线，补分割生成 | 当前 `MSE-to-zero` 不是官方评分 | 是（评分版） |
| P0 | `bench_v2_extra.py::_hstforu` | 传入 target frame，算 PSNR + 逐视频 min-max；确认 DA 使用哪个 `HSTforU_*`；保留 no-op checkpoint 说明 | 当前 Avenue 假设 + 代理评分 | 是 |
| P0 | `bench_v2_andt.py` | 用 target next-frame MSE；保留 `de_dense(feat[:,0])` 但标 adapted；记录 `num_frames` 与 config 默认 1 的差异 | 当前零目标；decoder 未证实 | 是 |
| P0 | `bench_tf1_extra.py` / `bench_tf2_compat.py` | 增加 PSNR（target GT）版本，或把字段固定为 `timing_scope=prediction_tensor_only`；Future Frame D 参数已补 | 当前只有 prediction tensor | 是（score-inclusive） |
| P1 | `bench_v2_extra.py::_madvit` / `bench_v2_madvit_variants.py` | 给 Multi/Last 加 `reconstructed=true`、`paper_variant_verified=false`；不要宣称三者 FLOPs 必然相同 | 论文变体没有独立权重/开关 | 否（文档/元数据） |
| P1 | `bench_v2_external.py::_ganomaly` | 记录 `isize=256` 是 DA 假设，repo 默认 32；或补 DA 配置来源 | 输入配置未证实 | 否（元数据） |
| P1 | `bench_v2_external.py::_memae` | 记录 `chnum_in=3` 是 DA 彩色假设，repo 默认 1 | 通道配置未证实 | 否（元数据） |
| P1 | `bench_v2_external.py::_mkd` | 记录 `dataset_config=MVTec-128`、`paper_matched=false` | 当前 AUC 来自 DA | 否（脚注） |
| P1 | `bench_v2_extra.py::_hstforu` | 记录 `variant=Avenue/PVT2-b2`、`da_config_unconfirmed=true`、`checkpoint=no-op` | repo 无 DA config | 否（元数据） |
| P1 | `comparison/da/Future Frame/default/*` | D 参数已补；在表格/README 标 `LAM_ADV` DA 未确认、latency prediction-only | P0 统计口径 | 已完成参数/文档 |
| P1 | `comparison/da/*/record.json` | 给需要脚注的方法加 `paper_matched`/`config_status` 说明（schema 不允许新 efficiency 键，写 notes） | 让读者区分 measured/paper-matched | 否（文档） |

## 6. 建议论文表格与表注

### 6.1 建议表结构

建议主表列：`Method | Input (H×W×T) | Inference Params (M) | FLOPs (G) | Latency@BS1 P50 (ms) | FPS@BS1 | FPS@BS16 | AUC | Notes`。  
训练参数单列附表：`Inference / Training Trainable / Training-only`；不要把训练参数混进推理效率表。

当前可放入主表的行（带标记）：

| Method | Input | Params (M) | FLOPs (G) | P50 (ms) | FPS@BS1 | FPS@BS16 | 标记 |
|---|---|---:|---:|---:|---:|---:|---|
| GANomaly | 256×256×1 | 143.950 | 16.428 | 8.088 | 123.6 | 848.5 | ¶ config assumption |
| MemAE | 256×256×16 | 6.503 | 70.808 | 7.321 | 136.5 | 130.3 | ¶ channel assumption |
| MKD | 128×128×1 | 15.059 | 10.477 | 2.495 | 394.5 | 2340.8 | ¶ MVTec-128 |
| ANDT | 256×256×6 | 70.347 | 74.490 | 5.015 | 199.4 | 197.5 | * adapted |
| Future Frame | 256×256×4 | 7.704 | 74.476 | 5.467 | 180.8 | 185.1 | †‡ prediction-only |
| ST Dissoc. | 256×256×5 | 6.120 | 60.612 | 3.668 | 269.7 | 288.2 | ‡ cluster-map-only |
| MLEP | 224×224×4 | 12.561 | 319.513 | 11.538 | 82.0 | 75.5 | †‡ prediction-only |
| ASTT | 384×384×4 | 148.220 | 109.416 | 19.048 | 52.2 | 146.6 | ? variant unconfirmed → 暂不放入 |
| DAD-FSM | 224×224×4 | 123.979 | 71.194 | 12.684 | 78.5 | 195.6 | § precomputed seg |
| HSTforU | 256×256×4 | 148.970 | 168.209 | 64.258 | 15.5 | 71.0 | ¶ variant/config |
| MADViT Full | 224×224×1 | 94.354 | 38.533 | 11.525 | 83.9 | 444.0 | — |
| MADViT Multi | 224×224×1 | 94.354 | 38.533 | 11.130 | 88.9 | 432.2 | * reconstructed |
| MADViT Last | 224×224×1 | 94.354 | 38.533 | 9.467 | 103.5 | 430.2 | * reconstructed |
| LightViT-AD | 224×224×1 | 8.840 | 3.753 | 6.519 | 148.8 | 2438.6 | — |

Ours V0–V5 按内部 variant 表另列；表中注明 `architecture-init`，AUC 来自 DA-trained 权重，二者权重不同但效率与权重无关。

### 6.2 建议表注（可直接用于论文，英文）

> All efficiency measurements were performed on a single NVIDIA RTX 5880 Ada GPU in FP32 with 50 warm-up iterations and 3×500 timed iterations; latency is the BS1 P50 per output score unless stated otherwise. FLOPs are counted for the inference graph only and exclude data loading, preprocessing, and post-processing. Input resolutions and temporal windows follow each method’s original configuration and are not identical across rows; FLOPs and latency should therefore not be compared across different input specifications without this caveat. PyTorch methods use PyTorch 2.8/CUDA 12.9; Future Frame and MLEP (†) use a TF2.20 runtime-adapted implementation because the official TF1.15/CUDA10 wheels cannot execute on sm_89, and their FLOPs are TF-profiler counts rather than the MAC-only PyTorch counter. ‡ marks prediction-only or cluster-map-only timing that does not include the final PSNR/MSE/cluster scoring. § marks methods that consume precomputed segmentation maps; the cost of segmentation generation is not included. ¶ marks measurements of the original-repository configuration (e.g., MVTec-128 for MKD, Avenue/PVT2-b2 for HSTforU) rather than the DA-adapted configuration used for the reported AUC. * marks adapted or reconstructed models (ANDT decoder; MADViT Multi-layer/Last-layer scoring). `measured` indicates that the v2 speed protocol is complete; `paper-matched` indicates that the measured graph, input, preprocessing, and scoring match the model that produced the reported AUC. A row can be `measured` without being `paper-matched`.

中文补充：如果期刊要求中文表注，可把最后一句写成“测速完成不等于与论文 AUC 配置一致；未标 paper-matched 的行只能作效率参考，不能用于声称复现了论文模型。”

## 7. 审计结论与下一步

1. **先不要重跑全部方法**。本轮只修正了 Future Frame 的训练参数（G+D），其余都是口径/文档问题；旧 `benchmark_v2.json` 未覆盖。
2. **优先处理 P0**：ASTT 变体（最严重，148M vs 63/65M）、ST 最终评分、DAD 分割目标、HSTforU target/PSNR、ANDT target/decoder、Future/MLEP PSNR。
3. **P1 只改元数据/脚注**：MKD MVTec、HSTforU Avenue、GANomaly 256、MemAE 3 通道、MADViT reconstructed；这些不改数字，只改解释。
4. **主表准入**：把 `measured` 和 `paper-matched` 分成两列/两个脚注；`ASTT` 当前行暂不可用；`Future Frame/MLEP/ST` 加 `‡ prediction-only/cluster-map-only`；`DAD-FSM` 加 `§ precomputed seg`；`MKD/HSTforU` 加 `¶ original-repo config`；`ANDT/MADViT Multi/Last` 加 `* adapted/reconstructed`。
5. **TF FLOPs**：不要把 Future Frame/MLEP 的 TF profiler FLOPs 与 PyTorch MAC-only FLOPs 当成同一口径；若要统一，另做 TF 图的 Conv/MatMul MAC-only 计数并重新验证。
