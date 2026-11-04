# adVAE (blocked for DA lightweight metrics)

- 论文：Wang et al., *adVAE: A self-adversarial variational autoencoder with Gaussian anomaly prior knowledge for anomaly detection*, Knowledge-Based Systems 2020（KBS 190:105187）。
- 本地代码：`/data1/data8tb/liangyuxuan/Comparison method/adVAE/`（HEAD `440d8af22b88`；`record.json.source.local_path` 指向这里）。
- 对比条目目录 `comparison/da/adVAE/default/` 只放对比记录，不复制上游代码；所以直接看这个目录只有 `record.json`，不是代码丢了。

## 为什么主表里标 blocked

这份**官方仓库是表格/向量数据版本**，不是 DA/ANDT 用的视频模型：

- `models/self_adVAE.py` 的 `Encoder` / `Decoder` / `Gauss_trans` 全是 `nn.Linear` 全连接层（`sample_dim=400, rep_dim=64`），`grep Conv2d/ConvTranspose` 无结果；
- `load_data/tabular.py` 读 ODDS 表格数据，`datasets/` 是 `letter.mat`、`cardio.mat`、`optdigits.mat`、`pendigits.mat`、`satellite.mat`；
- `checkpoints/` 也是 `cardio/letter/optdigits/pendigits/satellite` 的表格 checkpoint；
- README 明确写：`The whole code is coming soon...`，`Until now, we have only uploaded some code to show the performance of the adVAE model.`

ANDT 论文的 adVAE baseline（Table VI/VII 的 AUC）引用的是同一篇 KBS 论文，但 ANDT 作者实际用的**视频/图像适配版**不在这个仓库里；因此不能拿这份 FC 表格模型的参数/FLOPs 去填 DA 主表。

## 还需要什么

要解 blocked，需要满足其一：

1. ANDT 作者实际使用的 adVAE 视频实现/配置（输入尺寸、特征提取器、训练/打分方式）；
2. 论文中图像/视频版 adVAE 的网络结构说明（可据此重新实现并核对参数量）。

在此之前，`comparison_summary.md` 的 adVAE 行保持 `blocked`，不填猜测值。
