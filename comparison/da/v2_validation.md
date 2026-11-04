# v2 FLOPs 计数器验证记录（2026-11-04）

协议：[BENCHMARK_PROTOCOL.md](../BENCHMARK_PROTOCOL.md) v2。本轮先用 GANomaly（反卷积）、MemAE（3D 卷积/反卷积）、GLoRD V0（Attention）做小规模验证，再跑正式 50 warmup + 3×500。逐方法的 **公平性/paper-matched 复核**见 [fairness_audit.md](fairness_audit.md)：`measured` 只表示测速完成，不等于与论文 AUC 配置一致。

## 1. 计数器实现

- 主计数器：`torch.utils.flop_counter.FlopCounterMode`（TorchDispatchMode），覆盖 `Conv2d/ConvTranspose2d/Conv3d/ConvTranspose3d`、`Linear`（addmm/mm）、`MatMul/BMM`、融合 SDPA。
- 补丁：`aten._native_multi_head_attention` 不在 PyTorch 注册表里，`comparison/benchmarks/flops_counter.py` 增加 `native_mha_flop`（QKV + QK + AV + 输出投影）。
- 独立公式：`hand_module_flops` 对 Conv/ConvTranspose/Linear/MultiheadAttention 逐模块手算；与主计数器**分开跑**，不重复累加。
- Profiler：`torch.profiler(with_flops=True)` 只比较**共同支持**的算子类别；`ConvTranspose`、`Conv3d`、融合 Attention、native MHA 等盲区跳过，不用 profiler 总数做验收。
- 去重：module hook / matmul / SDPA 包装是分开的计数路径；按 module + 调用 + shape 对齐，`flops_duplicate_check` 记录是否存在 `convolution` 与 `_convolution` 同图等风险。
- 单位：`1 MAC = 2 FLOPs`。

## 2. 玩具算子验证

| 算子 | 手算期望 FLOPs | 计数器 | 差异 |
|---|---:|---:|---:|
| `ConvTranspose2d(3→4,k3,s2,p1,op1)` | 13,824 | 13,824 | 0 |
| `Conv3d(2→3,k3,s1,p1)` | 82,944 | 82,944 | 0 |
| `ConvTranspose3d(2→3,k3,s2,p1,op1)` | 10,368 | 10,368 | 0 |
| `Linear(8→16)` | 1,024 | 1,024 | 0 |
| `MultiheadAttention(8,2)` | 2,560 | 2,560 | 0 |
| Conv2d + Linear + MHA 组合 | 16,896 | 16,896 | 0 |

组合模型：手写公式 3 个模块全部匹配（max rel diff = 0）；profiler 共同类别 `conv` 差异 0；`matmul` 因 native MHA 被 profiler 分解而跳过。

## 3. 实际模型验证

| 模型 | 输入 | 计数器 Full FLOPs | 手写模块 | 手写最大误差 | Profiler 总数 | Profiler 盲区 |
|---|---|---:|---:|---:|---:|---|
| GANomaly `NetG` | 1×3×256×256 | **16.427778048G** | 21 模块 | 0 | 10.951852032G | 漏 `ConvTranspose2d` 5.475926016G |
| MemAE | 1×3×16×256×256 | **70.808240128G** | 8 模块 | 0 | 1.050624G | 漏 `Conv3d/ConvTranspose3d` 69.757616128G |
| GLoRD V0 | 1×3×256×256 | **95.106764800G** | 92 模块 | 0 | 90.204910640G | native MHA 被部分分解；v1 少 4.901854160G |
| ANDT adapted 6-frame | 1×6×3×256×256 | **74.489796608G** | 16 模块 | 0 | 73.812554496G | ConvTranspose2d 盲区；matmul 共同子集 0% |
| LightViT-AD | 1×3×224×224 | **3.753105408G** | 74 模块 | 0 | 3.212593921G | attention 跳过；conv/matmul 0% |
| MADViT Full (code default) | 1×3×224×224 | **38.533091328G** | 112 模块 | 0 | 36.731774980G | attention 跳过；conv/matmul 0% |
| ST Dissoc. | 1×15×256×256 | **60.611887104G** | 45 模块 | 0 | 60.646413312G | conv 0% |
| ASTT (STE-TTE) | 1×4×3×384×384 | **109.415536640G** | 64 模块 | 0 | 105.638578944G | attention/ConvTranspose 盲区；matmul 0% |
| DAD-FSM | 1×4×3×224×224 ×2 流 | **71.193985024G** | 78 模块 | 0 | 71.044112386G | ConvTranspose 盲区；matmul 0% |
| HSTforU (Avenue) | 4×3×256×256 | **168.209416192G** | 216 模块 | 0 | 144.628441088G | attention/ConvTranspose 盲区；matmul 0% |
| Future Frame | 4×3×256×256 | **74.476028824G** | — | — | 74.476028824G (TF1) | TF1 profiler；官方 latency 未测；TF2 adapted 见 §7 |
| MLEP | 4×3×224×224 | **319.512526355G** | — | — | 319.512526355G (TF1) | TF1 profiler；官方 latency 未测；TF2 adapted 见 §7 |

ANDT 的适配方式是 `de_dense(feat[:,0])` 预测下一帧（发布代码 decoder 只支持 1 帧），因此这一行是 **adapted**，不是 official；详见 [ANDT/default/README.md](ANDT/default/README.md)。

MADViT Full / Multi-layer / Last-layer 共享同一 teacher+student 前向图；FLOPs 都是 38.533G，差异只在 scoring（论文 Sec. 5.4）：Full=全层 feature+token，Multi=全层 feature only，Last=final token only。

DINOv3 teacher 的融合 Attention 还做了独立公式核对：12 blocks × `4·L²·E`（L=261 tokens，E=768）= **2,511,212,544 FLOPs**，与计数器 `aten._scaled_dot_product_efficient_attention` 完全一致。

结论：

- GANomaly 的 v1 10.952G 只覆盖 Conv2d/Linear，**漏了反卷积**；v2 16.428G 与实际 G 推理图一致。
- MemAE 的 v1 0.065664G/帧 ×16 = 1.050624G 只覆盖 memory 的 matmul，**漏了全部 3D 卷积/反卷积**；v2 全窗口 70.808G。
- GLoRD V0 的 v1 90.205G 漏了 student native MHA；v2 95.107G，与 2026-10-26 独立审计的 95.1067648G 一致。
- 除 TF1 的 Future Frame/MLEP 外，上述模型的手写模块与主计数器误差均为 0；没有 `hand_only` 模块；`flops_ops_missing=[]`，`flops_verified=true`。Future Frame/MLEP 的 FLOPs 来自 TF1 profiler，标 `partial/unverified`。
- Profiler 总数只作参考，**没有**用于 Full FLOPs 验收；这避免了 MemAE 这类正确结果被不完整 profiler 判失败。

## 4. 正式测量

执行脚本：

```bash
CUDA_VISIBLE_DEVICES=GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6 \
python comparison/benchmarks/bench_v2_ours.py

CUDA_VISIBLE_DEVICES=GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6 \
python comparison/benchmarks/bench_v2_external.py --method ganomaly
# ... --method memae / mkd

CUDA_VISIBLE_DEVICES=GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6 \
python comparison/benchmarks/bench_v2_andt.py

CUDA_VISIBLE_DEVICES=GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6 \
python comparison/benchmarks/bench_v2_extra.py --method lightvit
# ... --method madvit / st / astt / dadfsm / hstforu

CUDA_VISIBLE_DEVICES=GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6 \
python comparison/benchmarks/bench_v2_madvit_variants.py

CUDA_VISIBLE_DEVICES=GPU-3ae8a423-c7fb-db6c-8c0a-f7da0c5cc8f6 \
python comparison/benchmarks/bench_tf2_compat.py --method both
```

> `bench_tf2_compat.py` 是 Future Frame/MLEP 的 TF2.20 runtime-adapted 计时，不覆盖官方 `benchmark_v2.json`；说明见 §7。

- ANDT 为 adapted 6-frame：74.490G / 5.015ms / 199.4 FPS@BS1 / 197.5 FPS@BS16；不是官方代码。
- 额外方法：LightViT-AD 3.753G/6.519ms/148.8 FPS；ST Dissoc. 60.612G/3.668ms/269.7 FPS；ASTT 109.416G/19.048ms/52.2 FPS；DAD-FSM 71.194G/12.684ms/78.5 FPS；HSTforU 168.209G/64.258ms/15.5 FPS。
- MADViT 三变体（同一前向图）：Full 38.533G/11.525ms/83.9 FPS；Multi-layer 38.533G/11.130ms/88.9 FPS；Last-layer 38.533G/9.467ms/103.5 FPS。Multi/Last 为按论文 Sec. 5.4 改 scoring 的 adapted reconstruction。
- 预热 50 次；BS1 3×500 次（每次同步，P50/Mean/P95）；BS16 3×500 个 batch（只在每轮结束同步）。
- 结果只写对应方法/变体目录的 `benchmark_v2.json`；旧 `legacy_v1/benchmark_*.json` 未改动。
- 日志里的 `wrote ...` 是文件重排前的路径；文件已按方法目录归档，JSON/日志内容本身未改。
- 汇总见 [comparison_summary.md](comparison_summary.md) §3。

## 5. 依赖/代码限制（HSTforU 已解；TF1 官方 runtime 仍 blocked，TF2 adapted 已补）

- **HSTforU**：已用 `mmcv-lite 2.x` + 兼容 shim（`mmcv.runner`、`parrots_wrapper`、`UPSAMPLE_LAYERS`）构造 Avenue/PVT2-b2 模型并完成测量；不再是 blocked。
- **Future Frame**：TF1.15 环境已建，generator 图可构建并得到 7.704M / 74.476G；官方 FlowNet2/checkpoint 下载链接失效（OneDrive 301/401、AWS 404），且 TF1 CUDA10 在 5880 上的 GPU session 无法按协议完成 latency。官方记录保持 `partial/unverified`；另用 TF2.20 runtime-adapted 得到 5.467ms / 180.8 FPS@BS1（§7），仍不能冒充 TF1.15。
- **MLEP**：TF1.15 环境已建，`cyclegan_convlstm` 图可构建并得到 12.561M / 319.513G；官方 GPU latency 同样受 TF1 CUDA10 限制。官方记录保持 `partial/unverified`；另用 TF2.20 runtime-adapted 得到 11.538ms / 82.0 FPS@BS1（§7）。
- **adVAE**：官方仓库是表格/向量版，不是 DA 视频模型，保持 `blocked`。

### 5.1 MADViT 变体核对

`MADViT/code/train_val_VIT_Distillation.py` 只有一条评分路径：`multi_layer_distillation_loss` 对全部 12 个 teacher/student skip connection 做 cosine loss，再加 CLS/DIST token KL loss。代码里没有 Full / Multi-layer / Last-layer 参数或层选择开关。

论文 PDF（本地 `Comparison method/MADViT/...pdf`）Sec. 5.4 的定义：

- `Full`：全层 feature loss + final CLS/DIST token loss（代码默认路径）；
- `Multi-layer`：只做全层 feature loss，不做 token loss；
- `Last-layer`：只监督 final CLS/DIST token（token loss only）。

因此 Multi-layer / Last-layer 是用同一 teacher+student 前向图、只替换 scoring 的 **adapted reconstruction**，在 `bench_v2_madvit_variants.py` 中实现；结果标 adapted，不冒充代码原生变体。

## 6. 复现坑（已修复，保留记录）

MKD 前两次尝试失败，未写入正式结果：

1. `make_arch('B', cfg_b, ...)` 把 `cfg` 传成了 list；上游接口要求 dict `{'A': ..., 'B': ...}`，报 `TypeError: list indices must be integers or slices, not str`。
2. 上游 `VGG.forward` 在 `target_layer=11` 调 `x.register_hook(...)`；在 `torch.no_grad()`/`inference_mode()` 下报 `RuntimeError: cannot register a hook on a tensor that doesn't require gradient`。v2 改为 hook-free 的 `_cloner_features()`，按相同 taps `[2,5,8,11,14,17,20,23,26,29,32,35,38]` 收集特征。

最终成功日志：`comparison/da/MKD/default/benchmark_v2.log`。

## 7. TF1 方法的 TF2 runtime-adapted 验证（Future Frame / MLEP）

### 7.1 为什么需要 runtime adaptation

官方 TF1.15 wheel 绑定 CUDA 10.0/cuDNN 7.6.5；RTX 5880 是 Ada/sm_89。最小复现：

| 测试 | 结果 |
|---|---|
| TF1.15 识别 5880（compute capability 8.9） | 成功 |
| Session 创建 + 变量初始化 | 约 2.3s |
| `tf.add` elementwise GPU | 正常 |
| 64×64 `tf.matmul`（cuBLAS 10.0） | 40s 内不返回 |
| 32×32 `tf.nn.conv2d`（cuDNN 7） | 100s 内不返回 |
| 完整模型 smoke | 5–15 分钟不返回 |

因此官方 TF1.15 记录只能是 params/FLOPs + `partial/unverified`；latency/FPS 不能填。

### 7.2 TF2 适配方式和验收

`comparison/benchmarks/bench_tf2_compat.py` + `tf2_compat_shims.py`：

- 模型文件不改；只补 TF1 `contrib.layers`、`arg_scope`、`contrib.rnn.ConvLSTMCell`。
- TF2 版本：2.20.0（CUDA 12.5.1 build，cuDNN 9，Python 3.13.5）。
- **验收条件：TF2 图 trainable params 与 TF1 图逐参数完全一致**：
  - Future Frame：TF1 7,704,195 = TF2 7,704,195；
  - MLEP：TF1 12,560,835 = TF2 12,560,835。
  脚本在不等时直接 `RuntimeError`。
- 输出 shape 也一致：Future Frame `[B,256,256,3]`；MLEP `[B,224,224,3]`，ConvLSTM hidden `[B,56,56,256]`。
- FLOPs 沿用 TF1.15 profiler（同一图）：Future Frame 74.476G，MLEP 319.513G；没做 v2 PyTorch 计数器交叉验证，所以 `flops_verified=false`。
- 计时：5880 UUID、FP32、BS1/BS16 各 50 warmup + 3×500；每次 `Session.run` 同步计时。TF Session.run 的同步语义与 PyTorch BS16 异步 launch 不同，报告中已注明。

### 7.3 结果（runtime-adapted，不是官方 TF1.15）

| Method | Runtime | Params (M) | FLOPs (G) | P50 (ms) | Mean (ms) | P95 (ms) | FPS@BS1 | FPS@BS16 | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Future Frame | TF2.20/CUDA12.5.1/cuDNN9 | 7.704 | 74.476 | 5.467 | 5.531 | 6.217 | 180.8 | 185.1 | `partial/unverified`（runtime-adapted） |
| MLEP | TF2.20/CUDA12.5.1/cuDNN9 | 12.561 | 319.513 | 11.538 | 12.188 | 13.822 | 82.0 | 75.5 | `partial/unverified`（runtime-adapted） |

MLEP 的 BS16 低于 BS1，来自 ConvLSTM `dynamic_rnn` 的 while-loop 和 batch 16 调度开销；是实测值，不做外推。两方法都没有连续视频流实测，`fps_stream_measured=null`。

### 7.4 状态判定

- 官方 `benchmark_v2.json` 不变：TF1.15，latency null，`partial/unverified`。
- 新增 `benchmark_v2_tf2compat.json`：`runtime_adapted=true`、`model_code_modified=false`、`adapted_reason`、`tf1_params_check`。
- 因为 `flops_verified=false`，即使有 latency 也不能升为 `measured`；状态仍是 `partial/unverified`。
- 引用规则：主表可用 P50@BS1 + `runtime-adapted †` 脚注；不得覆盖官方 TF1.15 行，也不得把 TF2 数字写成论文 TF1 数字。

## 8. score-inclusive 补测（2026-11-04）

旧 JSON 保留；以下补测写独立 `benchmark_v2_scored.json`。

| Method | 旧口径 | 旧 P50 / FPS@BS1 / FPS@BS16 | 新评分 | 新 P50 / FPS@BS1 / FPS@BS16 |
|---|---|---|---:|---:|
| Future Frame | prediction-only | 5.467ms / 180.8 / 185.1 | 官方 PSNR vs GT next frame | **6.829ms / 145.9 / 178.2** |
| ST Dissoc. | stage-E forward-only | 3.668ms / 269.7 / 288.2 | 官方 loss_map + reciprocal | **4.280ms / 231.4 / 452.3** |
| MLEP | prediction-only | 11.538ms / 82.0 / 75.5 | 官方 PSNR vs GT next frame | **12.787ms / 78.3 / —（BS16 未测）** |
| DAD-FSM | MSE-to-zero proxy | 12.684ms / 78.5 / 195.6 | 未来分割 target MSE | **13.576ms / 73.4 / 289.1** |

- MLEP 的 scored BS16 未完成（长时 run 被停止），只写 BS1；不能用 prediction-only 的 75.5 冒充 scored BS16。
- 评分 elementwise 运算不计入 MAC-based FLOPs；DAD 的分割图仍是离线预计算，不含分割生成。
- `bench_v2_scored.py`（ST/DAD）与 `bench_tf2_scored.py`（Future Frame/MLEP）不覆盖旧 JSON。
