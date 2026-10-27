# DGT及后续融合版本：Rank2 Q执行优化后的统一测速

主表统计七场景、batch=1、OCBE＋GL192学生前向；不含教师、预处理、H2D、评分。每个条件在同一模型、同一场景、同一输入下配对测量。

## batch=1：七场景p50均值

| 版本 | GPU | 原循环 ms/FPS | 缓存Q原循环 ms/FPS | 融合Q ms/FPS | 融合相对原循环 |
|---|---|---:|---:|---:|---:|
| GL192＋Dense128 (`gl192_dense128`) | GPU0 | 1.5405 / 649.1 | 1.5412 / 648.8 | 1.5413 / 648.8 | -0.05% |
| GL192＋Dense128＋共享可学习随机Q (`rank2_random`) | GPU1 | 2.1261 / 470.3 | 2.0058 / 498.6 | 1.6174 / 618.3 | +31.45% |
| DGT固定图 (`dgt`) | GPU0 | 3.2019 / 312.3 | 3.0634 / 326.4 | 2.5219 / 396.5 | +26.96% |
| DGT-NoGraph (`nograph`) | GPU1 | 2.4429 / 409.4 | 2.3292 / 429.3 | 1.9179 / 521.4 | +27.37% |
| DGT-CrossAttnPool (`crossattn`) | GPU0 | 3.6934 / 270.8 | 3.5413 / 282.4 | 2.9266 / 341.7 | +26.20% |
| DGT-Bilinear (`bilinear`) | GPU1 | 3.0033 / 333.0 | 2.8874 / 346.3 | 2.4893 / 401.7 | +20.65% |
| Slot16 (`slot16`) | GPU0 | 2.3944 / 417.6 | 2.2795 / 438.7 | 1.8839 / 530.8 | +27.10% |
| ResidualDepth（原loss） (`resdepth_gb`) | GPU1 | 2.8027 / 356.8 | 2.6874 / 372.1 | 2.2920 / 436.3 | +22.28% |
| ResidualDepth（score-aligned） (`resdepth_sa`) | GPU0 | 2.7885 / 358.6 | 2.6751 / 373.8 | 2.2774 / 439.1 | +22.44% |

## batch=1：包含异常分数的 OCBE＋学生链路

这一节对应旧审计的 core_score 口径；仍不包含教师前向、图像预处理和数据传输。

| 版本 | GPU | 原循环 ms/FPS | 缓存Q原循环 ms/FPS | 融合Q ms/FPS | 融合相对原循环 |
|---|---|---:|---:|---:|---:|
| GL192＋Dense128 (`gl192_dense128`) | GPU0 | 3.0134 / 331.9 | 3.0157 / 331.6 | 3.0151 / 331.7 | -0.06% |
| GL192＋Dense128＋共享可学习随机Q (`rank2_random`) | GPU1 | 3.6172 / 276.5 | 3.5026 / 285.5 | 3.1194 / 320.6 | +15.96% |
| DGT固定图 (`dgt`) | GPU0 | 4.8653 / 205.5 | 4.7373 / 211.1 | 4.2251 / 236.7 | +15.15% |
| DGT-NoGraph (`nograph`) | GPU1 | 3.9368 / 254.0 | 3.8035 / 262.9 | 3.3941 / 294.6 | +15.99% |
| DGT-CrossAttnPool (`crossattn`) | GPU0 | 5.7635 / 173.5 | 5.6161 / 178.1 | 5.0027 / 199.9 | +15.21% |
| DGT-Bilinear (`bilinear`) | GPU1 | 4.4703 / 223.7 | 4.3616 / 229.3 | 3.9615 / 252.4 | +12.84% |
| Slot16 (`slot16`) | GPU0 | 3.8370 / 260.6 | 3.7290 / 268.2 | 3.3338 / 300.0 | +15.09% |
| ResidualDepth（原loss） (`resdepth_gb`) | GPU1 | 4.2626 / 234.6 | 4.1526 / 240.8 | 3.7567 / 266.2 | +13.47% |
| ResidualDepth（score-aligned） (`resdepth_sa`) | GPU0 | 4.2365 / 236.0 | 4.1283 / 242.2 | 3.7367 / 267.6 | +13.37% |

## 逐场景融合结果

| 版本 | 场景 | GPU | 原循环 FPS | 缓存循环 FPS | 融合 FPS | 融合相对原循环 |
|---|---|---|---:|---:|---:|---:|
| gl192_dense128 | bike | GPU0 | 648.0 | 648.1 | 648.3 | +0.04% |
| gl192_dense128 | crossroads | GPU0 | 649.7 | 647.5 | 647.3 | -0.37% |
| gl192_dense128 | farmland | GPU0 | 645.8 | 648.3 | 648.3 | +0.38% |
| gl192_dense128 | highway | GPU0 | 649.5 | 648.3 | 648.4 | -0.17% |
| gl192_dense128 | railway | GPU0 | 651.6 | 650.7 | 650.6 | -0.15% |
| gl192_dense128 | solar | GPU0 | 648.5 | 648.6 | 648.2 | -0.04% |
| gl192_dense128 | vehicle | GPU0 | 651.0 | 650.3 | 650.5 | -0.08% |
| rank2_random | bike | GPU1 | 473.5 | 502.5 | 622.8 | +31.54% |
| rank2_random | crossroads | GPU1 | 467.7 | 496.7 | 615.9 | +31.68% |
| rank2_random | farmland | GPU1 | 468.5 | 497.2 | 616.0 | +31.49% |
| rank2_random | highway | GPU1 | 469.0 | 496.3 | 616.1 | +31.38% |
| rank2_random | railway | GPU1 | 468.0 | 495.9 | 615.3 | +31.46% |
| rank2_random | solar | GPU1 | 479.6 | 508.0 | 631.0 | +31.57% |
| rank2_random | vehicle | GPU1 | 466.3 | 493.7 | 611.1 | +31.04% |
| dgt | bike | GPU0 | 290.5 | 303.4 | 370.9 | +27.69% |
| dgt | crossroads | GPU0 | 289.8 | 303.0 | 371.0 | +28.01% |
| dgt | farmland | GPU0 | 289.8 | 302.9 | 370.2 | +27.74% |
| dgt | highway | GPU0 | 282.9 | 295.8 | 361.5 | +27.78% |
| dgt | railway | GPU0 | 289.8 | 302.8 | 369.8 | +27.61% |
| dgt | solar | GPU0 | 392.9 | 410.6 | 489.7 | +24.63% |
| dgt | vehicle | GPU0 | 393.7 | 411.6 | 488.1 | +23.98% |
| nograph | bike | GPU1 | 408.2 | 428.2 | 520.3 | +27.47% |
| nograph | crossroads | GPU1 | 411.0 | 431.7 | 523.5 | +27.37% |
| nograph | farmland | GPU1 | 410.6 | 431.2 | 523.5 | +27.52% |
| nograph | highway | GPU1 | 408.2 | 427.6 | 518.9 | +27.12% |
| nograph | railway | GPU1 | 408.3 | 427.8 | 518.8 | +27.06% |
| nograph | solar | GPU1 | 410.2 | 430.7 | 524.6 | +27.90% |
| nograph | vehicle | GPU1 | 409.1 | 428.2 | 520.3 | +27.17% |
| crossattn | bike | GPU0 | 273.8 | 285.4 | 345.5 | +26.17% |
| crossattn | crossroads | GPU0 | 265.8 | 277.4 | 336.1 | +26.45% |
| crossattn | farmland | GPU0 | 271.3 | 282.8 | 342.2 | +26.11% |
| crossattn | highway | GPU0 | 271.1 | 282.7 | 341.8 | +26.10% |
| crossattn | railway | GPU0 | 271.0 | 282.5 | 341.9 | +26.13% |
| crossattn | solar | GPU0 | 270.7 | 282.7 | 341.2 | +26.04% |
| crossattn | vehicle | GPU0 | 271.6 | 283.3 | 343.3 | +26.39% |
| bilinear | bike | GPU1 | 333.9 | 347.3 | 403.0 | +20.72% |
| bilinear | crossroads | GPU1 | 333.1 | 346.4 | 402.7 | +20.92% |
| bilinear | farmland | GPU1 | 334.1 | 347.4 | 402.3 | +20.40% |
| bilinear | highway | GPU1 | 333.0 | 345.9 | 400.8 | +20.35% |
| bilinear | railway | GPU1 | 332.0 | 345.5 | 400.9 | +20.75% |
| bilinear | solar | GPU1 | 333.4 | 346.7 | 401.9 | +20.56% |
| bilinear | vehicle | GPU1 | 331.4 | 345.2 | 400.4 | +20.83% |
| slot16 | bike | GPU0 | 417.0 | 438.1 | 530.4 | +27.21% |
| slot16 | crossroads | GPU0 | 417.1 | 438.4 | 531.3 | +27.38% |
| slot16 | farmland | GPU0 | 417.8 | 438.1 | 531.0 | +27.11% |
| slot16 | highway | GPU0 | 417.3 | 438.3 | 528.6 | +26.69% |
| slot16 | railway | GPU0 | 418.5 | 438.7 | 531.0 | +26.87% |
| slot16 | solar | GPU0 | 417.3 | 439.0 | 530.8 | +27.19% |
| slot16 | vehicle | GPU0 | 418.5 | 440.3 | 532.5 | +27.24% |
| resdepth_gb | bike | GPU1 | 359.5 | 375.1 | 440.2 | +22.43% |
| resdepth_gb | crossroads | GPU1 | 350.1 | 364.9 | 427.7 | +22.16% |
| resdepth_gb | farmland | GPU1 | 358.5 | 373.5 | 437.7 | +22.07% |
| resdepth_gb | highway | GPU1 | 357.7 | 373.2 | 438.2 | +22.50% |
| resdepth_gb | railway | GPU1 | 357.0 | 372.5 | 435.4 | +21.97% |
| resdepth_gb | solar | GPU1 | 358.1 | 373.4 | 438.4 | +22.43% |
| resdepth_gb | vehicle | GPU1 | 356.8 | 372.3 | 436.8 | +22.43% |
| resdepth_sa | bike | GPU0 | 359.7 | 375.1 | 440.2 | +22.38% |
| resdepth_sa | crossroads | GPU0 | 357.9 | 373.4 | 438.8 | +22.61% |
| resdepth_sa | farmland | GPU0 | 358.8 | 373.4 | 439.0 | +22.37% |
| resdepth_sa | highway | GPU0 | 358.6 | 373.8 | 439.3 | +22.50% |
| resdepth_sa | railway | GPU0 | 358.4 | 373.5 | 439.0 | +22.50% |
| resdepth_sa | solar | GPU0 | 358.4 | 373.6 | 438.8 | +22.43% |
| resdepth_sa | vehicle | GPU0 | 358.6 | 373.8 | 438.7 | +22.33% |

## 解释与边界

原实现的每组三层Q投影被展开为逐元素小算子；缓存Q只消除了每帧Gram–Schmidt，仍保留乘法、加法、select和临时张量。融合路径将四组投影合并为单个CUDA kernel；DGT变体按[group,batch,patch,2×768]输出，保持原有每组Linear的输入形状。Q系数、有效方向、模型权重、loss和学生结构没有改变。

所有场景均核对原循环、缓存循环和融合路径的code、12个student重建和异常分数；最大绝对误差写入各GPU JSON。训练诊断在部署测速中显式关闭，旧版本和新版本口径一致；关闭诊断本身不作为Q融合收益。

参数、主要GFLOPs、Model Size和历史AUC沿用旧审计并在config中重新核对；本审计只重新测Latency/FPS。旧完整checkpoint缺失的DGT主路径与DenseRank2记录不补造AUC。不同GPU的绝对值不混合比较，正式主表保留GPU列。

启用：`set_diagnostics_enabled(False)`（DGT变体）→ `eval()` → `cache_depth_basis_for_inference(fuse_projection=True)` → `torch.inference_mode()`。训练及梯度开启的eval仍走原实现。
