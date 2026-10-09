# 边界探索

每个 DLBM 版本在 `dlbm/runs/<version>/`，早期小规模与 smoke 结果在 `dlbm/diagnostics/`；mask DLBM v001 在 `mask_dlbm/runs/`。进入每个目录先读 `RECORD.md` 和 `record.json`，再看 `REVIEW.md`、`summary.json`、逐场景结果与状态标记。

`source_snapshot_explore_1a68d0c/` 是从 `explore/distill-layer-boundary@1a68d0c` 导出的历史源码和测试，不是当前主模型。能定位到版本的原启动器已复制到该版本 `scripts/`；旧启动器可能包含迁移前输出路径，不能直接运行并覆盖现有资产。运行时精确提交未知时记录为 `null`，不能拿源码快照提交代替。
