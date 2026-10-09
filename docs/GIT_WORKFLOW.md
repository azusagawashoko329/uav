# Git 提交、分支与云端同步

## 提交说明

新提交标题采用 `<type>(<scope>): <动作与对象>`，type 取 `docs|experiment|model|data|fix|chore`。`scope` 用实验 ID 或模块名，标题不写未经核验的结论。实验提交正文填：

```
Experiment: <id 或 none>
Dataset: da|uit|da_uit|none
Source-Commit: <40 位运行源码提交，未知填 unknown>
Evidence: <相对路径，多个用逗号>
Status: planned|running|completed|failed|cancelled|unknown
```

源码、运行结果、评测和复盘分开提交，使每一步可溯源。修正既有指标用新提交并解释旧值，不改写已发布提交。旧历史的短哈希和标签已被笔记引用；不重写历史，也不批量改旧提交说明。原历史的不足用附加的实验记录补足。

## 分支

新开发从确认过的基线分支起 `research/<topic>`，实验之间尽量独立。当前历史存在 `research/next`、`master`、`explore/distill-layer-boundary` 多条线；不能凭时间或目录名断言它们谁是正式模型。切换分支前先提交/保存工作树，比较 `git merge-base` 与差异；模型分支合并须独立复核。提交里保留实验源码指针，不用可移动的 `main` 代替 commit。

## 本地与 GitHub

远端目标是 `https://github.com/azusagawashoko329/uav.git`。首次推送前核对 `git remote -v`、仓库权限和远端实际 HEAD；仓库为空时可用当前选定基线初始化 `main`，但不得默认覆盖其他人的远端工作。每次实验阶段提交后 `git push -u origin <branch>`，再 `git ls-remote --heads origin <branch>`，确认远端 40 位哈希等于 `git rev-parse HEAD`。若认证或网络失败，保留本地提交，记录原因和待推分支/哈希；不能写“已同步”。公开云端只保存代码、规范、可阅读的实验日志/指标/报告；`.pth/.pt`、视频和大缓存不上传。大文件若需共享，应使用明确的外部存储，并在 `artifacts.json` 登记 URI、SHA-256 和保留策略。

项目 `.gitignore` 是共享规则；不要只在 `.git/info/exclude` 中隐藏应上云的实验元数据。检查 `git status --ignored` 和 `git check-ignore -v <path>`，必要时有选择地 `git add -f` 小型证据，不能强制加入权重和视频。
