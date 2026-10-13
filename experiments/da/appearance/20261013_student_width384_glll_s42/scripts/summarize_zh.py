"""Render the verified width ablation results as a Chinese review report.

Run summarize_results.py first to recompute AUC from paired frame scores.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


EXP = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[5]
REF192 = REPO / "experiments/da/appearance/20261012_student_width192_glll_s42"
SCENES = ("highway", "farmland", "bike", "solar", "vehicle", "crossroads", "railway")
NAMES = {
    "highway": "Highway",
    "farmland": "Farmland",
    "bike": "Bike",
    "solar": "Solar",
    "vehicle": "Vehicle",
    "crossroads": "Crossroads",
    "railway": "Railway",
}


def main() -> None:
    current = json.loads((EXP / "results/summary.json").read_text())
    reference = json.loads((REF192 / "results/summary.json").read_text())
    rows384 = {row["scene"]: row for row in current["scenes"]}
    rows192 = {row["scene"]: row for row in reference["scenes"]}
    if set(rows384) != set(SCENES) or set(rows192) != set(SCENES):
        raise ValueError("七场景集合不一致")
    for scene in SCENES:
        a, b = rows384[scene], rows192[scene]
        if (a["split_fingerprint"] != b["split_fingerprint"] or
                a["frames"] != b["frames"] or
                a["positive_frames"] != b["positive_frames"] or
                abs(a["baseline_width768_auc_percent"] - b["baseline_none_auc_percent"]) > 1e-8):
            raise ValueError(f"数据或 768 维参照不一致: {scene}")

    auc768 = mean(rows384[s]["baseline_width768_auc_percent"] for s in SCENES)
    auc384 = mean(rows384[s]["auc_percent"] for s in SCENES)
    auc192 = mean(rows192[s]["auc_percent"] for s in SCENES)
    if abs(auc384 - current["macro"]["auc_percent"]) > 1e-8:
        raise ValueError("384 维宏平均不一致")
    if abs(auc192 - reference["macro"]["auc_percent"]) > 1e-8:
        raise ValueError("192 维宏平均不一致")
    params = {
        768: current["macro"]["baseline_trainable_parameters"],
        384: current["macro"]["trainable_parameters"],
        192: reference["macro"]["trainable_parameters"],
    }
    if params != {768: 30248000, 384: 12942656, 192: 7718336}:
        raise ValueError(f"参数量与预运行核算不一致: {params}")

    report = [
        "# Global-Local 学生维度 384：七场景实验总结",
        "",
        "## 实验设置与核验",
        "",
        "本次只把学生内部维度从 768 改为 384，12 个重建头相应使用 `384→768` 投影。冻结教师、OCBE、Global-Local 四个 block 的算子与顺序、三对一反向蒸馏对齐、loss 和逐帧 AUC 口径保持不变。七个场景分别训练，seed=42，15 epoch。",
        "",
        "调度器记录 7 次启动、7 次零返回结束；每张卡在场景结束后领取下一场景。已从逐帧分数重新计算 AUC，并逐场景核对帧键、标签、阳性帧数和数据划分指纹，与 768 维主对照一致；192 维参考实验的数据指纹也一致。",
        "",
        "## AUC 结果",
        "",
        "AUC 单位为百分数；差值单位为百分点，`Δ384−768` 和 `Δ384−192` 为正表示 384 维更高。",
        "",
        "| 场景 | 768 维 | 384 维 | 192 维 | Δ384−768 | Δ384−192 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scene in SCENES:
        a, b = rows384[scene], rows192[scene]
        x, y, z = a["baseline_width768_auc_percent"], a["auc_percent"], b["auc_percent"]
        report.append(f"| {NAMES[scene]} | {x:.4f} | {y:.4f} | {z:.4f} | {y-x:+.4f} | {y-z:+.4f} |")
    report.extend([
        f"| **七场景宏平均** | **{auc768:.4f}** | **{auc384:.4f}** | **{auc192:.4f}** | **{auc384-auc768:+.4f}** | **{auc384-auc192:+.4f}** |",
        "",
        "## 参数量与取舍",
        "",
        "下表只计可训练的 OCBE+学生参数，不含冻结教师、行为分支和 KNN 数据库。",
        "",
        "| 学生内部维度 | 学生参数 | OCBE 参数 | 可训练合计 | 相对 768 维减少 | 宏平均 AUC |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| 768 | 25,477,632 | 4,770,368 | {params[768]:,} | — | {auc768:.4f} |",
        f"| 384 | 8,172,288 | 4,770,368 | {params[384]:,} | {100*(1-params[384]/params[768]):.2f}% | {auc384:.4f} |",
        f"| 192 | 2,947,968 | 4,770,368 | {params[192]:,} | {100*(1-params[192]/params[768]):.2f}% | {auc192:.4f} |",
        "",
        f"384 维相对 768 维节省 **{params[768]-params[384]:,}** 个可训练参数（**{100*(1-params[384]/params[768]):.2f}%**），七场景宏平均 AUC 低 **{auc768-auc384:.4f}** 个百分点；相对 192 维多 **{params[384]-params[192]:,}** 个参数，宏平均 AUC 高 **{auc384-auc192:.4f}** 个百分点。",
        "",
        "## 观察、解释与限制",
        "",
        f"观察：384 维与 768 维最接近的是 Bike（{rows384['bike']['auc_delta_pp']:+.4f}）和 Farmland（{rows384['farmland']['auc_delta_pp']:+.4f}）；Solar 与 Railway 相比 192 维分别回升 {rows384['solar']['auc_percent']-rows192['solar']['auc_percent']:.4f} 和 {rows384['railway']['auc_percent']-rows192['railway']['auc_percent']:.4f} 个百分点。但 Vehicle 从 768 维的 {rows384['vehicle']['baseline_width768_auc_percent']:.4f} 降到 384 维的 {rows384['vehicle']['auc_percent']:.4f}，比 192 维的 {rows192['vehicle']['auc_percent']:.4f} 也低。",
        "",
        "推断：384 维在本次单种子结果中提供了参数量与宏平均 AUC 的中间折中，但逐场景 AUC 并不随宽度单调变化，不能据此断定 Vehicle 的下降由宽度单独导致或宣称 384 是最优点。若部署要求所有场景都不能明显下降，Vehicle 需要单独复核。",
        "",
        "限制：仅 seed=42；未估计跨种子方差，也未测量目标无人机硬件上的端到端推理延迟。AUC 是排序指标，报告中的分数均值差不能替代 AUC。",
    ])
    out = EXP / "results/summary.md"
    out.write_text("\n".join(report) + "\n")
    print(json.dumps({"auc_768": auc768, "auc_384": auc384, "auc_192": auc192,
                      "delta_384_vs_768_pp": auc384-auc768,
                      "delta_384_vs_192_pp": auc384-auc192,
                      "parameters_384": params[384]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
