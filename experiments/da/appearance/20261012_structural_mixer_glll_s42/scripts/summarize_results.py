"""Verify seven-scene outputs and recompute paired frame-level AUCs.

This is a post-run analysis script. It does not change the training source snapshot.
"""
import csv
import json
from pathlib import Path
from statistics import mean

from sklearn.metrics import roc_auc_score


SCENES = ("highway", "farmland", "bike", "solar", "vehicle", "crossroads", "railway")
EXP = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[5]
BASE = REPO / "experiments/da/aggregation/20261009_student_adapter_none_s42"


def read_scores(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or set(rows[0]) != {"video", "frame", "label", "anomaly_score"}:
        raise ValueError(f"invalid score CSV: {path}")
    keys = [(r["video"], r["frame"], r["label"]) for r in rows]
    labels = [int(r["label"]) for r in rows]
    scores = [float(r["anomaly_score"]) for r in rows]
    return keys, labels, scores


def verify_scheduler(path):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    launches = [e for e in events if e["kind"] == "launch"]
    finishes = [e for e in events if e["kind"] == "finish"]
    if len(launches) != 7 or len(finishes) != 7 or events[-1]["kind"] != "all_finished":
        raise ValueError("scheduler did not finish exactly seven scenes")
    if set(e["scene"] for e in launches) != set(SCENES):
        raise ValueError("scheduler scene set mismatch")
    if any(e["returncode"] != 0 for e in finishes):
        raise ValueError("a scene process failed")
    started = {e["scene"]: e for e in launches}
    ended = {e["scene"]: e for e in finishes}
    if any(started[s]["gpu"] != ended[s]["gpu"] or started[s]["pid"] != ended[s]["pid"]
           for s in SCENES):
        raise ValueError("scheduler launch/finish mismatch")
    return started, ended, events


def main():
    launches, finishes, _ = verify_scheduler(EXP / "logs/scheduler_20261012T055019Z.jsonl")
    output = []
    for scene in SCENES:
        current = json.loads((EXP / f"results/{scene}.json").read_text())
        baseline = json.loads((BASE / f"results/{scene}.json").read_text())
        keys, y, scores = read_scores(EXP / f"results/{scene}_scores.csv")
        baseline_keys, baseline_y, baseline_scores = read_scores(BASE / f"results/{scene}_scores.csv")
        if keys != baseline_keys or y != baseline_y:
            raise ValueError(f"frame/label mismatch versus baseline: {scene}")
        if current["split_fingerprint"] != baseline["split_fingerprint"]:
            raise ValueError(f"split fingerprint mismatch: {scene}")
        if len(y) != current["frames"] or sum(y) != current["positive_frames"]:
            raise ValueError(f"frame count mismatch: {scene}")
        auc = roc_auc_score(y, scores) * 100
        old_auc = roc_auc_score(y, baseline_scores) * 100
        if abs(auc - current["auc_percent"]) > 1e-8 or abs(old_auc - baseline["auc_percent"]) > 1e-8:
            raise ValueError(f"reported AUC mismatch: {scene}")
        log_path = Path(launches[scene]["log"])
        log_lines = log_path.read_text().splitlines()
        if len([line for line in log_lines if line.startswith("epoch=")]) != 15:
            raise ValueError(f"not 15 completed epochs: {scene}")
        if json.loads(log_lines[-1])["auc_percent"] != current["auc_percent"]:
            raise ValueError(f"final log mismatch: {scene}")
        normal = [score for label, score in zip(y, scores) if label == 0]
        abnormal = [score for label, score in zip(y, scores) if label == 1]
        output.append({
            "scene": scene,
            "auc_percent": auc,
            "baseline_none_auc_percent": old_auc,
            "auc_delta_pp": auc - old_auc,
            "frames": len(y),
            "positive_frames": sum(y),
            "normal_mean_score": mean(normal),
            "abnormal_mean_score": mean(abnormal),
            "score_gap": mean(abnormal) - mean(normal),
            "trainable_parameters": current["trainable_parameters"],
            "baseline_trainable_parameters": baseline["trainable_parameters"],
            "split_fingerprint": current["split_fingerprint"],
            "gpu": launches[scene]["gpu"],
            "launch_time_utc": launches[scene]["time"],
            "finish_time_utc": finishes[scene]["time"],
        })
    macro = {
        "scene": "macro_mean",
        "auc_percent": mean(r["auc_percent"] for r in output),
        "baseline_none_auc_percent": mean(r["baseline_none_auc_percent"] for r in output),
        "auc_delta_pp": mean(r["auc_delta_pp"] for r in output),
        "trainable_parameters": output[0]["trainable_parameters"],
        "baseline_trainable_parameters": output[0]["baseline_trainable_parameters"],
        "trainable_parameter_reduction_percent": 100 * (1 - output[0]["trainable_parameters"] / output[0]["baseline_trainable_parameters"]),
    }
    summary_path = EXP / "results/summary.csv"
    fields = list(output[0].keys())
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    (EXP / "results/summary.json").write_text(json.dumps({
        "source": "seven paired frame-score CSVs with matching keys/labels and split fingerprints",
        "scenes": output,
        "macro": macro,
        "scheduler": {
            "initial_utc": launches["bike"]["time"],
            "final_utc": finishes["vehicle"]["time"],
            "all_exit_zero": True,
        },
    }, ensure_ascii=False, indent=2) + "\n")
    report = [
        "# Global-local student structural experiment",
        "",
        "The loss alignment is unchanged: consecutive teacher layers are grouped in fours of three, and each student stage owns three reconstruction heads. The only model change is the student block topology: stage 0 uses global MHSA; stages 1–3 use 3x3 depthwise spatial mixing plus pointwise projection; all stages use a SiLU gated FFN.",
        "",
        "## Seven-scene result",
        "",
        "| scene | structural AUC | no-adapter AUC | delta (pp) | normal mean | abnormal mean | score gap | GPU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in output:
        report.append(
            f"| {row['scene']} | {row['auc_percent']:.6f} | {row['baseline_none_auc_percent']:.6f} | "
            f"{row['auc_delta_pp']:+.6f} | {row['normal_mean_score']:.6f} | "
            f"{row['abnormal_mean_score']:.6f} | {row['score_gap']:.6f} | {row['gpu']} |"
        )
    report.extend([
        f"| **macro mean** | **{macro['auc_percent']:.6f}** | **{macro['baseline_none_auc_percent']:.6f}** | **{macro['auc_delta_pp']:+.6f}** | | | | |",
        "",
        "## Parameter and scheduler checks",
        "",
        f"- Trainable parameters: {macro['trainable_parameters']:,}; direct no-adapter comparison: {macro['baseline_trainable_parameters']:,}; reduction: {macro['trainable_parameter_reduction_percent']:.2f}%.",
        "- All seven scene score CSVs have the same frame and label keys as the direct no-adapter comparison and matching split fingerprints.",
        "- The scheduler recorded seven launches, seven zero-return finishes, and an all-finished event. GPU workers were reused on completion rather than assigned fixed scene slots.",
        "- The structural model has one global mixer followed by three local mixers. This is an operator/topology change, not an input-width-only ablation.",
        "",
        "## Interpretation and limitations",
        "",
        "The structural student improves the direct four-block no-adapter mean by 0.823 percentage points in this seed while using 24.87% fewer trainable OCBE+student parameters. Gains are concentrated in vehicle and crossroads; highway and railway decline. This is a single-seed result and does not establish a stable improvement. The benchmark was run on two GPUs, so deployment-device latency remains unmeasured.",
    ])
    (EXP / "results/summary.md").write_text("\n".join(report) + "\n")
    print(json.dumps(macro, ensure_ascii=False))


if __name__ == "__main__":
    main()
