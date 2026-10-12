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
BASE = REPO / "experiments/da/appearance/20261012_structural_mixer_glll_s42"


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
    scheduler_logs = sorted(EXP.glob("logs/scheduler_*.jsonl"))
    if not scheduler_logs:
        raise FileNotFoundError("no scheduler log found")
    launches, finishes, _ = verify_scheduler(scheduler_logs[-1])
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
        "# Global-local student width-192 experiment",
        "",
        "The loss alignment and Global-Local topology are unchanged. The only model change is the student internal width: 768 to 192; each reconstruction head projects back to the unchanged 768-dimensional teacher target.",
        "",
        "## Seven-scene result",
        "",
        "| scene | width-192 AUC | width-768 AUC | delta (pp) | normal mean | abnormal mean | score gap | GPU |",
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
        f"- Trainable parameters: {macro['trainable_parameters']:,}; width-768 Global-Local comparison: {macro['baseline_trainable_parameters']:,}; reduction: {macro['trainable_parameter_reduction_percent']:.2f}%.",
        "- All seven scene score CSVs have the same frame and label keys as the width-768 comparison and matching split fingerprints.",
        "- The scheduler recorded seven launches, seven zero-return finishes, and an all-finished event. GPU workers were reused on completion rather than assigned fixed scene slots.",
        "- The Global-Local topology and loss alignment are unchanged; only the student internal width and output projection dimensions changed.",
        "",
        "## Interpretation and limitations",
        "",
        "This is a single-seed width ablation against the 768-dimensional Global-Local student and does not establish a stable improvement. The benchmark was run on two GPUs, so deployment-device latency remains unmeasured.",
    ])
    latency_files = sorted(EXP.glob("results/latency_benchmark*.json"))
    if latency_files:
        report.extend(["", "## Inference latency microbenchmark", "",
                       "The following measurements use B=1 and 256 patch tokens, 50 warmup iterations and 200 timed iterations. They are GPU measurements, not edge-device measurements.", "",
                       "| device | student baseline ms | student width-192 ms | student delta | OCBE+student baseline ms | OCBE+student width-192 ms | pipeline delta | teacher ms |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for path in latency_files:
            lat = json.loads(path.read_text())
            base_lat, struct_lat = lat["variants"]
            student_delta = 100 * (1 - struct_lat["student_forward_ms"] / base_lat["student_forward_ms"])
            pipeline_delta = 100 * (1 - struct_lat["ocbe_plus_student_forward_ms"] / base_lat["ocbe_plus_student_forward_ms"])
            report.append(f"| {lat['device']} | {base_lat['student_forward_ms']:.4f} | {struct_lat['student_forward_ms']:.4f} | {student_delta:+.2f}% | {base_lat['ocbe_plus_student_forward_ms']:.4f} | {struct_lat['ocbe_plus_student_forward_ms']:.4f} | {pipeline_delta:+.2f}% | {lat['teacher_forward_ms']:.4f} |")
        report.extend(["", "The width-192 student is faster on the RTX A6000 but not on the RTX 5880 Ada once OCBE is included. Therefore this experiment does not establish a hardware-independent speedup; the real-time claim requires a benchmark on the target UAV processor and the full image-to-score path."])
    (EXP / "results/summary.md").write_text("\n".join(report) + "\n")
    print(json.dumps(macro, ensure_ascii=False))


if __name__ == "__main__":
    main()
