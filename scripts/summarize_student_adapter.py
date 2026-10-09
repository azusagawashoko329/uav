"""Verify saved frame scores and summarize the 2026-10-09 adapter study."""
import csv
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1] / "experiments" / "da" / "aggregation"
SCENES = ("highway", "farmland", "bike", "solar", "vehicle", "crossroads", "railway")
VARIANTS = ("none", "channel", "spatial")


def main():
    values = {}
    for variant in VARIANTS:
        directory = ROOT / f"20261009_student_adapter_{variant}_s42"
        scene_rows = []
        for scene in SCENES:
            metric = json.loads((directory / "results" / f"{scene}.json").read_text())
            with (directory / "results" / f"{scene}_scores.csv").open(newline="") as f:
                scores = list(csv.DictReader(f))
            assert len(scores) == metric["frames"]
            assert sum(int(r["label"]) for r in scores) == metric["positive_frames"]
            auc = roc_auc_score([int(r["label"]) for r in scores],
                                [float(r["anomaly_score"]) for r in scores]) * 100
            assert abs(auc - metric["auc_percent"]) < 1e-8
            log = (directory / "logs" / f"{scene}_retry1.log").read_text()
            assert "epoch=15 " in log and '"auc_percent"' in log
            scene_rows.append(metric)
        values[variant] = scene_rows
        with (directory / "results" / "summary.csv").open("w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(("scene", "auc_percent", "frames", "positive_frames", "trainable_parameters"))
            for r in scene_rows:
                w.writerow((r["scene"], f'{r["auc_percent"]:.8f}', r["frames"],
                            r["positive_frames"], r["trainable_parameters"]))
            w.writerow(("macro_mean", f'{sum(r["auc_percent"] for r in scene_rows)/7:.8f}',
                        "", "", scene_rows[0]["trainable_parameters"]))

    # All three variants must have used the same scene-specific train/test inventory.
    for i in range(7):
        assert len({values[v][i]["split_fingerprint"] for v in VARIANTS}) == 1
        assert len({values[v][i]["frames"] for v in VARIANTS}) == 1
    with (ROOT / "20261009_student_adapter_comparison_s42.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(("scene", "none", "channel", "spatial", "channel_minus_none", "spatial_minus_none"))
        for i, scene in enumerate(SCENES):
            n, c, s = (values[v][i]["auc_percent"] for v in VARIANTS)
            w.writerow((scene, f"{n:.8f}", f"{c:.8f}", f"{s:.8f}", f"{c-n:.8f}", f"{s-n:.8f}"))
        means = [sum(r["auc_percent"] for r in values[v])/7 for v in VARIANTS]
        w.writerow(("macro_mean", *(f"{v:.8f}" for v in means),
                    f"{means[1]-means[0]:.8f}", f"{means[2]-means[0]:.8f}"))
    print("verified 21 scene metrics and score files")


if __name__ == "__main__":
    main()
