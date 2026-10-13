"""Measure normal teacher feature energy retained by a fixed depth-rank-two basis.

This diagnostic only reads evenly spaced normal training frames and does not use
anomaly labels, trainable models, or evaluation scores.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from model import build_appearance_teacher


SCENES = {
    "bike": "Bike Roundabout", "crossroads": "Crossroads",
    "farmland": "Farmland Inspection", "highway": "Highway",
    "railway": "Railway Inspection", "solar": "Solar Panel Inspection",
    "vehicle": "Vehicle Roundabout",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get(
        "UAV_DA_ROOT", "/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly")))
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]
                    / "results/depth_energy_probe.json")
    ap.add_argument("--frames-per-scene", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    if args.frames_per_scene < 2:
        raise ValueError("at least two frames per scene are required")
    device = torch.device(f"cuda:{args.gpu}")
    teacher = build_appearance_teacher(layers=list(range(12))).to(device).eval()
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(),
                    T.Normalize(teacher.img_mean, teacher.img_std)])
    q = torch.tensor([[1, 1, 1], [-1, 0, 1]], dtype=torch.float32, device=device)
    q = F.normalize(q, dim=1)
    output = {
        "method": "channel-layer-normalized teacher patch outputs; fixed orthonormal depth mean/trend",
        "sampling": "evenly spaced normal training frames by sorted filename; no anomaly labels",
        "frames_per_scene": args.frames_per_scene,
        "scenes": {},
    }
    for scene, name in SCENES.items():
        paths = sorted(glob.glob(str(args.data_root / name / "sequence*" /
                                     "train" / "*" / "*.jpg")))
        if len(paths) < args.frames_per_scene:
            raise FileNotFoundError(f"too few normal training frames for {scene}")
        chosen = [paths[round((len(paths) - 1) * i / (args.frames_per_scene - 1))]
                  for i in range(args.frames_per_scene)]
        gram_sum = torch.zeros((4, 3, 3), dtype=torch.float64, device=device)
        denominator = torch.zeros(4, dtype=torch.float64, device=device)
        for start in range(0, len(chosen), args.batch):
            images = []
            for path in chosen[start:start + args.batch]:
                with Image.open(path) as im:
                    images.append(tf(im.convert("RGB")))
            with torch.no_grad():
                taps = teacher(torch.stack(images).to(device))["patches"]
                normalized = [F.layer_norm(t, (t.shape[-1],)) for t in taps]
                for group in range(4):
                    a = torch.stack(normalized[group * 3:group * 3 + 3])
                    a = a.reshape(3, -1).to(torch.float64)
                    gram_sum[group] += a @ a.T
                    denominator[group] += a.shape[1]
        groups = []
        for group in range(4):
            gram = (gram_sum[group] / denominator[group]).to(torch.float32)
            eigenvalues = torch.linalg.eigvalsh(gram).flip(0)
            trace = torch.trace(gram)
            fixed_energy = torch.trace(q @ gram @ q.T) / trace
            groups.append({
                "teacher_layers": [group * 3 + j for j in range(3)],
                "fixed_mean_trend_energy_fraction": float(fixed_energy),
                "optimal_rank1_energy_fraction": float(eigenvalues[0] / trace),
                "optimal_rank2_energy_fraction": float(eigenvalues[:2].sum() / trace),
                "depth_gram": gram.cpu().tolist(),
            })
        output["scenes"][scene] = {
            "sample_paths": [str(Path(p).relative_to(args.data_root)) for p in chosen],
            "groups": groups,
        }
        print(scene, [round(g["fixed_mean_trend_energy_fraction"], 4)
                      for g in groups], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
