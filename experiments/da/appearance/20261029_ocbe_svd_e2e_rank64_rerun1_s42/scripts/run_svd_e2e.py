"""Train the LCDB low-rank cross-depth bottleneck for one scene."""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from model import anomaly_readouts, build_appearance_teacher
from model.student_structural import FourBlockGlobalLocalDecoder
from model.bottleneck_rank2_svd import build_end_to_end_factorized_rank2_bottleneck

SCENES = {
    "bike": "Bike Roundabout", "crossroads": "Crossroads",
    "farmland": "Farmland Inspection", "highway": "Highway",
    "railway": "Railway Inspection", "solar": "Solar Panel Inspection",
    "vehicle": "Vehicle Roundabout",
}


class Frames(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


def frame_key(path):
    return int(Path(path).stem)


def dataset_manifest(scene_dir):
    train = sorted(glob.glob(str(scene_dir / "sequence*" / "train" / "*" / "*.jpg")))
    test = sorted(Path(p) for p in glob.glob(str(scene_dir / "sequence*" / "test" / "*"))
                  if Path(p).is_dir())
    if not train or not test:
        raise FileNotFoundError(f"missing train/test frames in {scene_dir}")
    lines = [f"{Path(p).relative_to(scene_dir)}:{Path(p).stat().st_size}" for p in train]
    for video in test:
        frames = sorted(video.glob("*.jpg"), key=frame_key)
        label = Path(str(video) + ".npy")
        if not frames or not label.is_file():
            raise FileNotFoundError(f"missing frames or labels: {video}")
        lines.extend(f"{p.relative_to(scene_dir)}:{p.stat().st_size}" for p in frames)
        lines.append(f"{label.relative_to(scene_dir)}:{hashlib.sha256(label.read_bytes()).hexdigest()}")
    return train, test, hashlib.sha256("\n".join(lines).encode()).hexdigest()


def loss_for(taps, recons, k=5, alpha=0.5, mode="group_balanced"):
    per = []
    for a, b in zip(taps, recons):
        d = 1 - torch.nn.functional.cosine_similarity(a, b, dim=-1)
        per.append(d.topk(k, dim=-1).values.mean(-1))
    groups = torch.stack(per, dim=1).reshape(-1, 4, 3)
    avg, worst = groups.mean(-1), groups.max(-1).values
    group_loss = (avg + alpha * (worst - avg)).mean()
    if mode == "group_balanced":
        return group_loss
    if mode == "score_aligned":
        frame_mean = torch.stack(per, dim=1).mean()
        return 0.75 * group_loss + 0.25 * frame_mean
    raise ValueError(f"unknown loss mode: {mode}")


@torch.no_grad()
def basis_snapshot(ocbe, epoch, train_loss=None):
    raw = ocbe.raw_depth_basis.detach().float().cpu()
    q = ocbe.effective_depth_basis().detach().double().cpu()
    fixed = torch.tensor([[1 / math.sqrt(3)] * 3,
                          [-1 / math.sqrt(2), 0.0, 1 / math.sqrt(2)]], dtype=torch.float64)
    p, p_fixed = q.T @ q, fixed.T @ fixed
    normal = torch.nn.functional.normalize(torch.linalg.cross(q[0], q[1]), dim=0)
    fixed_normal = torch.nn.functional.normalize(torch.linalg.cross(fixed[0], fixed[1]), dim=0)
    overlap = torch.clamp(torch.abs(torch.dot(normal, fixed_normal)), 0.0, 1.0)
    return {"epoch": epoch, "train_loss": train_loss,
            "raw_q": raw.tolist(), "effective_q": q.tolist(),
            "principal_angle_deg": math.degrees(math.acos(float(overlap))),
            "projector_distance_fro": float(torch.linalg.matrix_norm(p - p_fixed)),
            "effective_orthogonality_error_fro": float(torch.linalg.matrix_norm(q @ q.T - torch.eye(2))),
            "raw_singular_values": torch.linalg.svdvals(raw).tolist(),
            "effective_singular_values": torch.linalg.svdvals(q).tolist()}


@torch.no_grad()
def evaluate(teacher, ocbe, student, videos, tf, device, out_csv, batch):
    rows, scores, labels = [], [], []
    for video in videos:
        paths = sorted(video.glob("*.jpg"), key=frame_key)
        y = np.asarray(np.load(str(video) + ".npy", allow_pickle=True)).astype(int).ravel()
        if len(y) != len(paths):
            raise ValueError(f"label/frame count mismatch: {video}: {len(y)} vs {len(paths)}")
        loader = DataLoader(Frames(paths, tf), batch_size=batch, shuffle=False, num_workers=4)
        vid_scores = []
        for xb in loader:
            xb = xb.to(device)
            taps = teacher(xb)["patches"]
            rec = student(ocbe(taps))
            vid_scores.extend(anomaly_readouts(taps, rec, k=5)["topk"].cpu().tolist())
        for path, label, score in zip(paths, y, vid_scores):
            rows.append((str(video.relative_to(video.parents[2])), path.name, int(label), score))
        scores.extend(vid_scores)
        labels.extend(y.tolist())
    auc = roc_auc_score(labels, scores) * 100
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "frame", "label", "anomaly_score"])
        writer.writerows(rows)
    return auc, len(labels), int(sum(labels))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, choices=SCENES)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--data-root", type=Path, default=Path(os.environ.get("UAV_DA_ROOT", "/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly")))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--variant", default="svd_e2e_rank64", choices=("svd_e2e_rank64",))
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--loss-mode", default="group_balanced", choices=("group_balanced", "score_aligned"))
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this experiment")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{a.gpu}")
    scene_dir = a.data_root / SCENES[a.scene]
    train, videos, fingerprint = dataset_manifest(scene_dir)
    teacher = build_appearance_teacher(layers=list(range(12))).to(device).eval()
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(),
                    T.Normalize(teacher.img_mean, teacher.img_std)])
    loader = DataLoader(Frames(train, tf), batch_size=a.batch, shuffle=True,
                        generator=torch.Generator().manual_seed(a.seed),
                        num_workers=a.workers, pin_memory=True, drop_last=True)
    # A new process/model/optimizer is used for every scene; no cross-scene state.
    ocbe = build_end_to_end_factorized_rank2_bottleneck(rank=a.rank, n_taps=12, code_dim=64, hidden=128, basis_init="random_orthonormal", basis_seed=a.seed).to(device)
    student = FourBlockGlobalLocalDecoder(dim=192, teacher_dim=768).to(device)
    params = list(ocbe.parameters()) + list(student.parameters())
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=5e-2)
    steps_per_epoch = min(len(loader), a.max_steps) if a.max_steps else len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4,
                                               steps_per_epoch=steps_per_epoch, epochs=a.epochs)
    print(json.dumps({"scene": a.scene, "architecture": f"ocbe_rank2_{a.variant}_r{a.rank}_randomorth_glll_width192", "seed": a.seed,
                      "train_frames": len(train), "test_videos": len(videos),
                      "split_fingerprint": fingerprint,
                      "trainable_parameters": sum(p.numel() for p in params),
                      "cuda": torch.cuda.get_device_name(device)}, ensure_ascii=False), flush=True)
    t0 = time.monotonic()
    basis_history = [basis_snapshot(ocbe, epoch=0)]
    for ep in range(a.epochs):
        ocbe.train(); student.train()
        total = count = 0
        for step, xb in enumerate(loader):
            if a.max_steps and step >= a.max_steps:
                break
            xb = xb.to(device, non_blocking=True)
            with torch.no_grad():
                taps = teacher(xb)["patches"]
            rec = student(ocbe(taps))
            loss = loss_for(taps, rec, mode=a.loss_mode)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += loss.item() * len(xb); count += len(xb)
        print(f"epoch={ep+1} loss={total/count:.6f} elapsed_s={time.monotonic()-t0:.1f}", flush=True)
        basis_history.append(basis_snapshot(ocbe, epoch=ep + 1, train_loss=total / count))
    ocbe.eval(); student.eval()
    out = a.output / "results"
    auc, n, pos = evaluate(teacher, ocbe, student, videos, tf, device,
                           out / f"{a.scene}_scores.csv", 32)
    checkpoint_path = a.output / "ckpt" / f"{a.scene}_final.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "scene": a.scene,
        "variant": a.variant,
        "loss_mode": a.loss_mode,
        "seed": a.seed,
        "split_fingerprint": fingerprint,
        "trainable_parameters": sum(p.numel() for p in params),
        "ocbe_state_dict": {k: v.detach().cpu() for k, v in ocbe.state_dict().items()},
        "student_state_dict": {k: v.detach().cpu() for k, v in student.state_dict().items()},
        "basis_snapshot": basis_history[-1],
        "fusion_diagnostics": getattr(ocbe, "diagnostics", lambda: {})(),
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    (out / f"{a.scene}_basis.json").write_text(json.dumps({
        "scene": a.scene, "seed": a.seed,
        "basis_sharing": "one_2x3_matrix_across_four_groups",
        "basis_initialization": "seeded_random_orthonormal",
        "basis_seed": a.seed,
        "orthogonality": "differentiable_gram_schmidt",
        "variant": a.variant,
        "diagnostics": getattr(ocbe, "diagnostics", lambda: {})(),
        "history": basis_history,
    }, indent=2) + "\n")
    metric = {"scene": a.scene, "architecture": f"ocbe_rank2_{a.variant}_r{a.rank}_{a.loss_mode}_randomorth_glll_width192", "variant": a.variant, "loss_mode": a.loss_mode, "seed": a.seed,
              "auc_percent": auc, "frames": n, "positive_frames": pos,
              "split_fingerprint": fingerprint,
              "trainable_parameters": sum(p.numel() for p in params),
              "basis_principal_angle_deg": basis_history[-1]["principal_angle_deg"],
              "basis_projector_distance_fro": basis_history[-1]["projector_distance_fro"],
              "basis_orthogonality_error_fro": basis_history[-1]["effective_orthogonality_error_fro"],
              "fusion_diagnostics": getattr(ocbe, "diagnostics", lambda: {})(),
              "checkpoint": str(checkpoint_path.relative_to(a.output)),
              "checkpoint_sha256": checkpoint_sha256,
              "elapsed_s": time.monotonic() - t0, "max_steps": a.max_steps}
    (out / f"{a.scene}.json").write_text(json.dumps(metric, indent=2) + "\n")
    print(json.dumps(metric), flush=True)


if __name__ == "__main__":
    main()
