"""Train one independent depth-retention Q for each three-layer teacher group."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from model import build_appearance_teacher
from model.bottleneck_depthrank_group_learnable import build_group_learnable_depthrank2_bottleneck
from model.student_structural import FourBlockGlobalLocalDecoder
from scripts.run_q_depthretain import DEFAULT_PROBE, Frames, SCENES, dataset_manifest, evaluate, loss_for


def depth_group_grams_from_probe(path: Path, scene: str, data_root: Path):
    probe = json.loads(path.read_text())
    item = probe["scenes"][scene]
    if probe["frames_per_scene"] != 16 or len(item["groups"]) != 4:
        raise ValueError("unexpected teacher depth probe format")
    for relative in item["sample_paths"]:
        if not (data_root / relative).is_file() or "/train/" not in f"/{relative}":
            raise FileNotFoundError(f"probe normal frame missing: {relative}")
    for group, data in enumerate(item["groups"]):
        if data["teacher_layers"] != list(range(group * 3, group * 3 + 3)):
            raise ValueError(f"unexpected teacher layers in group {group}")
    grams = np.asarray([group["depth_gram"] for group in item["groups"]], dtype=np.float64)
    if grams.shape != (4, 3, 3) or not np.allclose(grams, grams.transpose(0, 2, 1), atol=1e-5):
        raise ValueError("invalid group Gram matrices")
    eigenvalues = np.linalg.eigvalsh(grams)
    if np.any(eigenvalues[:, 1] - eigenvalues[:, 0] <= 0.01):
        raise ValueError(f"group depth optimum is not identifiable for {scene}")
    return grams, hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def batch_depth_group_grams(taps):
    if len(taps) != 12:
        raise ValueError(f"expected 12 teacher taps, got {len(taps)}")
    groups = []
    for start in range(0, 12, 3):
        triplet = torch.stack([
            torch.nn.functional.layer_norm(tap, (tap.shape[-1],))
            for tap in taps[start:start + 3]
        ]).reshape(3, -1)
        groups.append((triplet @ triplet.T) / triplet.shape[1])
    return torch.stack(groups)


def retention_loss(ocbe, grams):
    q = ocbe.effective_depth_basis()
    retained = torch.einsum("gai,gij,gaj->", q, grams, q)
    total = torch.diagonal(grams, dim1=-2, dim2=-1).sum()
    return 1 - retained / total


@torch.no_grad()
def basis_snapshot(ocbe, epoch, grams, target, train_loss=None):
    raw = ocbe.raw_depth_basis.detach().float().cpu()
    q = ocbe.effective_depth_basis().detach().double().cpu()
    grams = grams.detach().double().cpu()
    target = target.detach().double().cpu()
    fixed = torch.tensor([[1 / math.sqrt(3)] * 3,
                          [-1 / math.sqrt(2), 0, 1 / math.sqrt(2)]], dtype=torch.float64)
    group_angles = []
    fixed_angles = []
    group_energy = []
    ortho_error = []
    for group in range(4):
        normal = torch.nn.functional.normalize(torch.linalg.cross(q[group, 0], q[group, 1]), dim=0)
        optimum = torch.nn.functional.normalize(torch.linalg.cross(target[group, 0], target[group, 1]), dim=0)
        fixed_normal = torch.nn.functional.normalize(torch.linalg.cross(fixed[0], fixed[1]), dim=0)
        group_angles.append(math.degrees(math.acos(float(torch.clamp(torch.abs(normal @ optimum), 0, 1)))))
        fixed_angles.append(math.degrees(math.acos(float(torch.clamp(torch.abs(normal @ fixed_normal), 0, 1)))))
        group_energy.append(float(torch.trace(q[group] @ grams[group] @ q[group].T) / torch.trace(grams[group])))
        ortho_error.append(float(torch.linalg.matrix_norm(q[group] @ q[group].T - torch.eye(2))))
    return {
        "epoch": epoch, "train_loss": train_loss,
        "raw_q": raw.tolist(), "effective_q": q.tolist(),
        "group_angle_to_retention_optimum_deg": group_angles,
        "group_angle_to_fixed_mean_trend_deg": fixed_angles,
        "group_retained_normal_energy_fraction": group_energy,
        "combined_retained_normal_energy_fraction": float(
            sum(torch.trace(q[group] @ grams[group] @ q[group].T) for group in range(4))
            / sum(torch.trace(grams[group]) for group in range(4))
        ),
        "group_effective_orthogonality_error_fro": ortho_error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=SCENES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get(
        "UAV_DA_ROOT", "/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly")))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--basis-init", choices=("mean_trend", "random_orthonormal"), required=True)
    parser.add_argument("--probe-json", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--q-pretrain-steps", type=int, default=300)
    parser.add_argument("--q-pretrain-lr", type=float, default=0.01)
    parser.add_argument("--retain-weight", type=float, default=1.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this experiment")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{args.gpu}")
    scene_dir = args.data_root / SCENES[args.scene]
    train, videos, fingerprint = dataset_manifest(scene_dir)
    probe_array, probe_hash = depth_group_grams_from_probe(args.probe_json, args.scene, args.data_root)
    teacher = build_appearance_teacher(layers=list(range(12))).to(device).eval()
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(),
                    T.Normalize(teacher.img_mean, teacher.img_std)])
    loader = DataLoader(Frames(train, tf), batch_size=args.batch, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed),
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    ocbe = build_group_learnable_depthrank2_bottleneck(
        n_taps=12, code_dim=64, fuse_hidden=128, use_tap_norm=True,
        basis_init=args.basis_init, basis_seed=args.seed).to(device)
    student = FourBlockGlobalLocalDecoder(dim=192, teacher_dim=768).to(device)
    probe_grams = torch.tensor(probe_array, device=device, dtype=torch.float32)
    eigenvalues, eigenvectors = torch.linalg.eigh(probe_grams.double())
    probe_target = eigenvectors[:, :, 1:].transpose(1, 2).float()
    out = args.output / "results"
    out.mkdir(parents=True, exist_ok=True)
    pretrain_history = [basis_snapshot(ocbe, 0, probe_grams, probe_target)]
    q_optimizer = torch.optim.Adam([ocbe.raw_depth_basis], lr=args.q_pretrain_lr)
    for step in range(1, args.q_pretrain_steps + 1):
        objective = retention_loss(ocbe, probe_grams)
        q_optimizer.zero_grad(); objective.backward(); q_optimizer.step()
        if step in {25, 50, 100, args.q_pretrain_steps}:
            pretrain_history.append(basis_snapshot(ocbe, step, probe_grams, probe_target))
    pretrain = {
        "scene": args.scene, "basis_init": args.basis_init, "basis_seed": args.seed,
        "probe_path": str(args.probe_json.resolve()), "probe_sha256": probe_hash,
        "normal_frame_count": 16, "depth_gram_four_groups": probe_array.tolist(),
        "eigenvalues_ascending_by_group": eigenvalues.cpu().tolist(),
        "target_plane_basis_by_group": probe_target.cpu().tolist(),
        "q_pretrain_steps": args.q_pretrain_steps, "q_pretrain_lr": args.q_pretrain_lr,
        "history": pretrain_history,
    }
    (out / f"{args.scene}_pretrain.json").write_text(json.dumps(pretrain, indent=2) + "\n")
    print(json.dumps({"scene": args.scene, "phase": "q_pretrain_done", "basis_init": args.basis_init,
                      "initial_group_angles_deg": pretrain_history[0]["group_angle_to_retention_optimum_deg"],
                      "final_group_angles_deg": pretrain_history[-1]["group_angle_to_retention_optimum_deg"]}), flush=True)
    params = list(ocbe.parameters()) + list(student.parameters())
    optimizer = torch.optim.AdamW(params, lr=5e-4, weight_decay=5e-2)
    steps_per_epoch = min(len(loader), args.max_steps) if args.max_steps else len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4,
                                                    steps_per_epoch=steps_per_epoch, epochs=args.epochs)
    print(json.dumps({"scene": args.scene, "architecture": "ocbe_group4_depthrank2_retention_h128_glll_width192",
                      "seed": args.seed, "basis_init": args.basis_init, "retain_weight": args.retain_weight,
                      "train_frames": len(train), "test_videos": len(videos),
                      "split_fingerprint": fingerprint,
                      "trainable_parameters": sum(p.numel() for p in params),
                      "cuda": torch.cuda.get_device_name(device)}, ensure_ascii=False), flush=True)
    started = time.monotonic()
    basis_history = [basis_snapshot(ocbe, 0, probe_grams, probe_target)]
    last_epoch_grams = None
    for epoch in range(args.epochs):
        ocbe.train(); student.train()
        total = distill_total = retain_total = count = 0
        epoch_gram_sum = torch.zeros_like(probe_grams)
        for step, images in enumerate(loader):
            if args.max_steps and step >= args.max_steps:
                break
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                taps = teacher(images)["patches"]
                group_grams = batch_depth_group_grams(taps)
            reconstructed = student(ocbe(taps))
            distill = loss_for(taps, reconstructed)
            retained = retention_loss(ocbe, group_grams)
            loss = distill + args.retain_weight * retained
            optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
            n = len(images)
            total += loss.item() * n; distill_total += distill.item() * n
            retain_total += retained.item() * n; count += n
            epoch_gram_sum += group_grams * n
        last_epoch_grams = epoch_gram_sum / count
        _, epoch_vectors = torch.linalg.eigh(last_epoch_grams.double())
        epoch_target = epoch_vectors[:, :, 1:].transpose(1, 2).float()
        snapshot = basis_snapshot(ocbe, epoch + 1, last_epoch_grams, epoch_target, total / count)
        snapshot["distill_train_loss"] = distill_total / count
        snapshot["retention_loss"] = retain_total / count
        basis_history.append(snapshot)
        print(f"epoch={epoch+1} loss={total/count:.6f} distill={distill_total/count:.6f} "
              f"retention={retain_total/count:.6f} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    ocbe.eval(); student.eval()
    auc, n_frames, positives = evaluate(teacher, ocbe, student, videos, tf, device,
                                        out / f"{args.scene}_scores.csv", 32)
    (out / f"{args.scene}_basis.json").write_text(json.dumps({
        "scene": args.scene, "seed": args.seed, "basis_initialization": args.basis_init,
        "basis_sharing": "four_independent_2x3_matrices",
        "retention_probe_sha256": probe_hash, "retention_weight": args.retain_weight,
        "joint_retention_gram": "four_per_batch_normal_teacher_layer_norm_group_grams",
        "final_train_group_grams": last_epoch_grams.cpu().tolist(),
        "orthogonality": "per_group_differentiable_gram_schmidt",
        "history": basis_history,
    }, indent=2) + "\n")
    final = basis_history[-1]
    metric = {
        "scene": args.scene, "architecture": "ocbe_group4_depthrank2_retention_h128_glll_width192",
        "seed": args.seed, "basis_init": args.basis_init,
        "auc_percent": auc, "frames": n_frames, "positive_frames": positives,
        "split_fingerprint": fingerprint,
        "trainable_parameters": sum(p.numel() for p in params),
        "group_angle_to_retention_optimum_deg": final["group_angle_to_retention_optimum_deg"],
        "group_retained_normal_energy_fraction": final["group_retained_normal_energy_fraction"],
        "combined_retained_normal_energy_fraction": final["combined_retained_normal_energy_fraction"],
        "retention_probe_sha256": probe_hash, "retention_weight": args.retain_weight,
        "elapsed_s": time.monotonic() - started, "max_steps": args.max_steps,
    }
    (out / f"{args.scene}.json").write_text(json.dumps(metric, indent=2) + "\n")
    print(json.dumps(metric), flush=True)


if __name__ == "__main__":
    main()
