"""Feasibility probe: is "pack 3 adjacent teacher layers into 1 target" a real operation?

Training-free. Frozen DINOv3 teacher, normal train frames only — no student, no training, no
labels. It answers the questions that decide whether the packing design has any substance at all:

  Q1 组内三层互补性   per-patch cosine between the 3 members of each group.
                      接近 1.0 => 三层近似重复, 打包是个空操作, 整个设计就是空的。
                      0.6~0.9  => 相关但不同, 合并是一个真实的操作。
  Q2 均值丢了多少     ridge 回归: 从组均值线性解码回每个组员, held-out R^2。
                      高 => 均值已经把组员保住了, 更花哨的聚合器没有空间。
                      低 => 均值有损, "concat+MLP 太粗糙" 的担心成立。
  Q3 LN 是否承重      同样做均值但不做 per-layer LayerNorm。深层原始特征范数远大于浅层,
                      原样均值应当被最深的那个组员主导。
  Q4 目标锐度         合并目标的逐 patch 空间起伏 / 逐 patch 跨帧起伏, 与原始层对比。
                      太糊的目标会削弱逐 patch top-k 这个真正的检测读出。

Usage:
  python probe_group_agg.py --gpu 1 --scenes railway bike --frames 128
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '../../../../..')))
import argparse
import glob
import os

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import build_appearance_teacher

DATA_ROOT = os.environ.get('UAV_DA_ROOT', '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly')
SCENES = {
    'bike': 'Bike Roundabout', 'crossroads': 'Crossroads', 'farmland': 'Farmland Inspection',
    'highway': 'Highway', 'railway': 'Railway Inspection', 'solar': 'Solar Panel Inspection',
    'vehicle': 'Vehicle Roundabout',
}
GROUPS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
PAIRS = [(0, 1), (1, 2), (0, 2)]


class Frames(Dataset):
    """Same frame-loading contract as train.py (per-scene merge of all sequences)."""

    def __init__(self, scene_dir, img, mean, std, limit):
        paths = sorted(glob.glob(os.path.join(scene_dir, 'sequence*', 'train', '*', '*.jpg')))
        if not paths:
            raise FileNotFoundError(f'no train .jpg under {scene_dir}')
        step = max(1, len(paths) // limit)          # spread the sample over the whole scene
        self.paths = paths[::step][:limit]
        self.tf = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('RGB'))


def ridge_r2(x, y, fit_frac=0.8, lam_scale=1e-3):
    """Held-out R^2 of a ridge map x->y.  x,y: (N, D) with rows ordered by frame."""
    n_fit = int(x.shape[0] * fit_frac)
    xf, yf, xe, ye = x[:n_fit], y[:n_fit], x[n_fit:], y[n_fit:]
    xtx, xty = xf.T @ xf, xf.T @ yf
    lam = lam_scale * xtx.diagonal().mean().clamp_min(1e-12)
    eye = torch.eye(xtx.shape[0], device=x.device, dtype=x.dtype)
    w = torch.linalg.solve(xtx + lam * eye, xty)
    ss_res = (ye - xe @ w).pow(2).sum()
    ss_tot = (ye - ye.mean(0, keepdim=True)).pow(2).sum()
    return (1 - ss_res / ss_tot).item()


def cos_stats(a, b, n_patch):
    """Per-patch cosine between two (F*P, D) tensors -> (mean, p10, p50, p90)."""
    c = torch.cosine_similarity(a, b, dim=-1)
    q = torch.quantile(c, torch.tensor([0.1, 0.5, 0.9], device=c.device))
    return c.mean().item(), q.tolist()


def spread(t, n_frames, n_patch):
    """(F*P, D) -> (逐patch空间起伏, 逐patch跨帧起伏)."""
    v = t.reshape(n_frames, n_patch, -1)
    return v.std(dim=1).mean().item(), v.std(dim=0).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--scenes', nargs='+', default=['railway', 'bike'])
    ap.add_argument('--frames', type=int, default=128, help='normal train frames sampled per scene')
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--img', type=int, default=256)
    ap.add_argument('--fit_frac', type=float, default=0.8)
    a = ap.parse_args()

    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    teacher = build_appearance_teacher(layers=list(range(12))).to(dev).eval()
    ln = nn.LayerNorm(teacher.embed_dim).to(dev)        # == OCBE tap_norms at init
    print(f'teacher: {teacher.num_blocks} blocks, embed={teacher.embed_dim}, '
          f'prefix={teacher.num_prefix}, img={a.img}', flush=True)

    for scene_key in a.scenes:
        ds = Frames(os.path.join(DATA_ROOT, SCENES.get(scene_key, scene_key)),
                    a.img, teacher.img_mean, teacher.img_std, a.frames)
        dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers)
        ln_feat, raw_feat = [[] for _ in range(12)], [[] for _ in range(12)]
        n_seen, n_patch = 0, None
        with torch.no_grad():
            for x in dl:
                layers = teacher(x.to(dev))['patches']        # 12 x (B,P,768), raw (pre-final-norm)
                n_seen += layers[0].shape[0]
                n_patch = layers[0].shape[1]
                for i in range(12):
                    ln_feat[i].append(ln(layers[i]).reshape(-1, 768).float().cpu())
                    raw_feat[i].append(layers[i].reshape(-1, 768).float().cpu())
                del layers
        ln_feat = [torch.cat(v).to(dev) for v in ln_feat]
        raw_feat = [torch.cat(v).to(dev) for v in raw_feat]

        print(f'\n===== scene={scene_key}  frames={n_seen}  patches/frame={n_patch}  '
              f'samples={n_seen * n_patch} =====')

        print('Q1 组内三层互补性  per-patch cos (LN后特征):')
        for g, idx in enumerate(GROUPS):
            txt = []
            for i, j in PAIRS:
                m, q = cos_stats(ln_feat[idx[i]], ln_feat[idx[j]], n_patch)
                txt.append(f'L{idx[i]}-L{idx[j]}: {m:.3f} (p10/p50/p90 {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f})')
            print(f'   G{g} {idx}: ' + ' | '.join(txt))

        print('Q2 均值对组员的保真  (均值 -> 组员 的 held-out R^2, 1.0 = 完全保真):')
        for g, idx in enumerate(GROUPS):
            mem = torch.stack([ln_feat[i] for i in idx])           # (3, N, 768)
            raw = torch.stack([raw_feat[i] for i in idx])
            m_ln, m_raw = mem.mean(0), raw.mean(0)
            r_ln = [ridge_r2(m_ln, mem[j], a.fit_frac) for j in range(3)]
            r_raw = [ridge_r2(m_raw, raw[j], a.fit_frac) for j in range(3)]
            c_ln = [cos_stats(m_ln, mem[j], n_patch)[0] for j in range(3)]
            c_raw = [cos_stats(m_raw, raw[j], n_patch)[0] for j in range(3)]
            print(f'   G{g} {idx}:  R^2(LN均值)={" ".join(f"{v:.3f}" for v in r_ln)}  '
                  f'R^2(原样均值)={" ".join(f"{v:.3f}" for v in r_raw)}')
            print(f'            cos(均值, 组员): LN={" ".join(f"{v:.3f}" for v in c_ln)}  '
                  f'原样={" ".join(f"{v:.3f}" for v in c_raw)}')

        print('Q4 目标锐度  (空间起伏 = 帧内跨patch std; 跨帧起伏 = 同patch跨帧 std):')
        for g, idx in enumerate(GROUPS):
            mem = torch.stack([ln_feat[i] for i in idx])
            sp_m, tp_m = spread(mem.mean(0), n_seen, n_patch)
            per = [spread(ln_feat[i], n_seen, n_patch) for i in idx]
            print(f'   G{g}: 合并目标 空间={sp_m:.4f} 跨帧={tp_m:.4f}   |   '
                  f'组员 空间={"/".join(f"{p[0]:.4f}" for p in per)}  跨帧={"/".join(f"{p[1]:.4f}" for p in per)}')


if __name__ == '__main__':
    main()
