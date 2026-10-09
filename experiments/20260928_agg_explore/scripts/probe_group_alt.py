"""Probe 3: can a DESIGNED (frozen, label-free) aggregation beat the plain mean?

probe_group_gap.py showed mean aggregation costs 28-40% of the target's anomaly displacement in
3 of 4 groups, because it weights all 3 members equally even when one is nearly blind
(L11 gap 0.07 vs L9 0.53).  A *learned* aggregator cannot fix that: the reconstruction loss only
sees normal frames, so its gradient pushes any trainable target toward "easier to fit" = smoother
= less discriminative.  So the aggregator has to be frozen, and the question becomes whether a
*designed* frozen aggregation can recover the cost.

Compared per group — all frozen, no gradient training, no labels (pseudo-anomalies only):

  mean   组内 per-layer LN 后等权平均            (当前设计)
  gapw   按逐层敏感度加权 —— 权重在 fit 帧上量出、在 eval 帧上使用, 不循环
  diff   t_last - t_first —— 层间差而不是平均。平均会把"某一层没变"这个信息抹掉,
         而一层灵敏/一层不变时, 差值在扰动下反而动得更大
  best   敏感度最高的那个组员 (在 fit 帧上选出) —— 上界参考, 不是可用设计

Usage:  python probe_group_alt.py --gpu 0 --scenes railway bike --frames 96
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import argparse
import glob
import os

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import build_appearance_teacher

DATA_ROOT = '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly'
SCENES = {
    'bike': 'Bike Roundabout', 'crossroads': 'Crossroads', 'farmland': 'Farmland Inspection',
    'highway': 'Highway', 'railway': 'Railway Inspection', 'solar': 'Solar Panel Inspection',
    'vehicle': 'Vehicle Roundabout',
}
GROUPS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
PATCH = 16


class Frames(Dataset):
    def __init__(self, scene_dir, img, mean, std, limit):
        paths = sorted(glob.glob(os.path.join(scene_dir, 'sequence*', 'train', '*', '*.jpg')))
        if not paths:
            raise FileNotFoundError(f'no train .jpg under {scene_dir}')
        step = max(1, len(paths) // limit)
        self.paths = paths[::step][:limit]
        self.tf = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('RGB'))


def cutpaste(x, size, gen):
    b, _, h, w = x.shape
    n = h // PATCH
    out = x.clone()
    mask = torch.zeros(b, n * n, dtype=torch.bool, device=x.device)
    for i in range(b):
        sy = int(torch.randint(0, h - size + 1, (1,), generator=gen))
        sx = int(torch.randint(0, w - size + 1, (1,), generator=gen))
        dy = int(torch.randint(0, h - size + 1, (1,), generator=gen))
        dx = int(torch.randint(0, w - size + 1, (1,), generator=gen))
        out[i, :, dy:dy + size, dx:dx + size] = x[i, :, sy:sy + size, sx:sx + size]
        for r in range(dy // PATCH, (dy + size - 1) // PATCH + 1):
            for c in range(dx // PATCH, (dx + size - 1) // PATCH + 1):
                mask[i, r * n + c] = True
    return out, mask


def gap(a, b):
    return (1.0 - torch.cosine_similarity(a, b, dim=-1)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--scenes', nargs='+', default=['railway', 'bike'])
    ap.add_argument('--frames', type=int, default=96)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--img', type=int, default=256)
    ap.add_argument('--paste', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    gen = torch.Generator().manual_seed(a.seed)
    teacher = build_appearance_teacher(layers=list(range(12))).to(dev).eval()
    ln = nn.LayerNorm(teacher.embed_dim).to(dev)
    print(f'teacher 12 blocks | CutPaste {a.paste}px | frames={a.frames}/scene '
          f'(fit half -> weights, eval half -> report)', flush=True)

    for scene_key in a.scenes:
        ds = Frames(os.path.join(DATA_ROOT, SCENES.get(scene_key, scene_key)),
                    a.img, teacher.img_mean, teacher.img_std, a.frames)
        dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers)
        n_fit = max(1, len(ds) // 2)

        def batches(want_fit):
            """Yield (ln_normal, ln_pseudo, mask) for the fit or the eval half (deterministic split)."""
            seen = 0
            for x in dl:
                x = x.to(dev)
                lo, hi = seen, seen + x.shape[0]
                seen = hi
                if (lo < n_fit) != want_fit:
                    continue
                g = torch.Generator().manual_seed(a.seed + lo)      # per-batch reproducibility
                xp, mask = cutpaste(x, a.paste, g)
                fn, fp = teacher(x)['patches'], teacher(xp)['patches']
                yield [ln(t) for t in fn], [ln(t) for t in fp], mask

        # --- pass 1: measure per-layer sensitivity on the fit half ---
        fit_sum, fit_n = torch.zeros(12, device=dev), 0
        with torch.no_grad():
            for ln_n, ln_p, _ in batches(True):
                for i in range(12):
                    fit_sum[i] += (1.0 - torch.cosine_similarity(ln_n[i], ln_p[i], dim=-1)).mean()
                fit_n += 1
        fit_gap = (fit_sum / max(fit_n, 1))
        w_all = {}
        for gi, idx in enumerate(GROUPS):
            t = fit_gap[list(idx)]
            w_all[gi] = t / t.sum()                    # 权重只在 fit 帧上量, eval 帧上不回头调

        # --- pass 2: report on the eval half, with the frozen weights ---
        ev = {(gi, k): [] for gi in range(len(GROUPS)) for k in ('mean', 'gapw', 'diff', 'best')}
        layer_eval = [[] for _ in range(12)]
        with torch.no_grad():
            for ln_n, ln_p, mask in batches(False):
                for i in range(12):
                    layer_eval[i].append(
                        (1.0 - torch.cosine_similarity(ln_n[i], ln_p[i], dim=-1))[mask].mean().item())
                for gi, idx in enumerate(GROUPS):
                    w = w_all[gi]
                    n_ = torch.stack([ln_n[i] for i in idx])
                    p_ = torch.stack([ln_p[i] for i in idx])
                    b = int(w.argmax())
                    for key, (cn, cp) in {
                        'mean': (n_.mean(0), p_.mean(0)),
                        'gapw': ((w.view(3, 1, 1, 1) * n_).sum(0), (w.view(3, 1, 1, 1) * p_).sum(0)),
                        'diff': (n_[2] - n_[0], p_[2] - p_[0]),
                        'best': (n_[b], p_[b]),
                    }.items():
                        ev[(gi, key)].append(gap(cn[mask], cp[mask]))

        avg = lambda v: sum(v) / len(v)
        print(f'\n===== scene={scene_key}  fit={n_fit} eval={len(ds) - n_fit} frames  '
              f'(gap = 1-cos(正常,伪异常), 粘贴区) =====')
        print('   逐层 eval gap: ' + ' '.join(f'L{i}={avg(layer_eval[i]):.3f}' for i in range(12)))
        print('   组               mean    gapw     diff     best   | 逐层权重(由 fit 帧量出)')
        for gi, idx in enumerate(GROUPS):
            w = w_all[gi]
            print(f'   G{gi} {idx}   {avg(ev[(gi, "mean")]):.4f}  {avg(ev[(gi, "gapw")]):.4f}  '
                  f'{avg(ev[(gi, "diff")]):.4f}  {avg(ev[(gi, "best")]):.4f}  | ' +
                  ' '.join(f'L{i}={v:.2f}' for i, v in zip(idx, w.tolist())))


if __name__ == '__main__':
    main()
